use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

pub mod evolution;
pub mod evaluator;
pub mod reinforcement;
pub mod database;
pub mod sandbox;

const MODEL: &str = "gemini-3.5-flash-lite";
const API_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

/// Shared resources that all agents draw on, so they don't each open their own connection or duplicate state.
///
/// `agent_evaluations` maps agent name -> that agent's most recent
/// evaluation (sample code, performance, novelty). It's populated by
/// `evaluator::run` (Evaluator.md steps 23-27) and is the source of truth
/// for the rank positions `evaluator::compute_rank_positions` produces,
/// which `reinforcement::run` fetches per Reinforcement.md step 1.
/// Callers constructing `SharedTools` should initialize it as
/// `Arc::new(Mutex::new(HashMap::new()))`.
#[derive(Clone)]
pub struct SharedTools {
    pub db: Arc<Mutex<database::Connection>>,
    pub agent_evaluations: Arc<Mutex<HashMap<String, evaluator::AgentEvaluation>>>,
}

pub struct Agent {
    pub name: String,
    algorithm: String,
    api_key: String,
    client: reqwest::Client,
    contents: Vec<Value>,
    tools: Value,
    system_instruction: Value,
    shared: SharedTools,
}

impl Agent {
    /// `algorithm` (e.g. `"kmeans"`, `"svm"`) is the ML algorithm this
    /// agent is assigned to optimize — it decides which training/test
    /// data and code the `evolution`/`evaluator` tools fetch from the
    /// database, and is set here rather than left for the model to
    /// choose per tool call, so it can't be gotten wrong or omitted.
    pub fn new(name: impl Into<String>, algorithm: impl Into<String>, api_key: impl Into<String>, shared: SharedTools) -> Self {
        let name = name.into();
        let algorithm = algorithm.into();

        let tools = json!([{
            "functionDeclarations": [
                {
                    "name": "evolution",
                    "description": "Runs an evolutionary step over a population",
                    "parameters": {
                        "type": "object",
                        "properties": { "generations": { "type": "integer" } },
                        "required": ["generations"]
                    }
                },
                {
                    "name": "evaluator",
                    "description": "Scores a candidate solution",
                    "parameters": {
                        "type": "object",
                        "properties": { "candidate": { "type": "string" } },
                        "required": ["candidate"]
                    }
                },
                {
                    "name": "reinforcement",
                    "description": "Runs a reinforcement learning training step",
                    "parameters": {
                        "type": "object",
                        "properties": { "episodes": { "type": "integer" } },
                        "required": ["episodes"]
                    }
                },
                {
                    "name": "database",
                    "description": "Runs a query against the persistent store",
                    "parameters": {
                        "type": "object",
                        "properties": { "query": { "type": "string" } },
                        "required": ["query"]
                    }
                }
            ]
        }]);

        // Without this, the model has no reason to ever call the tools
        // above — Gemini's function calling is AUTO by default, meaning
        // it only fires when the model judges it's clearly warranted.
        // A bare user prompt plus generic tool schemas isn't enough
        // signal; this spells out the agent's role and instructs it to
        // actually invoke the tools rather than just describe a plan.
        let system_instruction = json!({
            "parts": [{
                "text": format!(
                    "You are {name}, one of several competing agents in an evolutionary \
                     code-improvement system, currently assigned to optimize the '{algorithm}' \
                     algorithm. You must actually IMPROVE code by calling your tools — do not \
                     just describe what you would do; call the functions.\n\n\
                     Your tools and the order they're normally used in:\n\
                     1. database — read existing code, training data, or test data \
                        (action: 'read'), or write new code/data (action: 'write'). \
                        Use this first to check what already exists, and to seed \
                        training/test data or a baseline implementation for '{algorithm}' \
                        if none exists yet.\n\
                     2. evolution — generates and scores candidate implementations of \
                        '{algorithm}' seeded from the database, and writes the best one \
                        back. Call this to produce an improved candidate.\n\
                     3. evaluator — actually runs your evolved candidate against real \
                        training/test data and scores its performance and novelty against \
                        the baseline and other agents' candidates.\n\
                     4. reinforcement — fetches your reward and rank position after \
                        evaluation, as feedback on how you're doing relative to the \
                        other agents.\n\n\
                     Call these tools whenever the task calls for producing, testing, \
                     or comparing code — do not simply answer in prose when a tool call \
                     would make real progress on the task."
                )
            }]
        });

        Self {
            name,
            algorithm,
            api_key: api_key.into(),
            // No timeout previously meant a network hiccup (or Gemini
            // just being slow) could hang forever with zero output or
            // error — indistinguishable from the whole program being
            // frozen. A generous but finite timeout turns that into a
            // visible error instead of silence.
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .expect("failed to build HTTP client"),
            contents: Vec::new(),
            tools,
            system_instruction,
            shared,
        }
    }

    /// Sends a user message, runs any tool calls Gemini requests, and
    /// returns the agent's final text reply. History persists across calls.
    pub async fn send(&mut self, message: &str) -> Result<String> {
        eprintln!("[{}] sending message, awaiting Gemini...", self.name);
        self.contents.push(json!({
            "role": "user",
            "parts": [{ "text": message }]
        }));

        loop {
            let candidate = self.step().await?;

            let parts = candidate["content"]["parts"]
                .as_array()
                .cloned()
                .unwrap_or_default();

            self.contents.push(candidate["content"].clone());

            let mut made_call = false;
            let mut final_text = String::new();

            for part in &parts {
                if let Some(fc) = part.get("functionCall") {
                    made_call = true;
                    let name = fc["name"].as_str().unwrap_or_default();
                    eprintln!("[{}] calling tool '{name}' with args: {}", self.name, fc["args"]);
                    let result = self.call_tool(name, &fc["args"]).await?;
                    eprintln!("[{}] tool '{name}' finished", self.name);

                    self.contents.push(json!({
                        "role": "user",
                        "parts": [{
                            "functionResponse": {
                                "name": name,
                                "response": { "result": result }
                            }
                        }]
                    }));
                } else if let Some(text) = part.get("text") {
                    final_text.push_str(text.as_str().unwrap_or_default());
                }
            }

            if !made_call {
                eprintln!("[{}] done, no further tool calls", self.name);
                return Ok(final_text);
            }

            eprintln!("[{}] awaiting Gemini again after tool result(s)...", self.name);
        }
    }

    /// Same as `send`, but takes an owned String — convenient when
    /// building a batch of futures from a loop.
    pub async fn send_owned(&mut self, message: String) -> Result<String> {
        self.send(&message).await
    }

    /// One round-trip to the Gemini API. Returns the first candidate.
    ///
    /// Retries automatically on a 429 RESOURCE_EXHAUSTED quota error,
    /// waiting for the server-suggested `retryDelay` (Google's free tier
    /// is easy to hit with several agents each making multiple calls per
    /// turn). Other errors are returned immediately.
    async fn step(&self) -> Result<Value> {
        let body = json!({
            "systemInstruction": self.system_instruction,
            "contents": self.contents,
            "tools": self.tools
        });

        let url = format!("{}/{}:generateContent?key={}", API_URL, MODEL, self.api_key);

        const MAX_RETRIES: u32 = 5;
        let mut attempt = 0;

        loop {
            let resp: Value = self.client.post(&url).json(&body).send().await?.json().await?;

            if let Some(candidate) = resp["candidates"][0].as_object() {
                return Ok(Value::Object(candidate.clone()));
            }

            let status = resp["error"]["status"].as_str().unwrap_or_default();
            if status == "RESOURCE_EXHAUSTED" && attempt < MAX_RETRIES {
                let retry_delay_secs = resp["error"]["details"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .find(|d| d["@type"] == "type.googleapis.com/google.rpc.RetryInfo")
                    .and_then(|d| d["retryDelay"].as_str())
                    .and_then(|s| s.trim_end_matches('s').parse::<f64>().ok())
                    .unwrap_or(10.0);

                attempt += 1;
                eprintln!(
                    "[{}] quota exceeded, retrying in {:.1}s (attempt {}/{MAX_RETRIES})",
                    self.name,
                    retry_delay_secs,
                    attempt
                );
                tokio::time::sleep(std::time::Duration::from_secs_f64(retry_delay_secs + 0.5)).await;
                continue;
            }

            return Err(anyhow!("no candidates in Gemini response: {resp}"));
        }
    }

    /// Dispatches a tool call by name. Stateless tools take just `args`;
    /// `database` also gets the shared, mutex-guarded connection.
    /// `evolution` additionally needs this agent's name and algorithm (to
    /// tag its DB write and fetch the right data), the API key + HTTP
    /// client (to generate samples), and the database (to fetch/write
    /// code). `evaluator` needs the shared tools, this agent's name, and
    /// its algorithm. `reinforcement` needs the shared tools and this
    /// agent's name.
    async fn call_tool(&self, name: &str, args: &Value) -> Result<String> {
        match name {
            "evolution" => {
                evolution::run(args, &self.name, &self.algorithm, &self.api_key, &self.client, &self.shared.db).await
            }
            "evaluator" => evaluator::run(args, &self.shared, &self.name, &self.algorithm).await,
            "reinforcement" => reinforcement::run(args, &self.shared, &self.name).await,
            "database" => {
                let mut conn = self.shared.db.lock().await;
                database::run(&mut conn, args)
            }
            _ => Ok(format!("no such tool: {name}")),
        }
    }
}

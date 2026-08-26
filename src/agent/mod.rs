use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

pub mod evolution;
pub mod evaluator;
pub mod reinforcement;
pub mod database;

const MODEL: &str = "gemini-3.6-flash";
const API_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

#[derive(Clone)]
pub struct SharedTools {
    pub db: Arc<Mutex<database::Connection>>,
    pub agent_evaluations: Arc<Mutex<HashMap<String, evaluator::AgentEvaluation>>>,
}

pub struct Agent {
    pub name: String,
    api_key: String,
    client: reqwest::Client,
    contents: Vec<Value>,
    tools: Value,
    system_instruction: Value,
    shared: SharedTools,
}

impl Agent {
    pub fn new(name: impl Into<String>, api_key: impl Into<String>, shared: SharedTools) -> Self {
        let name = name.into();

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

        let system_instruction = json!({
            "parts": [{
                "text": format!(
                    "You are {name}, one of several competing agents in an evolutionary \
                     code-improvement system. You are given a coding/ML task and must \
                     actually IMPROVE code by calling your tools — do not just describe \
                     what you would do; call the functions.\n\n\
                     Your tools and the order they're normally used in:\n\
                     1. database — read existing code, training data, or test data \
                        (action: 'read'), or write new code/data (action: 'write'). \
                        Use this first to check what already exists, and to seed \
                        training/test data or a baseline implementation if none exists yet.\n\
                     2. evolution — generates and scores candidate implementations seeded \
                        from the database, and writes the best one back. Call this to \
                        produce an improved candidate.\n\
                     3. evaluator — scores your evolved candidate's performance and \
                        novelty against the baseline and other agents' candidates.\n\
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
            api_key: api_key.into(),
            client: reqwest::Client::new(),
            contents: Vec::new(),
            tools,
            system_instruction,
            shared,
        }
    }

    pub async fn send(&mut self, message: &str) -> Result<String> {
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
                    let result = self.call_tool(name, &fc["args"]).await?;

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
                return Ok(final_text);
            }
        }
    }

    pub async fn send_owned(&mut self, message: String) -> Result<String> {
        self.send(&message).await
    }

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

    async fn call_tool(&self, name: &str, args: &Value) -> Result<String> {
        match name {
            "evolution" => {
                evolution::run(args, &self.name, &self.api_key, &self.client, &self.shared.db).await
            }
            "evaluator" => evaluator::run(args, &self.shared, &self.name).await,
            "reinforcement" => reinforcement::run(args, &self.shared, &self.name).await,
            "database" => {
                let mut conn = self.shared.db.lock().await;
                database::run(&mut conn, args)
            }
            _ => Ok(format!("no such tool: {name}")),
        }
    }
}

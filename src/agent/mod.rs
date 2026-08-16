use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;

pub mod evolution;
pub mod evaluator;
pub mod reinforcement;
pub mod database;

const MODEL: &str = "gemini-3.6-flash";
const API_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

/// Shared resources that all agents draw on, so they don't each open their own connection or duplicate state.
#[derive(Clone)]
pub struct SharedTools {
    pub db: Arc<Mutex<database::Connection>>,
}

pub struct Agent {
    pub name: String,
    api_key: String,
    client: reqwest::Client,
    contents: Vec<Value>,
    tools: Value,
    shared: SharedTools,
}

impl Agent {
    pub fn new(name: impl Into<String>, api_key: impl Into<String>, shared: SharedTools) -> Self {
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

        Self {
            name: name.into(),
            api_key: api_key.into(),
            client: reqwest::Client::new(),
            contents: Vec::new(),
            tools,
            shared,
        }
    }

    /// Sends a user message, runs any tool calls Gemini requests, and
    /// returns the agent's final text reply. History persists across calls.
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

    /// Same as `send`, but takes an owned String — convenient when
    /// building a batch of futures from a loop.
    pub async fn send_owned(&mut self, message: String) -> Result<String> {
        self.send(&message).await
    }

    /// One round-trip to the Gemini API. Returns the first candidate.
    async fn step(&self) -> Result<Value> {
        let body = json!({
            "contents": self.contents,
            "tools": self.tools
        });

        let url = format!("{}/{}:generateContent?key={}", API_URL, MODEL, self.api_key);

        let resp: Value = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await?
            .json()
            .await?;

        resp["candidates"][0]
            .as_object()
            .map(|_| resp["candidates"][0].clone())
            .ok_or_else(|| anyhow!("no candidates in Gemini response: {resp}"))
    }

    /// Dispatches a tool call by name. Stateless tools take just `args`;
    /// `database` also gets the shared, mutex-guarded connection.
    async fn call_tool(&self, name: &str, args: &Value) -> Result<String> {
        match name {
            "evolution" => evolution::run(args),
            "evaluator" => evaluator::run(args),
            "reinforcement" => reinforcement::run(args),
            "database" => {
                let mut conn = self.shared.db.lock().await;
                database::run(&mut conn, args)
            }
            _ => Ok(format!("no such tool: {name}")),
        }
    }
}
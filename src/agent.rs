//! One competing agent: a Gemini tool-calling loop.

use anyhow::Result;
use serde_json::{Value, json};

use crate::archive::Candidate;
use crate::evolution::{self, EvolveRequest, Operator, Shared};
use crate::gemini::{collect_text, truncate};
use crate::harness::extract_block;
use crate::prompts;

pub struct Agent {
    pub name: String,
    /// Plain-text notes from the previous round, the agent's only memory.
    pub notes: String,
    pub tool_calls_total: usize,
}

impl Agent {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            notes: String::new(),
            tool_calls_total: 0,
        }
    }

    /// A fresh conversation per round; ends when the model stops or the budget is spent.
    pub async fn run_round(
        &mut self,
        shared: &Shared,
        round: usize,
        prompt: String,
    ) -> Result<String> {
        let system = prompts::agent_system(shared, &self.name);
        let tools = tool_declarations();
        let budget = shared.cfg.max_tool_calls_per_round;
        let model = shared.cfg.gemini.agent_model.clone();

        let mut contents = vec![json!({ "role": "user", "parts": [{ "text": prompt }] })];
        let mut calls_this_round = 0usize;
        let mut last_text = String::new();

        loop {
            let content = shared
                .gemini
                .generate(&model, Some(&system), &contents, Some(&tools), 0.7)
                .await?;
            let parts = content["parts"].as_array().cloned().unwrap_or_default();
            contents.push(json!({ "role": "model", "parts": parts }));

            let mut responses = Vec::new();
            let mut text = String::new();
            for part in &parts {
                if let Some(fc) = part.get("functionCall") {
                    let name = fc["name"].as_str().unwrap_or_default().to_string();
                    let args = fc.get("args").cloned().unwrap_or(json!({}));
                    calls_this_round += 1;
                    self.tool_calls_total += 1;
                    println!(
                        "[{}] tool {name} {}",
                        self.name,
                        truncate(&args.to_string(), 160)
                    );
                    let mut response = self.handle_tool(shared, round, &name, &args).await;
                    let remaining = budget.saturating_sub(calls_this_round);
                    response["tool_calls_remaining"] = json!(remaining);
                    if remaining == 0 {
                        response["note"] = json!(
                            "Budget spent. Reply now with your plain-text notes for the next round."
                        );
                    }
                    responses.push(
                        json!({ "functionResponse": { "name": name, "response": response } }),
                    );
                } else if let Some(t) = part["text"].as_str() {
                    text.push_str(t);
                }
            }
            if !text.trim().is_empty() {
                last_text = text;
            }
            if responses.is_empty() {
                break;
            }
            contents.push(json!({ "role": "user", "parts": responses }));

            if calls_this_round >= budget {
                // Final turn without tools to collect notes.
                match shared
                    .gemini
                    .generate(&model, Some(&system), &contents, None, 0.7)
                    .await
                {
                    Ok(final_content) => {
                        let t = collect_text(&final_content);
                        if !t.trim().is_empty() {
                            last_text = t;
                        }
                    }
                    Err(e) => eprintln!("[{}] could not collect final notes: {e:#}", self.name),
                }
                break;
            }
        }

        self.notes = last_text.clone();
        Ok(last_text)
    }

    async fn handle_tool(&self, shared: &Shared, round: usize, name: &str, args: &Value) -> Value {
        match name {
            "evolve" => {
                let operator = Operator::parse(args["operator"].as_str().unwrap_or("mutate"));
                let default_n = shared.cfg.samples_per_evolve;
                let n_samples = args["n_samples"]
                    .as_u64()
                    .map(|n| n as usize)
                    .unwrap_or(default_n)
                    .clamp(1, default_n * 2);
                let guidance = args["guidance"]
                    .as_str()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());
                let req = EvolveRequest {
                    agent: self.name.clone(),
                    round,
                    operator,
                    n_samples,
                    guidance,
                };
                match evolution::evolve(shared, &req).await {
                    Ok(v) => v,
                    Err(e) => json!({ "error": format!("{e:#}") }),
                }
            }
            "submit_code" => {
                let Some(code) = args["code"].as_str() else {
                    return json!({ "error": "submit_code requires a `code` string" });
                };
                let block = extract_block(code);
                let note = args["note"].as_str().map(str::to_string);
                match evolution::evaluate_block(
                    shared,
                    &self.name,
                    round,
                    "manual",
                    vec!["agent".into()],
                    note,
                    block,
                )
                .await
                {
                    Ok(v) => v,
                    Err(e) => json!({ "error": format!("{e:#}") }),
                }
            }
            "archive" => {
                let top_n = args["top_n"].as_u64().unwrap_or(5).clamp(1, 20) as usize;
                let agent = args["agent"].as_str().filter(|a| !a.is_empty());
                let include_code = args["include_code"].as_bool().unwrap_or(false);
                let archive = shared.archive.lock().await;
                let top: Vec<Value> = archive
                    .top(top_n, agent)
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let mut s = c.summary();
                        if include_code && i == 0 {
                            s["code"] = json!(truncate(&c.block, 8000));
                        }
                        s
                    })
                    .collect();
                json!({
                    "archive_size": archive.len(),
                    "rejected_near_duplicates": archive.rejected_near_duplicates,
                    "top": top,
                })
            }
            "leaderboard" => {
                let archive = shared.archive.lock().await;
                let standings = archive.standings();
                let me = standings.iter().find(|s| s.agent == self.name);
                let best_rival = standings
                    .iter()
                    .filter(|s| s.agent != self.name)
                    .filter_map(|s| s.best_id.as_ref())
                    .filter_map(|id| archive.entries().iter().find(|c| &c.id == id))
                    .max_by(|a, b| {
                        a.score
                            .partial_cmp(&b.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                json!({
                    "standings": standings,
                    "your_position": me.map(|s| s.rank_position),
                    "your_reward_if_round_ended_now": me.map(|s| s.reward),
                    "best_rival": best_rival.map(|c| {
                        let mut s = Candidate::summary(c);
                        s["code"] = json!(truncate(&c.block, 8000));
                        s
                    }),
                })
            }
            other => json!({ "error": format!("unknown tool `{other}`") }),
        }
    }
}

fn tool_declarations() -> Value {
    json!([{
        "functionDeclarations": [
            {
                "name": "evolve",
                "description": "Generate and evaluate new candidates from parents sampled out of the shared archive. Returns real harness metrics for each sample.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operator": { "type": "string", "enum": ["mutate", "crossover"], "description": "mutate = improve one parent; crossover = combine two parents, one preferably from a rival" },
                        "n_samples": { "type": "integer", "description": "How many candidates to generate (1 to 2x the configured default)" },
                        "guidance": { "type": "string", "description": "Concrete instructions for the generator: which part of the algorithm to change and how" }
                    },
                    "required": ["operator", "guidance"]
                }
            },
            {
                "name": "submit_code",
                "description": "Evaluate a complete EVOLVE block you wrote yourself (helper functions plus class Model). Use it for precise algorithmic changes or to fix a reported error.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": { "type": "string", "description": "The full Python code for the block" },
                        "note": { "type": "string", "description": "One line describing the change" }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "archive",
                "description": "List the best archived candidates with their metrics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "top_n": { "type": "integer" },
                        "agent": { "type": "string", "description": "Restrict to one agent's candidates" },
                        "include_code": { "type": "boolean", "description": "Include the code of the top candidate" }
                    }
                }
            },
            {
                "name": "leaderboard",
                "description": "Current standings, your provisional reward, and the best rival's code.",
                "parameters": { "type": "object", "properties": {} }
            }
        ]
    }])
}

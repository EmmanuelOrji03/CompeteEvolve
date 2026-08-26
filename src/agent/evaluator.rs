use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Instant;

use super::database;
use super::evolution::squash;
use super::SharedTools;


#[derive(Clone, Debug)]
pub struct AgentEvaluation {
    pub code: String,
    pub performance: f64, // a1
    pub novelty: f64,     // N
}


pub async fn run(args: &Value, shared: &SharedTools, agent_name: &str) -> Result<String> {
   
    let sample_code = match args.get("candidate").and_then(|v| v.as_str()) {
        Some(code) => code.to_string(),
        None => fetch_agent_sample(shared, agent_name).await?,
    };

   
    let original_code = match args.get("original_code").and_then(|v| v.as_str()) {
        Some(code) => code.to_string(),
        None => fetch_by_category(shared, "code", None).await?,
    };

    
    let training_data = match args.get("training_data").and_then(|v| v.as_str()) {
        Some(d) => d.to_string(),
        None => fetch_by_category(shared, "training_data", None).await?,
    };

    
    let test_data = match args.get("test_data").and_then(|v| v.as_str()) {
        Some(d) => d.to_string(),
        None => fetch_by_category(shared, "test_data", None).await?,
    };

    
    let (t0, a0_accuracy) = simulate_run(&original_code, &training_data, &test_data);
    let a0 = a0_accuracy / t0.max(1e-6); // a0 = A0 / t0

    
    let syntax_errors = count_syntax_errors(&sample_code);
    let runtime_errors = count_runtime_errors(&sample_code);

    
    let (t1, a1_accuracy) = simulate_run(&sample_code, &training_data, &test_data);
    let code_length = sample_code.split_whitespace().count().max(1) as f64;
    // e = (a0 * 2^-(SE+RE)) / L
    let error_scale = (a0 * 2f64.powi(-((syntax_errors + runtime_errors) as i32))) / code_length;
    // a1 = (A1 * e) / t1
    let performance = (a1_accuracy * error_scale) / t1.max(1e-6);

    // ---- steps 23-26: novelty ----
    let sim1 =
        database::cosine_similarity(&database::embed(&sample_code), &database::embed(&original_code)) as f64;
    let sim2 = average_similarity_to_other_agents(shared, agent_name, &sample_code).await;
    let novelty = (sim1 + sim2) / 2.0;

   
    {
        let mut evaluations = shared.agent_evaluations.lock().await;
        evaluations.insert(
            agent_name.to_string(),
            AgentEvaluation { code: sample_code.clone(), performance, novelty },
        );
    }

   
    let positions = compute_rank_positions(shared).await?;
    let agent_count = positions.len();
    let rank_position = positions
        .iter()
        .find(|(name, _)| name == agent_name)
        .map(|(_, rp)| *rp)
        .ok_or_else(|| anyhow!("internal error: agent '{agent_name}' missing from its own rank computation"))?;

    Ok(json!({
        "status": "ok",
        "agent": agent_name,
        "a0": a0,
        "performance": performance,
        "sim1": sim1,
        "sim2": sim2,
        "novelty": novelty,
        "syntax_errors": syntax_errors,
        "runtime_errors": runtime_errors,
        "rank_position": rank_position,
        "agent_count": agent_count,
    })
    .to_string())
}


pub async fn compute_rank_positions(shared: &SharedTools) -> Result<Vec<(String, usize)>> {
    let snapshot: HashMap<String, AgentEvaluation> = shared.agent_evaluations.lock().await.clone();

    if snapshot.is_empty() {
        return Err(anyhow!(
            "no agent has been evaluated yet; run the 'evaluator' tool for at least one agent first"
        ));
    }

    let mut ordered: Vec<(String, f64)> = snapshot
        .into_iter()
        .map(|(name, eval)| (name, eval.novelty + squash(eval.performance)))
        .collect();
    ordered.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(ordered
        .into_iter()
        .enumerate()
        .map(|(i, (name, _combined_score))| (name, i + 1))
        .collect())
}


async fn fetch_agent_sample(shared: &SharedTools, agent_name: &str) -> Result<String> {
    let mut conn = shared.db.lock().await;
    let raw = database::run(
        &mut conn,
        &json!({ "action": "read", "top_n": 1, "category": "code", "agent": agent_name }),
    )?;
    let parsed: Value = serde_json::from_str(&raw)?;

    parsed["results"][0]["code"].as_str().map(String::from).ok_or_else(|| {
        anyhow!(
            "no sample found in the database for agent '{agent_name}'; \
             run 'evolution' for this agent first, or pass 'candidate' explicitly"
        )
    })
}


async fn fetch_by_category(shared: &SharedTools, category: &str, exclude_agent: Option<&str>) -> Result<String> {
    let mut conn = shared.db.lock().await;
    let raw = database::run(&mut conn, &json!({ "action": "read", "top_n": 20, "category": category }))?;
    let parsed: Value = serde_json::from_str(&raw)?;
    drop(conn);

    let results = parsed["results"].as_array().cloned().unwrap_or_default();
    let hit = results.into_iter().find(|r| {
        exclude_agent.map_or(true, |ex| r.get("agent").and_then(|v| v.as_str()) != Some(ex))
    });

    hit.and_then(|r| r["code"].as_str().map(String::from)).ok_or_else(|| {
        anyhow!(
            "no '{category}' entry found in the database; write one first \
             (e.g. database write with category: '{category}')"
        )
    })
}


async fn average_similarity_to_other_agents(shared: &SharedTools, agent_name: &str, sample_code: &str) -> f64 {
    let mut conn = shared.db.lock().await;
    let raw = database::run(&mut conn, &json!({ "action": "read", "top_n": 50, "category": "code" }));
    drop(conn);

    let Ok(raw) = raw else { return 0.0 };
    let Ok(parsed) = serde_json::from_str::<Value>(&raw) else { return 0.0 };
    let results = parsed["results"].as_array().cloned().unwrap_or_default();

    // Best (highest-ranked) entry per other agent, deduplicated by agent.
    let mut seen_agents: Vec<String> = Vec::new();
    let mut other_codes: Vec<String> = Vec::new();
    for r in results {
        let Some(agent) = r.get("agent").and_then(|v| v.as_str()) else { continue };
        if agent == agent_name || seen_agents.iter().any(|a| a == agent) {
            continue;
        }
        if let Some(code) = r.get("code").and_then(|v| v.as_str()) {
            seen_agents.push(agent.to_string());
            other_codes.push(code.to_string());
        }
    }

    if other_codes.is_empty() {
        return 0.0;
    }

    let sample_embedding = database::embed(sample_code);
    let total: f64 = other_codes
        .iter()
        .map(|c| database::cosine_similarity(&sample_embedding, &database::embed(c)) as f64)
        .sum();
    total / other_codes.len() as f64
}


fn simulate_run(code: &str, training_data: &str, test_data: &str) -> (f64, f64) {
    let complexity = code.split_whitespace().count().max(1);
    let data_size = (training_data.split_whitespace().count() + test_data.split_whitespace().count()).max(1);

    let start = Instant::now();
    let mut checksum: u64 = 0;
    // Capped so pathologically large inputs can't make this hang; still
    // deterministic and reproducible for a given (code, data) pair.
    for i in 0..(complexity * data_size).min(2_000_000) {
        checksum = checksum.wrapping_add((i as u64).wrapping_mul(31));
    }
    std::hint::black_box(checksum);
    let time_seconds = start.elapsed().as_secs_f64().max(1e-6);

    let mut score = 0.5_f64;
    if code.contains("return") {
        score += 0.2;
    }
    if code.contains('{') && code.contains('}') {
        score += 0.2;
    }
    // More training/test data nudges the accuracy proxy up slightly.
    score += (data_size as f64).ln().max(0.0) * 0.01;
    let accuracy = score.clamp(0.05, 1.0);

    (time_seconds, accuracy)
}


fn count_syntax_errors(code: &str) -> u32 {
    let mut parens = 0i32;
    let mut braces = 0i32;
    let mut brackets = 0i32;
    let mut errors = 0u32;

    for ch in code.chars() {
        match ch {
            '(' => parens += 1,
            ')' => {
                parens -= 1;
                if parens < 0 {
                    errors += 1;
                    parens = 0;
                }
            }
            '{' => braces += 1,
            '}' => {
                braces -= 1;
                if braces < 0 {
                    errors += 1;
                    braces = 0;
                }
            }
            '[' => brackets += 1,
            ']' => {
                brackets -= 1;
                if brackets < 0 {
                    errors += 1;
                    brackets = 0;
                }
            }
            _ => {}
        }
    }

    errors + parens.max(0) as u32 + braces.max(0) as u32 + brackets.max(0) as u32
}


fn count_runtime_errors(code: &str) -> u32 {
    const RISKY_PATTERNS: [&str; 4] = [".unwrap()", "panic!(", "/ 0", "/0"];
    RISKY_PATTERNS.iter().map(|p| code.matches(p).count() as u32).sum()
}

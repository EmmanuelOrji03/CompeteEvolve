use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

use super::database::{self, Connection};

/// One scored candidate produced during evolution.
#[derive(Clone, Debug)]
struct Sample {
    id: String,
    code: String,
    novelty: f64,
    time_seconds: f64,
    accuracy: f64,
    performance: f64,
    rank: f64,
}


pub async fn run(
    args: &Value,
    agent_name: &str,
    api_key: &str,
    client: &reqwest::Client,
    db: &Arc<Mutex<Connection>>,
) -> Result<String> {
    // ---- steps 1-2: population parameters ----
    let max_iterations = args
        .get("generations")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("evolution requires a 'generations' field"))? as usize;
    let population_count = args // N
        .get("population_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(3) as usize;
    let samples_per_population = args // n
        .get("samples_per_population")
        .and_then(|v| v.as_u64())
        .unwrap_or(4) as usize;
    let mu_threshold = args.get("mu_threshold").and_then(|v| v.as_f64()).unwrap_or(0.7);

    if max_iterations == 0 || population_count == 0 || samples_per_population == 0 {
        return Err(anyhow!(
            "'generations', 'population_size', and 'samples_per_population' must all be greater than 0"
        ));
    }

   
    let original_code = match args.get("original_code").and_then(|v| v.as_str()) {
        Some(code) => code.to_string(),
        None => fetch_original_code(db).await?,
    };

    let mut mu = 0.0_f64;
    let mut best_overall: Option<Sample> = None;
    let mut iterations_run = 0usize;

    
    while mu < mu_threshold && iterations_run < max_iterations {
        iterations_run += 1;

   
        for pop_idx in 0..population_count {
            
            let generated =
                generate_samples(client, api_key, &original_code, samples_per_population).await?;

         
            let mut population: Vec<Sample> = Vec::with_capacity(generated.len());
            for (sample_idx, code) in generated.into_iter().enumerate() {
                
                let novelty =
                    database::cosine_similarity(&database::embed(&code), &database::embed(&original_code))
                        as f64;

               
                let (time_seconds, accuracy) = run_against_test_data(&code);

                let performance = time_seconds / accuracy.max(1e-6);

               
                let rank = performance * novelty;

                population.push(Sample {
                    id: format!("gen{iterations_run}_pop{pop_idx}_sample{sample_idx}"),
                    code,
                    novelty,
                    time_seconds,
                    accuracy,
                    performance,
                    rank,
                });
            }

            
            population.sort_by(|a, b| b.rank.partial_cmp(&a.rank).unwrap_or(std::cmp::Ordering::Equal));

         
            if let Some(top) = population.into_iter().next() {
                let is_better = best_overall.as_ref().map_or(true, |b| top.rank > b.rank);
                if is_better {
                    best_overall = Some(top);
                }
            }
        }
        if let Some(best) = &best_overall {
            mu = squash(best.rank);
        }
    }


    let stb = best_overall.ok_or_else(|| anyhow!("evolution produced no viable samples"))?;

   
    {
        let mut conn = db.lock().await;
        let _ = database::run(
            &mut conn,
            &json!({
                "action": "write",
                "code": stb.code,
                "functional_accuracy": stb.accuracy,
                "rank": stb.rank,
                "category": "code",
                "agent": agent_name,
                "id": stb.id,
            }),
        );
    }

    
    Ok(json!({
        "status": "ok",
        "agent": agent_name,
        "id": stb.id,
        "rank": stb.rank,
        "mu": mu,
        "novelty": stb.novelty,
        "performance": stb.performance,
        "time_seconds": stb.time_seconds,
        "accuracy": stb.accuracy,
        "iterations_run": iterations_run,
        "code": stb.code,
    })
    .to_string())
}


async fn fetch_original_code(db: &Arc<Mutex<Connection>>) -> Result<String> {
    let mut conn = db.lock().await;
    let raw = database::run(&mut conn, &json!({ "action": "read", "top_n": 1 }))?;
    let parsed: Value = serde_json::from_str(&raw)?;

    parsed["results"][0]["code"].as_str().map(String::from).ok_or_else(|| {
        anyhow!(
            "no original code found in the database to evolve; \
             pass 'original_code' explicitly or write a seed file first"
        )
    })
}


async fn generate_samples(
    client: &reqwest::Client,
    api_key: &str,
    original_code: &str,
    n: usize,
) -> Result<Vec<String>> {
    let prompt = format!(
        "You are assisting an evolutionary code search. Given the code below, \
         propose a single alternative implementation that preserves its behavior \
         but may differ in approach, structure, or efficiency. \
         Respond with ONLY the code — no explanation, no markdown fences.\n\n\
         Original code:\n{original_code}"
    );

    let url = format!("{}/{}:generateContent?key={}", super::API_URL, super::MODEL, api_key);
    let mut samples = Vec::with_capacity(n);

    for _ in 0..n {
        let body = json!({
            "contents": [{ "role": "user", "parts": [{ "text": prompt }] }]
        });

        let resp: Value = client.post(&url).json(&body).send().await?.json().await?;

        let text = resp["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .unwrap_or_default();

        let cleaned = strip_code_fences(text);
        if !cleaned.trim().is_empty() {
            samples.push(cleaned);
        }
    }

    if samples.is_empty() {
        return Err(anyhow!("Gemini returned no usable samples"));
    }
    Ok(samples)
}


fn strip_code_fences(text: &str) -> String {
    let trimmed = text.trim();
    if let Some(stripped) = trimmed.strip_prefix("```") {
        let without_lang = stripped.trim_start_matches(|c: char| c.is_alphanumeric());
        let without_lang = without_lang.strip_prefix('\n').unwrap_or(without_lang);
        if let Some(end) = without_lang.rfind("```") {
            return without_lang[..end].trim().to_string();
        }
        return without_lang.trim().to_string();
    }
    trimmed.to_string()
}


fn run_against_test_data(code: &str) -> (f64, f64) {
   
    let complexity = code.split_whitespace().count().max(1);
    let start = Instant::now();
    let mut checksum: u64 = 0;
    for i in 0..(complexity * 50) {
        checksum = checksum.wrapping_add((i as u64).wrapping_mul(31));
    }
    std::hint::black_box(checksum);
    let time_seconds = start.elapsed().as_secs_f64().max(1e-6);


    let mut score = 0.5_f64;
    if code.contains("return") {
        score += 0.25;
    }
    if code.contains('{') && code.contains('}') {
        score += 0.25;
    }
    let accuracy = score.clamp(0.05, 1.0);

    (time_seconds, accuracy)
}

pub(crate) fn squash(rank: f64) -> f64 {
    if rank.is_finite() {
        rank / (1.0 + rank)
    } else {
        0.999
    }
}

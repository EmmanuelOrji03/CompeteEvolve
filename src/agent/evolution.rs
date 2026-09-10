//! Evolution
//!
//! Implements the algorithm described in `Evolution.md`:
//!
//! ```text
//! 1.  Initialize N  — number of sample populations
//! 2.  Initialize n  — samples per population
//! 3.  Fetch the original code from the database
//! 4.  While mu < 0.7:
//! 5.    For each of the N populations:
//! 6.      Generate n samples via the Gemini API
//! 7.      Novelty (NO)   = cosine similarity(sample, original)
//! 8-9.    Run the sample against test data, measuring time T
//! 10.     Accuracy (A) of the result
//! 11.     Performance (a) = T / A
//! 12.     Rank (R)        = a * NO
//! 13-14.  Build the population array, sorted by rank descending
//! 15.     Keep the top-ranked sample of the population, TB
//! 16.   end for
//! 17. STB = highest-ranked TB across all populations
//! 18. Return the rank of STB
//! 19. Write the sample (and its rank) to the vector database
//! ```
//!
//! This tool needs things `evaluator`/`reinforcement` don't: the Gemini
//! API (to generate candidate samples) and the shared database (to fetch
//! the original code / training / test data, and write the winning
//! candidate back). It also needs to know which agent and which
//! algorithm it's running for — the database entry it writes (step 19) is
//! tagged with both, so `evaluator` can later fetch "the sample from a
//! particular agent" for a particular algorithm (Evaluator.md step 1),
//! and other agents' samples for the same algorithm (step 25). All of
//! this is threaded in from `Agent::call_tool` in `mod.rs`.
//!
//! ## Real execution
//! Steps 8-10 ("run the algorithm against the test data") now actually
//! run each sample, via `sandbox::run` — see `sandbox.rs` for exactly
//! what that does and doesn't protect against.

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;

use super::database::{self, Connection};
use super::sandbox;

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
    syntax_errors: u32,
    runtime_errors: u32,
}

/// Entry point called from `Agent::call_tool`.
///
/// `algorithm` (e.g. `"kmeans"`) selects which algorithm's code/training
/// data/test data to fetch and which tag the winning sample is written
/// back under — it's supplied by the `Agent` itself (set at
/// construction), not read from `args`.
///
/// Expected `args`:
/// ```json
/// {
///   "generations": 5,              // required — max iterations of the while loop (safety cap)
///   "population_size": 3,          // optional — N, default 3
///   "samples_per_population": 4,   // optional — n, default 4
///   "mu_threshold": 0.7,           // optional — default 0.7, per spec
///   "original_code": "..."         // optional — skips the database fetch in step 3
/// }
/// ```
pub async fn run(
    args: &Value,
    agent_name: &str,
    algorithm: &str,
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

    // ---- step 3: get the original code, training data, and test data ----
    let original_code = match args.get("original_code").and_then(|v| v.as_str()) {
        Some(code) => code.to_string(),
        None => fetch_by_category(db, "code", algorithm, None).await?,
    };
    let training_data = match args.get("training_data").and_then(|v| v.as_str()) {
        Some(d) => d.to_string(),
        None => fetch_by_category(db, "training_data", algorithm, None).await?,
    };
    let test_data = match args.get("test_data").and_then(|v| v.as_str()) {
        Some(d) => d.to_string(),
        None => fetch_by_category(db, "test_data", algorithm, None).await?,
    };

    let time_limit = sandbox::default_time_limit();

    eprintln!(
        "[evolution:{agent_name}] starting on '{algorithm}': up to {max_iterations} generation(s), \
         {population_count} population(s) x {samples_per_population} sample(s), \
         sandbox timeout {:.0}s per sample",
        time_limit.as_secs_f64()
    );

    let mut mu = 0.0_f64;
    let mut best_overall: Option<Sample> = None;
    let mut iterations_run = 0usize;

    // ---- step 4: evolve until the quality threshold is met ----
    while mu < mu_threshold && iterations_run < max_iterations {
        iterations_run += 1;
        eprintln!("[evolution:{agent_name}] generation {iterations_run}/{max_iterations} (mu={mu:.3})");

        // ---- step 5: for each sample population ----
        for pop_idx in 0..population_count {
            eprintln!(
                "[evolution:{agent_name}]   population {}/{population_count}: requesting \
                 {samples_per_population} sample(s) from Gemini...",
                pop_idx + 1
            );
            // ---- step 6: generate n samples via the Gemini API ----
            let generated =
                generate_samples(client, api_key, &original_code, samples_per_population).await?;

            // ---- steps 7-12: score every sample in this population ----
            let mut population: Vec<Sample> = Vec::with_capacity(generated.len());
            for (sample_idx, code) in generated.into_iter().enumerate() {
                // NO: novelty, the cosine similarity of the sample against the original.
                let novelty =
                    database::cosine_similarity(&database::embed(&code), &database::embed(&original_code))
                        as f64;

                eprintln!(
                    "[evolution:{agent_name}]     sample {}/{samples_per_population}: running in sandbox...",
                    sample_idx + 1
                );
                // T, A: actually run the sample against the training/test data.
                let sandbox_result = sandbox::run(&code, &training_data, &test_data, time_limit).await?;
                let time_seconds = sandbox_result.training_time_seconds;
                let accuracy = sandbox_result.accuracy;
                eprintln!(
                    "[evolution:{agent_name}]     sample {}/{samples_per_population}: accuracy={accuracy:.3} \
                     time={time_seconds:.3}s syntax_err={} runtime_err={}",
                    sample_idx + 1,
                    sandbox_result.syntax_errors,
                    sandbox_result.runtime_errors
                );

                // a = T / A (accuracy floored away from zero to avoid a division blow-up).
                let performance = time_seconds / accuracy.max(1e-6);

                // R = a * NO
                let rank = performance * novelty;

                population.push(Sample {
                    id: format!("gen{iterations_run}_pop{pop_idx}_sample{sample_idx}"),
                    code,
                    novelty,
                    time_seconds,
                    accuracy,
                    performance,
                    rank,
                    syntax_errors: sandbox_result.syntax_errors,
                    runtime_errors: sandbox_result.runtime_errors,
                });
            }

            // ---- steps 13-14: population array indexed by decreasing rank ----
            population.sort_by(|a, b| b.rank.partial_cmp(&a.rank).unwrap_or(std::cmp::Ordering::Equal));

            // ---- step 15: TB — top sample of this population ----
            if let Some(top) = population.into_iter().next() {
                let is_better = best_overall.as_ref().map_or(true, |b| top.rank > b.rank);
                if is_better {
                    best_overall = Some(top);
                }
            }
        }

        // Rank is unbounded (a = T/A can be arbitrarily large), so it is
        // squashed into [0, 1) before being compared against mu_threshold —
        // otherwise "mu < 0.7" would rarely mean anything stable.
        if let Some(best) = &best_overall {
            mu = squash(best.rank);
        }
    }

    // ---- step 17: STB — best sample across every population/iteration ----
    let stb = best_overall.ok_or_else(|| anyhow!("evolution produced no viable samples"))?;
    eprintln!(
        "[evolution:{agent_name}] finished after {iterations_run} generation(s): best accuracy={:.3} \
         time={:.3}s rank={:.3} (mu={mu:.3})",
        stb.accuracy, stb.time_seconds, stb.rank
    );

    // ---- step 19: write the sample and its rank to the (vector) database ----
    // Tagged with `agent_name` and `algorithm` so `evaluator` can later
    // fetch this specific agent's sample for this algorithm
    // (Evaluator.md step 1) and compare it against other agents' samples
    // for the same algorithm (step 25).
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
                "algorithm": algorithm,
                "id": stb.id,
            }),
        );
    }

    // ---- step 18: return the rank value of STB (with supporting context) ----
    Ok(json!({
        "status": "ok",
        "agent": agent_name,
        "algorithm": algorithm,
        "id": stb.id,
        "rank": stb.rank,
        "mu": mu,
        "novelty": stb.novelty,
        "performance": stb.performance,
        "time_seconds": stb.time_seconds,
        "accuracy": stb.accuracy,
        "syntax_errors": stb.syntax_errors,
        "runtime_errors": stb.runtime_errors,
        "iterations_run": iterations_run,
        "code": stb.code,
    })
    .to_string())
}

/// Step 3 (and its training/test-data counterparts): pull the top-ranked
/// entry of a given category, for a given algorithm, out of the
/// database. `exclude_agent`, when set, skips entries tagged with that
/// agent (unused here today, kept for symmetry with `evaluator.rs`'s
/// identical helper).
async fn fetch_by_category(
    db: &Arc<Mutex<Connection>>,
    category: &str,
    algorithm: &str,
    exclude_agent: Option<&str>,
) -> Result<String> {
    let mut conn = db.lock().await;
    let raw = database::run(
        &mut conn,
        &json!({ "action": "read", "top_n": 20, "category": category, "algorithm": algorithm }),
    )?;
    let parsed: Value = serde_json::from_str(&raw)?;
    drop(conn);

    let results = parsed["results"].as_array().cloned().unwrap_or_default();
    let hit = results.into_iter().find(|r| {
        exclude_agent.map_or(true, |ex| r.get("agent").and_then(|v| v.as_str()) != Some(ex))
    });

    hit.and_then(|r| r["code"].as_str().map(String::from)).ok_or_else(|| {
        anyhow!(
            "no '{category}' entry found for algorithm '{algorithm}' in the database; write one first \
             (e.g. database write with category: '{category}', algorithm: '{algorithm}')"
        )
    })
}

/// Step 6: ask Gemini for `n` alternative implementations of `original_code`.
async fn generate_samples(
    client: &reqwest::Client,
    api_key: &str,
    original_code: &str,
    n: usize,
) -> Result<Vec<String>> {
    let prompt = format!(
        "You are assisting an evolutionary code search. Given the Python code below, \
         propose a single alternative implementation that preserves its behavior \
         but may differ in approach, structure, or efficiency.\n\n\
         The code MUST define a function with exactly this signature, since it will \
         be executed by an automated harness:\n\
         def run(train_path: str, test_path: str) -> float:\n\
         \x20   # trains on the CSV at train_path, evaluates on the CSV at test_path,\n\
         \x20   # and returns test-set accuracy as a float in [0, 1]\n\n\
         Respond with ONLY the Python code — no explanation, no markdown fences.\n\n\
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

/// Strips a leading/trailing ``` fence (with an optional language tag) if
/// the model wrapped its answer in one despite being asked not to.
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

/// Maps an unbounded non-negative rank into [0, 1) so it can be compared
/// against `mu_threshold` (spec: "while mu < 0.7"), and, in `evaluator.rs`,
/// combined with a bounded [0,1] novelty score. Uses `ln(1+x)` before
/// squashing rather than squashing `x` directly: `performance` values
/// here are `accuracy / time`, and `time` is frequently a small fraction
/// of a second, which routinely pushes raw `x` into the thousands or
/// billions. A direct `x / (1 + x)` squash saturates to ~1.0 for
/// anything past roughly `x = 20` — so two candidates with wildly
/// different performance (e.g. 10x more accurate, or 100x faster) would
/// both squash to indistinguishable values near 1.0, making performance
/// silently stop affecting rank at exactly the scale it actually occurs
/// at. Taking `ln(1+x)` first compresses the huge range down before the
/// final squash, so relative differences stay meaningful across many
/// orders of magnitude instead of only across the first order or two.
pub(crate) fn squash(rank: f64) -> f64 {
    if rank.is_finite() && rank >= 0.0 {
        let l = (1.0 + rank).ln();
        l / (1.0 + l)
    } else {
        0.999
    }
}

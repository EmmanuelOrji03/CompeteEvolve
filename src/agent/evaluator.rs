//! Evaluator
//!
//! Implements the algorithm described in `Evaluator.md`:
//!
//! ```text
//! 1.  Fetch the generated code sample (for this agent, this algorithm) from the hybrid database
//! 2.  Fetch training data (for this algorithm) from the hybrid database
//! 3.  Fetch test data (for this algorithm) from the hybrid database
//! 4-9.   a0 = A0 / t0, running the original code (baseline performance)
//! 10-11. Initialize SE (syntax errors), RE (runtime errors)
//! 12-22. For the sample: run it, catch syntax/runtime errors, apply the
//!        error-scale penalty e = (a0 * 2^-(SE+RE)) / L, then
//!        a1 = (A1 * e) / t1
//! 23-27. Novelty N = (Sim1 + Sim2) / 2, where Sim1 is similarity to the
//!        original code and Sim2 is similarity to other agents' samples
//! 28-31. Rank agents by N and a1, assign rank position (RP, 1 = best),
//!        return the rank positions to the reinforcement system
//! ```
//!
//! ## Assumptions made explicit (the spec leaves these underspecified)
//! - Step 5 says "pass the training data through the original training
//!   data", which can't be literal (training data through itself). This
//!   is read as "pass the training data through the **original code**",
//!   mirroring step 13's "pass the training data through the sample
//!   code".
//! - Step 18's error-scale formula `e = (a * 2^-(SE+RE)) / L` doesn't
//!   define `a` or `L`. `a` is read as `a0` (the only performance value
//!   computed so far at that point — `a1` isn't computed until steps
//!   19-21). `L` is read as the sample code's length, measured in
//!   whitespace-separated tokens (consistent with the complexity measure
//!   `evolution.rs` already uses).
//! - Step 29's "RP = 1 based on the agents with the highest value of N
//!   and a1" doesn't give a combination formula. This combines them as
//!   `novelty + squash(a1)`, squashing `a1` into [0, 1) first (via the
//!   same helper `evolution.rs` uses for its own unbounded rank) so it's
//!   on a comparable scale to novelty before the two are added.
//!
//! ## Real execution
//! "Running" the original and sample code and "catching" syntax/runtime
//! errors is now done for real, via `sandbox::run` — each is actually
//! executed against the training/test CSV data in an isolated, timed
//! subprocess (see `sandbox.rs` for exactly what that does and doesn't
//! protect against). `syntax_errors`/`runtime_errors` below come directly
//! from what the sandbox observed, not a heuristic guess.
//!
//! ## Cross-agent state
//! Steps 25 and 28-31 need every agent's current sample/performance/
//! novelty, so this tool keeps a shared snapshot in
//! `SharedTools::agent_evaluations`, updated each time an agent is
//! evaluated. `reinforcement::run` reads the rank positions computed here
//! via `compute_rank_positions`, per Reinforcement.md step 1 ("fetch the
//! rank position from the Evaluator").

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::collections::HashMap;

use super::database;
use super::evolution::squash;
use super::sandbox;
use super::SharedTools;

/// One agent's most recent evaluation, kept so novelty/rank-position
/// calculations can see across agents. Also carries the raw sandbox
/// numbers (accuracy, time, error counts) — not needed for ranking, but
/// this is the one place they're available across agents, so
/// `benchmark.rs` reads them from here for its per-agent report.
#[derive(Clone, Debug)]
pub struct AgentEvaluation {
    pub code: String,
    pub performance: f64, // a1
    pub novelty: f64,     // N
    pub accuracy: f64,
    pub training_time_seconds: f64,
    pub syntax_errors: u32,
    pub runtime_errors: u32,
}

/// Entry point called from `Agent::call_tool`.
///
/// `algorithm` (e.g. `"kmeans"`) selects which algorithm's code/training
/// data/test data to fetch — it's supplied by the `Agent` itself (set at
/// construction), not read from `args`, so evaluation always uses the
/// algorithm the agent is actually assigned to regardless of what the
/// model puts in a tool call.
///
/// Expected `args` (all optional — each falls back to fetching from the
/// database):
/// ```json
/// {
///   "candidate": "...",     // sample code to evaluate; defaults to this agent's latest DB entry
///   "original_code": "...", // defaults to the top-ranked "code" entry for this algorithm
///   "training_data": "...", // defaults to the stored "training_data" entry for this algorithm
///   "test_data": "..."      // defaults to the stored "test_data" entry for this algorithm
/// }
/// ```
pub async fn run(args: &Value, shared: &SharedTools, agent_name: &str, algorithm: &str) -> Result<String> {
    // ---- step 1: fetch the generated code sample for this agent ----
    let sample_code = match args.get("candidate").and_then(|v| v.as_str()) {
        Some(code) => code.to_string(),
        None => fetch_agent_sample(shared, agent_name, algorithm).await?,
    };

    // ---- (needed for a0/Sim1): the original code this sample was evolved from ----
    // Deliberately NOT `fetch_by_category` here: that picks whichever
    // "code" entry currently ranks highest, which drifts to an agent's
    // own high-accuracy submission as the run progresses — silently
    // comparing agents against each other instead of against the fixed
    // baseline. `fetch_original_code` specifically requires no `agent`
    // tag, matching how `evolution.rs` writes the seed and how
    // `benchmark.rs` identifies it.
    let original_code = match args.get("original_code").and_then(|v| v.as_str()) {
        Some(code) => code.to_string(),
        None => fetch_original_code(shared, algorithm).await?,
    };

    // ---- step 2: fetch training data ----
    let training_data = match args.get("training_data").and_then(|v| v.as_str()) {
        Some(d) => d.to_string(),
        None => fetch_by_category(shared, "training_data", algorithm, None).await?,
    };

    // ---- step 3: fetch test data ----
    let test_data = match args.get("test_data").and_then(|v| v.as_str()) {
        Some(d) => d.to_string(),
        None => fetch_by_category(shared, "test_data", algorithm, None).await?,
    };

    let time_limit = sandbox::default_time_limit();
    eprintln!("[evaluator:{agent_name}] running baseline ('{algorithm}' original code) in sandbox...");

    // ---- steps 4-9: baseline performance of the original code ----
    // "Pass the training data through the original code" (see module docs
    // on step 5's likely typo), then check it against the test data.
    let baseline = sandbox::run(&original_code, &training_data, &test_data, time_limit).await?;

    // If the baseline fails, the system must NOT halt — it should count
    // the error and keep going, exactly like a per-agent sample failure
    // already does. What changes is how `a0` is computed: baseline
    // accuracy of 0.0 would otherwise multiply straight through
    // `error_scale` and zero out *every* agent's performance regardless
    // of their own code's quality (that was the bug fixed previously,
    // but fixing it by hard-erroring the whole tool call was wrong too —
    // it aborted this agent's turn instead of just reporting the
    // problem). So: if the baseline is broken, skip the a0 factor
    // entirely (fall back to `error_scale = 2^-(SE+RE) / L`, still
    // driven by *this sample's own* error count) rather than crashing or
    // silently multiplying by zero, and say so clearly in the result so
    // it's visible and actionable rather than a mysteriously flat 0.000.
    let baseline_ok = baseline.syntax_errors == 0 && baseline.runtime_errors == 0;
    let a0 = if baseline_ok {
        Some(baseline.accuracy / baseline.training_time_seconds.max(1e-6)) // a0 = A0 / t0
    } else {
        eprintln!(
            "[evaluator:{agent_name}] WARNING: original '{algorithm}' code failed in the sandbox \
             ({} syntax error(s), {} runtime error(s)) — continuing without halting, but its \
             performance can't be measured until it's fixed to define \
             `def run(train_path, test_path) -> float`.",
            baseline.syntax_errors, baseline.runtime_errors
        );
        None
    };
    eprintln!(
        "[evaluator:{agent_name}] baseline: accuracy={:.3} time={:.3}s ok={baseline_ok}",
        baseline.accuracy, baseline.training_time_seconds
    );

    eprintln!("[evaluator:{agent_name}] running this agent's evolved sample in sandbox...");
    // ---- steps 10-22: run the sample for real, get SE/RE/A1/t1 from the sandbox ----
    let sample_result = sandbox::run(&sample_code, &training_data, &test_data, time_limit).await?;
    let syntax_errors = sample_result.syntax_errors;
    let runtime_errors = sample_result.runtime_errors;
    eprintln!(
        "[evaluator:{agent_name}] sample: accuracy={:.3} time={:.3}s syntax_err={syntax_errors} runtime_err={runtime_errors}",
        sample_result.accuracy, sample_result.training_time_seconds
    );

    let code_length = sample_code.split_whitespace().count().max(1) as f64;
    let error_penalty = 2f64.powi(-((syntax_errors + runtime_errors) as i32));
    // e = (a0 * 2^-(SE+RE)) / L, or just 2^-(SE+RE) / L if the baseline
    // itself couldn't be measured (see above) — either way, this
    // sample's own errors still reduce its score.
    let error_scale = match a0 {
        Some(a0) => (a0 * error_penalty) / code_length,
        None => error_penalty / code_length,
    };
    // a1 = (A1 * e) / t1
    let performance = (sample_result.accuracy * error_scale) / sample_result.training_time_seconds.max(1e-6);

    // ---- steps 23-26: novelty ----
    let sim1 =
        database::cosine_similarity(&database::embed(&sample_code), &database::embed(&original_code)) as f64;
    let sim2 = average_similarity_to_other_agents(shared, agent_name, algorithm, &sample_code).await;
    // When no other agent has submitted a sample yet, there is nothing
    // for Sim2 to measure — averaging it in as 0.0 would read as "not
    // novel at all" and unfairly deflate whichever agent happens to be
    // evaluated first in a batch, purely because of evaluation order
    // rather than anything about its code. Falling back to Sim1 alone
    // keeps novelty meaningful (and order-independent) until there's an
    // actual peer to compare against.
    let novelty = match sim2 {
        Some(sim2) => (sim1 + sim2) / 2.0,
        None => sim1,
    };

    // Record this agent's evaluation so novelty/ranking stays current for
    // every agent, then compute rank positions across everyone known so far.
    {
        let mut evaluations = shared.agent_evaluations.lock().await;
        evaluations.insert(
            agent_name.to_string(),
            AgentEvaluation {
                code: sample_code.clone(),
                performance,
                novelty,
                accuracy: sample_result.accuracy,
                training_time_seconds: sample_result.training_time_seconds,
                syntax_errors,
                runtime_errors,
            },
        );
    }

    // ---- steps 28-31: rank position across agents, returned for reinforcement ----
    let positions = compute_rank_positions(shared).await?;
    let agent_count = positions.len();
    let rank_position = positions
        .iter()
        .find(|(name, _)| name == agent_name)
        .map(|(_, rp)| *rp)
        .ok_or_else(|| anyhow!("internal error: agent '{agent_name}' missing from its own rank computation"))?;

    // Actionable feedback folded into the tool result — this is what
    // "the agent whose code has an error is supposed to fix it" actually
    // requires: the error must reach the agent's next turn as something
    // it can act on (via the normal functionResponse mechanism in
    // `mod.rs`), not just a number buried in a table, and never a hard
    // failure that ends its turn before it can respond to it.
    let mut notes: Vec<String> = Vec::new();
    if syntax_errors > 0 || runtime_errors > 0 {
        notes.push(format!(
            "Your submitted code had {syntax_errors} syntax error(s) and {runtime_errors} \
             runtime error(s), which reduced its performance score. Fix these and evolve a new \
             sample."
        ));
    }
    if !baseline_ok {
        notes.push(format!(
            "The original '{algorithm}' baseline code failed in the sandbox ({} syntax error(s), \
             {} runtime error(s)), so performance couldn't be scaled against it this round. If \
             you can, write a corrected baseline via the database tool (category: 'code', \
             algorithm: '{algorithm}', no 'agent' field) defining \
             `def run(train_path, test_path) -> float`.",
            baseline.syntax_errors, baseline.runtime_errors
        ));
    }

    Ok(json!({
        "status": "ok",
        "agent": agent_name,
        "algorithm": algorithm,
        "a0": a0.unwrap_or(0.0),
        "baseline_ok": baseline_ok,
        "baseline_syntax_errors": baseline.syntax_errors,
        "baseline_runtime_errors": baseline.runtime_errors,
        "performance": performance,
        "sim1": sim1,
        "sim2": sim2,
        "novelty": novelty,
        "syntax_errors": syntax_errors,
        "runtime_errors": runtime_errors,
        "accuracy": sample_result.accuracy,
        "training_time_seconds": sample_result.training_time_seconds,
        "rank_position": rank_position,
        "agent_count": agent_count,
        "notes": notes,
        // Present only when something actually went wrong, for debugging.
        "sandbox_stderr": if sample_result.stderr.is_empty() { Value::Null } else { json!(sample_result.stderr) },
        "baseline_stderr": if baseline.stderr.is_empty() { Value::Null } else { json!(baseline.stderr) },
    })
    .to_string())
}

/// Steps 28-31: rank agents by (novelty, performance) and assign a rank
/// position 1..N, 1 being best. Exposed so `reinforcement::run` can fetch
/// rank positions "from the Evaluator" per Reinforcement.md step 1,
/// rather than recomputing them independently.
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

/// Step 1: fetch the most recent sample this specific agent produced for
/// this algorithm.
async fn fetch_agent_sample(shared: &SharedTools, agent_name: &str, algorithm: &str) -> Result<String> {
    let mut conn = shared.db.lock().await;
    let raw = database::run(
        &mut conn,
        &json!({ "action": "read", "top_n": 1, "category": "code", "agent": agent_name, "algorithm": algorithm }),
    )?;
    let parsed: Value = serde_json::from_str(&raw)?;

    parsed["results"][0]["code"].as_str().map(String::from).ok_or_else(|| {
        anyhow!(
            "no '{algorithm}' sample found in the database for agent '{agent_name}'; \
             run 'evolution' for this agent first, or pass 'candidate' explicitly"
        )
    })
}

/// The baseline "code" entry for `algorithm` — the one written with no
/// `agent` tag, distinguishing it from every agent's evolved submissions
/// (which are always tagged with an agent by `evolution.rs`). Mirrors
/// `benchmark.rs`'s identical helper; kept separate since the two modules
/// intentionally don't share internal fetch logic (see other modules'
/// docs on this pattern).
async fn fetch_original_code(shared: &SharedTools, algorithm: &str) -> Result<String> {
    let mut conn = shared.db.lock().await;
    let raw = database::run(
        &mut conn,
        &json!({ "action": "read", "top_n": 100, "category": "code", "algorithm": algorithm }),
    )?;
    drop(conn);

    let parsed: Value = serde_json::from_str(&raw)?;
    let results = parsed["results"].as_array().cloned().unwrap_or_default();

    results
        .into_iter()
        .find(|r| r.get("agent").map_or(true, |v| v.is_null()))
        .and_then(|r| r["code"].as_str().map(String::from))
        .ok_or_else(|| {
            anyhow!(
                "no baseline (agent-less) 'code' entry found for algorithm '{algorithm}'; \
                 write the original implementation first (database write with category: 'code', \
                 algorithm: '{algorithm}', and no 'agent' field)"
            )
        })
}

/// Steps 2-3: pull the top-ranked entry of a given category, for a given
/// algorithm, from the database.
/// `exclude_agent`, when set, skips entries tagged with that agent (used
/// for Sim2's "other agents" fetch).
async fn fetch_by_category(
    shared: &SharedTools,
    category: &str,
    algorithm: &str,
    exclude_agent: Option<&str>,
) -> Result<String> {
    let mut conn = shared.db.lock().await;
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

/// Step 25: cosine similarity of this agent's sample against every other
/// agent's most recent sample *for the same algorithm*, averaged.
/// Returns `None` if no other agent has submitted one yet, rather than
/// `0.0` — see the "no peers yet" note where this is called from `run`.
async fn average_similarity_to_other_agents(
    shared: &SharedTools,
    agent_name: &str,
    algorithm: &str,
    sample_code: &str,
) -> Option<f64> {
    let mut conn = shared.db.lock().await;
    let raw = database::run(
        &mut conn,
        &json!({ "action": "read", "top_n": 50, "category": "code", "algorithm": algorithm }),
    );
    drop(conn);

    let raw = raw.ok()?;
    let parsed = serde_json::from_str::<Value>(&raw).ok()?;
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
        return None;
    }

    let sample_embedding = database::embed(sample_code);
    let total: f64 = other_codes
        .iter()
        .map(|c| database::cosine_similarity(&sample_embedding, &database::embed(c)) as f64)
        .sum();
    Some(total / other_codes.len() as f64)
}

//! Reinforcement
//!
//! Implements the algorithm described in `Reinforcement.md`:
//!
//! ```text
//! 1. Fetch the rank position from the Evaluator
//! 2. For each agent:
//! 3.     if RP < N/2:
//! 4.         Reward (Q) = 1
//! 5.     else:
//! 6.         Reward (Q) = 0
//! 7. End For Loop
//! 8. Return the reward, Q, and the rank, RP
//! 9. Add the values to the feedback prompt to the agent's input
//! ```
//!
//! ## Correction: the reward rule (2026-09)
//! This previously implemented `RP >= N/2 && RP != 1` (reward the worse
//! half, explicitly excluding the very best), matching one literal
//! reading of the project's research paper — but that produces exactly
//! the opposite of the intended behavior: it punishes top performers and
//! rewards weaker ones. Confirmed against real benchmark output and
//! corrected per explicit instruction: an agent is rewarded when its
//! rank position is in the *better* half (`RP < N/2`, where `RP = 1` is
//! best) — the conventional, intuitive shape for a competitive reward
//! system, and consistent with the system's stated goal of agents being
//! "rewarded or punished based on rank" to drive improvement. No `!= 1`
//! exclusion is needed under this rule: rank 1 always satisfies `RP <
//! N/2` for any `N >= 2` and is rewarded, as it should be.
//!
//! ## What changed from the previous version
//! Rank positions used to be computed here, directly from each agent's
//! raw evolution rank. The updated spec makes the Evaluator the source of
//! truth for rank position (step 1: "fetch ... from the Evaluator"), so
//! this now calls `evaluator::compute_rank_positions` instead of
//! maintaining its own ranking logic — `evaluator.rs` already tracks
//! every agent's (novelty, performance) and ranks them per Evaluator.md
//! steps 28-31.
//!
//! ## Step 9: feedback into the agent's input
//! `Agent::send` (in `mod.rs`) already appends every tool's JSON result
//! as a `functionResponse` into the conversation history, which is what
//! gets sent back to Gemini on the next turn — that already satisfies
//! "add the values to the agent's input" structurally. To make sure the
//! reward/rank actually read as *feedback* rather than inert JSON, this
//! also includes a short natural-language `feedback_prompt` string in the
//! result.

use anyhow::{anyhow, Result};
use serde_json::{json, Value};

use super::evaluator;
use super::SharedTools;

/// Entry point called from `Agent::call_tool`.
///
/// Expected `args`:
/// ```json
/// { "episodes": 1 }
/// ```
/// `episodes` is part of the tool schema declared in `mod.rs` ("Runs a
/// reinforcement learning training step"), but the algorithm in
/// `Reinforcement.md` is a stateless computation over whatever rank
/// positions the Evaluator currently reports — it isn't itself an
/// episodic rollout. It's accepted and validated (must be >= 1) for
/// schema compatibility, but doesn't change the result of a single call.
pub async fn run(args: &Value, shared: &SharedTools, agent_name: &str) -> Result<String> {
    let episodes = args.get("episodes").and_then(|v| v.as_u64()).unwrap_or(1);
    if episodes == 0 {
        return Err(anyhow!("'episodes' must be at least 1"));
    }

    // ---- step 1: fetch the rank position from the Evaluator ----
    let positions = evaluator::compute_rank_positions(shared).await?;
    let n = positions.len();

    // ---- steps 2-7: reward for every agent ----
    // Reward the better half (lower rank position = better); see the
    // module-level correction note above for why this isn't `>= half`.
    let half = n as f64 / 2.0;
    let mut leaderboard: Vec<Value> = Vec::with_capacity(n);
    for (name, rp) in &positions {
        let reward: u8 = if (*rp as f64) < half { 1 } else { 0 };
        leaderboard.push(json!({
            "agent": name,
            "rank_position": rp,
            "reward": reward,
        }));
    }

    // ---- step 8: this agent's own reward and rank position ----
    let self_entry = leaderboard
        .iter()
        .find(|e| e.get("agent").and_then(|v| v.as_str()) == Some(agent_name))
        .cloned()
        .ok_or_else(|| anyhow!("agent '{agent_name}' has not been evaluated yet"))?;

    let rank_position = self_entry["rank_position"].as_u64().unwrap_or(0);
    let reward = self_entry["reward"].as_u64().unwrap_or(0);

    // ---- step 9: natural-language feedback to fold into the agent's next input ----
    let feedback_prompt = if reward == 1 {
        format!(
            "Reinforcement feedback: your last sample ranked position {rank_position} of {n} agents \
             and earned a reward of 1. Keep exploring in this direction."
        )
    } else {
        format!(
            "Reinforcement feedback: your last sample ranked position {rank_position} of {n} agents \
             and earned a reward of 0. Consider trying a different mutation strategy to improve \
             your rank."
        )
    };

    Ok(json!({
        "status": "ok",
        "agent": agent_name,
        "rank_position": rank_position,
        "reward": reward,
        "agent_count": n,
        "feedback_prompt": feedback_prompt,
        "leaderboard": leaderboard,
    })
    .to_string())
}


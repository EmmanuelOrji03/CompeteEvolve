use anyhow::{anyhow, Result};
use serde_json::{json, Value};

use super::evaluator;
use super::SharedTools;


pub async fn run(args: &Value, shared: &SharedTools, agent_name: &str) -> Result<String> {
    let episodes = args.get("episodes").and_then(|v| v.as_u64()).unwrap_or(1);
    if episodes == 0 {
        return Err(anyhow!("'episodes' must be at least 1"));
    }

    let positions = evaluator::compute_rank_positions(shared).await?;
    let n = positions.len();

    let half = n as f64 / 2.0;
    let mut leaderboard: Vec<Value> = Vec::with_capacity(n);
    for (name, rp) in &positions {
        let reward: u8 = if (*rp as f64) >= half && *rp != 1 { 1 } else { 0 };
        leaderboard.push(json!({
            "agent": name,
            "rank_position": rp,
            "reward": reward,
        }));
    }


    let self_entry = leaderboard
        .iter()
        .find(|e| e.get("agent").and_then(|v| v.as_str()) == Some(agent_name))
        .cloned()
        .ok_or_else(|| anyhow!("agent '{agent_name}' has not been evaluated yet"))?;

    let rank_position = self_entry["rank_position"].as_u64().unwrap_or(0);
    let reward = self_entry["reward"].as_u64().unwrap_or(0);
    let feedback_prompt = if reward == 1 {
        format!(
            "Reinforcement feedback: your last sample ranked position {rank_position} of {n} agents \
             and earned a reward of 1. Keep exploring in this direction."
        )
    } else if rank_position == 1 {
        format!(
            "Reinforcement feedback: your last sample ranked position 1 of {n} agents (the best) \
             and earned a reward of 0 — top performers aren't rewarded further, they've already \
             set the bar; keep it up."
        )
    } else {
        format!(
            "Reinforcement feedback: your last sample ranked position {rank_position} of {n} agents \
             and earned a reward of 0. Consider trying a different mutation strategy."
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

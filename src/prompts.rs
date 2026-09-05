//! All prompt text in one place.

use serde_json::json;

use crate::archive::{Archive, Parent, Standing};
use crate::evolution::Shared;
use crate::gemini::truncate;
use crate::harness::{round4, round5};

pub const GENERATION_SYSTEM: &str = "You are an expert in numerical algorithms and NumPy performance engineering. \
You write correct, vectorised, dependency-free Python. You reply with code only.";

fn constraints(shared: &Shared) -> String {
    format!(
        "Constraints:\n\
         - {contract}\n\
         - Only `numpy` (already imported as `np` above the block) and the Python standard \
           library may be imported. scikit-learn, scipy and every other package are rejected.\n\
         - Your output replaces everything between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`, \
           so it must be complete: helper functions plus the whole `class Model`.\n\
         - Quality must stay at or above {min_q:.4} (baseline {q:.4} minus tolerance {tol}); \
           otherwise the candidate is invalid no matter how fast it is.\n\
         - Lower wall time (fit + predict) is rewarded: performance = (quality/{q:.4})^{qe} x ({t:.5}s/time)^{se}.",
        contract = shared.task.contract(),
        min_q = shared.baseline.quality - shared.cfg.fitness.quality_tolerance,
        q = shared.baseline.quality,
        tol = shared.cfg.fitness.quality_tolerance,
        t = shared.baseline.time,
        qe = shared.cfg.fitness.quality_exponent,
        se = shared.cfg.fitness.speed_exponent,
    )
}

fn parent_section(title: &str, p: &Parent) -> String {
    format!(
        "{title}: {}\nMetrics: {}\n```python\n{}\n```\n",
        p.label, p.metrics, p.block
    )
}

pub fn mutate_prompt(shared: &Shared, parent: &Parent, guidance: Option<&str>) -> String {
    format!(
        "Task: {desc}\nGoal: {goal}\n\n{constraints}\n\n{guidance}\n{parent}\n\
         Produce ONE improved variant of the parent. Make a real algorithmic, numerical or \
         vectorisation change (initialisation, convergence, data layout, avoiding Python loops, \
         better numerics), not cosmetic edits. Keep the code self-contained.\n\
         Respond with ONLY the Python code for the block. No markdown fences, no explanation.",
        desc = shared.task.description,
        goal = shared.goal,
        constraints = constraints(shared),
        guidance = guidance
            .map(|g| format!("Strategy from the supervising agent: {g}\n"))
            .unwrap_or_default(),
        parent = parent_section("Parent", parent),
    )
}

pub fn crossover_prompt(shared: &Shared, a: &Parent, b: &Parent, guidance: Option<&str>) -> String {
    format!(
        "Task: {desc}\nGoal: {goal}\n\n{constraints}\n\n{guidance}\n{pa}\n{pb}\n\
         Produce ONE new implementation that combines the strengths of Parent A and Parent B \
         (for example A's algorithmic idea with B's vectorisation), and improves on both. \
         Do not copy either parent verbatim.\n\
         Respond with ONLY the Python code for the block. No markdown fences, no explanation.",
        desc = shared.task.description,
        goal = shared.goal,
        constraints = constraints(shared),
        guidance = guidance
            .map(|g| format!("Strategy from the supervising agent: {g}\n"))
            .unwrap_or_default(),
        pa = parent_section("Parent A", a),
        pb = parent_section("Parent B", b),
    )
}

pub fn agent_system(shared: &Shared, name: &str) -> String {
    let n = shared.cfg.agents;
    format!(
        "You are {name}, one of {n} agents competing in CompeteEvolve, a system that improves \
         machine-learning algorithm implementations through LLM-guided evolutionary search.\n\n\
         TASK: {desc}\nGOAL: {goal}\n\n{constraints}\n\n\
         HOW YOU ARE SCORED\n\
         - Every candidate is executed for real by a Python harness on {datasets}. Nothing is estimated.\n\
         - performance = 2^-(errors) x reliability x (quality/baseline)^{qe} x (baseline_time/time)^{se}. \
           1.0 means 'as good as the seed'.\n\
         - A candidate is VALID only if every fold ran without error and quality >= baseline - {tol}.\n\
         - novelty = 1 - mean similarity to the seed, your previous candidate and the rivals' best. \
           Near-duplicates of anything already archived are rejected before evaluation.\n\
         - rank score = squash(performance) x (1 - {w} + {w} x novelty), squash(x) = x/(1+x).\n\
         - Agents are ranked by their best VALID candidate's rank score. The top half earn reward 1, \
           the rest 0. The run stops once squash(best performance) >= {mu}.\n\n\
         TOOLS\n\
         - evolve(operator, n_samples, guidance): a generator model writes n_samples candidates from \
           parents sampled out of the shared archive ('mutate' = one parent, 'crossover' = two parents, \
           one preferably from a rival). Your guidance steers what it changes. Each candidate is \
           evaluated and archived; you get the metrics and errors back.\n\
         - submit_code(code): evaluate a block you wrote yourself. Best for precise algorithmic changes \
           or for fixing an error you saw.\n\
         - archive(top_n, agent, include_code): inspect the best archived candidates.\n\
         - leaderboard(): current standings, your reward, and the best rival's code.\n\n\
         RULES\n\
         - You have at most {budget} tool calls this round. Use them; prose alone scores nothing.\n\
         - When you are done (or the budget is spent) reply with plain text: what worked, what failed, \
           and the concrete strategy for your next round. That text is the only memory you keep.\n\
         - Copying a rival's code earns nothing: it is rejected as a duplicate or scored with zero novelty.",
        desc = shared.task.description,
        goal = shared.goal,
        constraints = constraints(shared),
        datasets = shared.task.datasets.join(", "),
        qe = shared.cfg.fitness.quality_exponent,
        se = shared.cfg.fitness.speed_exponent,
        tol = shared.cfg.fitness.quality_tolerance,
        w = shared.cfg.fitness.novelty_weight,
        mu = shared.cfg.mu_threshold,
        budget = shared.cfg.max_tool_calls_per_round,
    )
}

#[derive(Debug, Clone)]
pub struct RoundFeedback {
    pub standings: Vec<Standing>,
    pub position: usize,
    pub reward: u8,
    pub best_rival_id: Option<String>,
    pub best_rival_agent: Option<String>,
    pub best_rival_block: Option<String>,
    pub best_rival_metrics: Option<serde_json::Value>,
}

pub fn round_prompt(
    shared: &Shared,
    archive: &Archive,
    name: &str,
    round: usize,
    feedback: Option<&RoundFeedback>,
    notes: &str,
) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "ROUND {round} of {total}.\n\nBaseline (seed) measured by the harness: quality {q:.4} (std {qs:.4}), \
         fit {ft:.5}s, predict {pt:.5}s per fold.\n",
        total = shared.cfg.rounds,
        q = shared.baseline.quality,
        qs = shared.baseline_result.quality_std.unwrap_or(0.0),
        ft = shared.baseline_result.fit_time.unwrap_or(0.0),
        pt = shared.baseline_result.predict_time.unwrap_or(0.0),
    ));
    if let Some(sk) = &shared.sklearn_result {
        out.push_str(&format!(
            "For reference, scikit-learn's own implementation scores quality {:.4} in {:.5}s per fold.\n",
            sk.quality.unwrap_or(0.0),
            sk.total_time().unwrap_or(0.0)
        ));
    }

    out.push_str(&format!(
        "\nSEED EVOLVE BLOCK (the original code):\n```python\n{}\n```\n",
        archive.seed_block()
    ));

    if let Some(best) = archive.best_for_agent(name) {
        out.push_str(&format!(
            "\nYOUR BEST VALID CANDIDATE SO FAR: {}\n```python\n{}\n```\n",
            best.summary(),
            truncate(&best.block, 6000)
        ));
    } else if round > 1 {
        out.push_str(
            "\nYou have no valid candidate yet. Priority: produce one that runs on every fold.\n",
        );
    }

    if !notes.trim().is_empty() {
        out.push_str(&format!(
            "\nYOUR NOTES FROM LAST ROUND:\n{}\n",
            truncate(notes, 2500)
        ));
    }

    if let Some(fb) = feedback {
        let table: Vec<String> = fb
            .standings
            .iter()
            .map(|s| {
                format!(
                    "  #{} {:<10} score {:.4}  performance {:.4}  reward {}  valid {}/{}",
                    s.rank_position,
                    s.agent,
                    s.best_score,
                    s.best_performance,
                    s.reward,
                    s.valid_candidates,
                    s.candidates
                )
            })
            .collect();
        out.push_str(&format!(
            "\nREINFORCEMENT FEEDBACK: you finished round {} at position {} of {} and earned reward {}.\n\
             Leaderboard:\n{}\n",
            round - 1,
            fb.position,
            fb.standings.len(),
            fb.reward,
            table.join("\n")
        ));
        if fb.reward == 1 {
            out.push_str("You are in the rewarded half. Keep the direction that worked and push it further.\n");
        } else {
            out.push_str("You are in the unrewarded half. Change strategy: a different algorithmic idea, or crossover with a rival's approach.\n");
        }
        if let (Some(agent), Some(block)) = (&fb.best_rival_agent, &fb.best_rival_block) {
            out.push_str(&format!(
                "\nBEST RIVAL ({agent}, {}): {}\n```python\n{}\n```\n",
                fb.best_rival_id.clone().unwrap_or_default(),
                fb.best_rival_metrics.clone().unwrap_or(json!({})),
                truncate(block, 6000)
            ));
        }
    }

    if round == 1 {
        if let Some(ctx) = &shared.context_excerpt {
            out.push_str(&format!(
                "\nREFERENCE MATERIAL (excerpt, may be truncated):\n{ctx}\n"
            ));
        }
    }

    out.push_str(&format!(
        "\nYou have {} tool calls. Begin by calling a tool. Finish with your plain-text notes.",
        shared.cfg.max_tool_calls_per_round
    ));
    out
}

pub fn metrics_line(quality: Option<f64>, time: Option<f64>, performance: f64) -> String {
    format!(
        "quality {} time {}s performance {}",
        quality
            .map(round4)
            .map(|q| q.to_string())
            .unwrap_or_else(|| "-".into()),
        time.map(round5)
            .map(|t| t.to_string())
            .unwrap_or_else(|| "-".into()),
        round4(performance)
    )
}

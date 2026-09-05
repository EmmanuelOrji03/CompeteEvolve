//! Baseline, rounds with a barrier, feedback, summary.

use anyhow::{Context, Result, bail};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::agent::Agent;
use crate::archive::{Archive, now_secs};
use crate::config::Config;
use crate::evolution::Shared;
use crate::fitness::{Baseline, squash};
use crate::gemini::{GeminiClient, truncate};
use crate::harness::{Harness, HarnessResult, Template, round4, round5};
use crate::prompts::{self, RoundFeedback};

pub struct RunOptions {
    pub task: String,
    pub goal: Option<String>,
    pub context_file: Option<PathBuf>,
    pub run_id: Option<String>,
    pub baseline_only: bool,
}

pub async fn run(cfg: Config, api_key: Option<String>, opts: RunOptions) -> Result<()> {
    let harness = Arc::new(Harness::new(&cfg.harness)?);
    let task = harness
        .describe(&opts.task)
        .await
        .with_context(|| format!("describing task '{}'", opts.task))?;
    let seed_source = std::fs::read_to_string(&task.seed_file)
        .with_context(|| format!("reading {}", task.seed_file))?;
    let template = Template::parse(&seed_source)?;

    println!(
        "Task: {} ({}) on {}",
        task.task,
        task.kind,
        task.datasets.join(", ")
    );
    println!("{}", task.description);
    println!(
        "Harness: {} ({} folds x {} repeats)",
        harness.script_path().display(),
        cfg.harness.folds,
        cfg.harness.repeats
    );

    println!("\nMeasuring baseline (seed implementation)...");
    let baseline_result = harness.evaluate_seed(&task.task).await?;
    if !baseline_result.is_ok() {
        bail!(
            "the seed implementation failed in the harness: {:?}",
            baseline_result.runtime_errors
        );
    }
    let baseline = Baseline {
        quality: baseline_result.quality.unwrap_or(0.0),
        time: baseline_result.total_time().unwrap_or(1e-6).max(1e-6),
    };
    print_result("seed", &baseline_result);

    println!("Measuring scikit-learn reference...");
    let sklearn_result = match harness.evaluate_sklearn(&task.task).await {
        Ok(r) if r.is_ok() => {
            print_result("sklearn", &r);
            Some(r)
        }
        Ok(r) => {
            eprintln!("scikit-learn reference failed: {:?}", r.runtime_errors);
            None
        }
        Err(e) => {
            eprintln!("scikit-learn reference failed: {e:#}");
            None
        }
    };

    if opts.baseline_only {
        return Ok(());
    }
    let api_key = api_key.context("GEMINI_API_KEY is not set (copy .env.example to .env)")?;

    let run_id = opts
        .run_id
        .clone()
        .unwrap_or_else(|| format!("{}-{}", task.task, now_secs()));
    let run_dir = Path::new("runs").join(&run_id);
    let agent_names: Vec<String> = (1..=cfg.agents).map(|i| format!("agent-{i}")).collect();
    let archive = Archive::open(&run_dir, &template.block, agent_names.clone())?;
    std::fs::write(
        run_dir.join("config.json"),
        serde_json::to_string_pretty(&cfg)?,
    )?;
    println!("\nRun directory: {}", run_dir.display());

    let context_excerpt = match &opts.context_file {
        Some(p) => {
            let text = std::fs::read_to_string(p)
                .with_context(|| format!("reading context {}", p.display()))?;
            Some(truncate(&text, cfg.context_max_chars))
        }
        None => None,
    };
    let goal = opts
        .goal
        .clone()
        .unwrap_or_else(|| default_goal(&task.kind));

    let shared = Arc::new(Shared {
        gemini: Arc::new(GeminiClient::new(api_key, cfg.gemini.clone())?),
        harness,
        task,
        template,
        baseline,
        baseline_result,
        sklearn_result,
        archive: Arc::new(Mutex::new(archive)),
        rng: Arc::new(Mutex::new(StdRng::seed_from_u64(
            cfg.harness.seed ^ 0x5eed_c0de,
        ))),
        goal,
        context_excerpt,
        cfg,
    });

    let mut agents: Vec<Agent> = agent_names.iter().map(Agent::new).collect();
    let mut feedback: HashMap<String, RoundFeedback> = HashMap::new();
    let mut rounds_completed = 0usize;

    for round in 1..=shared.cfg.rounds {
        println!(
            "\n================ ROUND {round}/{} ================",
            shared.cfg.rounds
        );

        let mut round_prompts = Vec::with_capacity(agents.len());
        {
            let archive = shared.archive.lock().await;
            for agent in &agents {
                round_prompts.push(prompts::round_prompt(
                    &shared,
                    &archive,
                    &agent.name,
                    round,
                    feedback.get(&agent.name),
                    &agent.notes,
                ));
            }
        }

        let futures = agents
            .iter_mut()
            .zip(round_prompts)
            .map(|(agent, prompt)| agent.run_round(&shared, round, prompt));
        let results = futures::future::join_all(futures).await;
        for (agent, result) in agents.iter().zip(results) {
            match result {
                Ok(notes) => println!("\n[{}] notes: {}", agent.name, truncate(notes.trim(), 600)),
                Err(e) => eprintln!("\n[{}] round {round} failed: {e:#}", agent.name),
            }
        }
        rounds_completed = round;

        let mut archive = shared.archive.lock().await;
        let standings = archive.standings();
        println!("\nLeaderboard after round {round}:");
        for s in &standings {
            println!(
                "  #{} {:<10} score {:.4}  performance {:.4}  reward {}  valid {}/{}  best {}",
                s.rank_position,
                s.agent,
                s.best_score,
                s.best_performance,
                s.reward,
                s.valid_candidates,
                s.candidates,
                s.best_id.clone().unwrap_or_else(|| "-".into())
            );
        }
        let best = archive.best().cloned();
        let mu = best.as_ref().map(|b| squash(b.performance)).unwrap_or(0.0);
        if let Some(b) = &best {
            println!(
                "Best so far: {} by {} - {} (mu {:.3})",
                b.id,
                b.agent,
                prompts::metrics_line(b.result.quality, b.result.total_time(), b.performance),
                mu
            );
        }
        archive.log_event(json!({ "event": "round_end", "round": round, "standings": standings, "mu": mu, "llm_calls": shared.gemini.calls() }));

        feedback.clear();
        for s in &standings {
            let rival = standings
                .iter()
                .filter(|o| o.agent != s.agent)
                .filter_map(|o| o.best_id.as_ref())
                .filter_map(|id| archive.entries().iter().find(|c| &c.id == id))
                .max_by(|a, b| {
                    a.score
                        .partial_cmp(&b.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            feedback.insert(
                s.agent.clone(),
                RoundFeedback {
                    standings: standings.clone(),
                    position: s.rank_position,
                    reward: s.reward,
                    best_rival_id: rival.map(|c| c.id.clone()),
                    best_rival_agent: rival.map(|c| c.agent.clone()),
                    best_rival_block: rival.map(|c| c.block.clone()),
                    best_rival_metrics: rival.map(|c| c.summary()),
                },
            );
        }
        let _ = archive.write_summary(json!({ "run_id": run_id, "rounds_completed": round, "llm_calls": shared.gemini.calls() }));
        drop(archive);

        if mu >= shared.cfg.mu_threshold {
            println!(
                "\nThreshold reached: mu {mu:.3} >= {}. Stopping early.",
                shared.cfg.mu_threshold
            );
            break;
        }
    }

    let archive = shared.archive.lock().await;
    let best = archive.best().cloned();
    let mut extra = json!({
        "run_id": run_id,
        "task": shared.task.task,
        "rounds_completed": rounds_completed,
        "llm_calls": shared.gemini.calls(),
        "baseline": { "quality": shared.baseline.quality, "time": shared.baseline.time, "result": shared.baseline_result },
        "sklearn_reference": shared.sklearn_result,
    });
    if let Some(b) = &best {
        let best_path = run_dir.join("best.py");
        std::fs::write(&best_path, shared.template.assemble(&b.block))?;
        let speedup = shared.baseline.time
            / b.result
                .total_time()
                .unwrap_or(shared.baseline.time)
                .max(1e-9);
        let quality_delta = b.result.quality.unwrap_or(0.0) - shared.baseline.quality;
        extra["improvement"] = json!({
            "speedup": round4(speedup),
            "quality_delta": round4(quality_delta),
            "performance": round4(b.performance),
            "best_file": best_path.to_string_lossy(),
        });
        println!("\n================ RESULT ================");
        println!(
            "Best candidate: {} by {} ({}, round {})",
            b.id, b.agent, b.operator, b.round
        );
        println!(
            "  quality {:.4} vs baseline {:.4} ({:+.4}); time {:.5}s vs {:.5}s ({:.2}x); performance {:.4}",
            b.result.quality.unwrap_or(0.0),
            shared.baseline.quality,
            quality_delta,
            b.result.total_time().unwrap_or(0.0),
            shared.baseline.time,
            speedup,
            b.performance
        );
        println!("  saved to {}", best_path.display());
    } else {
        println!(
            "\nNo valid candidate was produced. See {} for what was tried.",
            run_dir.join("candidates.jsonl").display()
        );
    }
    let summary_path = archive.write_summary(extra)?;
    println!(
        "Candidates: {} ({} valid, {} near-duplicates rejected). LLM calls: {}. Summary: {}",
        archive.len(),
        archive.entries().iter().filter(|c| c.valid).count(),
        archive.rejected_near_duplicates,
        shared.gemini.calls(),
        summary_path.display()
    );
    Ok(())
}

fn default_goal(kind: &str) -> String {
    if kind == "clustering" {
        "Reduce fit time and improve clustering quality (adjusted Rand index) of the implementation without using external libraries.".into()
    } else {
        "Reduce training and prediction time while keeping or improving held-out accuracy, without using external libraries.".into()
    }
}

fn print_result(label: &str, r: &HarnessResult) {
    println!(
        "  {label:<8} quality {:.4} (std {:.4})  fit {}s  predict {}s  folds {}/{}",
        r.quality.unwrap_or(0.0),
        r.quality_std.unwrap_or(0.0),
        r.fit_time.map(round5).unwrap_or(0.0),
        r.predict_time.map(round5).unwrap_or(0.0),
        r.n_folds_ok,
        r.n_folds_total
    );
}

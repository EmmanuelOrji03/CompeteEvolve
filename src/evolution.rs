//! Evolutionary operators: parent selection, generation, evaluation, archiving.

use anyhow::Result;
use rand::rngs::StdRng;
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::archive::{Archive, Candidate, now_secs};
use crate::config::Config;
use crate::fitness::{self, Baseline};
use crate::gemini::{GeminiClient, truncate};
use crate::harness::{Harness, HarnessResult, TaskInfo, Template, extract_block};
use crate::prompts;

pub struct Shared {
    pub cfg: Config,
    pub gemini: Arc<GeminiClient>,
    pub harness: Arc<Harness>,
    pub task: TaskInfo,
    pub template: Template,
    pub baseline: Baseline,
    pub baseline_result: HarnessResult,
    pub sklearn_result: Option<HarnessResult>,
    pub archive: Arc<Mutex<Archive>>,
    pub rng: Arc<Mutex<StdRng>>,
    pub goal: String,
    pub context_excerpt: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operator {
    Mutate,
    Crossover,
}

impl Operator {
    pub fn parse(s: &str) -> Self {
        if s.trim().eq_ignore_ascii_case("crossover") {
            Operator::Crossover
        } else {
            Operator::Mutate
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Operator::Mutate => "mutate",
            Operator::Crossover => "crossover",
        }
    }
}

pub struct EvolveRequest {
    pub agent: String,
    pub round: usize,
    pub operator: Operator,
    pub n_samples: usize,
    pub guidance: Option<String>,
}

pub async fn evolve(shared: &Shared, req: &EvolveRequest) -> Result<Value> {
    let mut samples = Vec::with_capacity(req.n_samples);

    for _ in 0..req.n_samples {
        // Locks are never held across an await.
        let (prompt, parents, operator) = {
            let archive = shared.archive.lock().await;
            let mut rng = shared.rng.lock().await;
            match req.operator {
                Operator::Crossover => match archive.sample_pair(
                    &mut rng,
                    shared.cfg.seed_parent_probability,
                    &req.agent,
                ) {
                    Some((a, b)) => (
                        prompts::crossover_prompt(shared, &a, &b, req.guidance.as_deref()),
                        vec![a.id, b.id],
                        Operator::Crossover,
                    ),
                    None => {
                        let p = archive.sample_parent(&mut rng, 1.0, None, None);
                        (
                            prompts::mutate_prompt(shared, &p, req.guidance.as_deref()),
                            vec![p.id],
                            Operator::Mutate,
                        )
                    }
                },
                Operator::Mutate => {
                    let p = archive.sample_parent(
                        &mut rng,
                        shared.cfg.seed_parent_probability,
                        None,
                        None,
                    );
                    (
                        prompts::mutate_prompt(shared, &p, req.guidance.as_deref()),
                        vec![p.id],
                        Operator::Mutate,
                    )
                }
            }
        };

        let generated = shared
            .gemini
            .generate_text(
                &shared.cfg.gemini.generation_model,
                Some(prompts::GENERATION_SYSTEM),
                &prompt,
                shared.cfg.gemini.temperature,
            )
            .await;
        let text = match generated {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[{}] generation failed: {e:#}", req.agent);
                samples.push(json!({ "status": "generation_failed", "error": truncate(&e.to_string(), 300) }));
                continue;
            }
        };

        let block = extract_block(&text);
        let outcome = evaluate_block(
            shared,
            &req.agent,
            req.round,
            operator.name(),
            parents,
            req.guidance.clone(),
            block,
        )
        .await?;
        samples.push(outcome);
    }

    let archive = shared.archive.lock().await;
    Ok(json!({
        "samples": samples,
        "your_best": archive.best_for_agent(&req.agent).map(Candidate::summary),
        "global_best": archive.best().map(Candidate::summary),
        "archive_size": archive.len(),
    }))
}

pub async fn evaluate_block(
    shared: &Shared,
    agent: &str,
    round: usize,
    operator: &str,
    parents: Vec<String>,
    guidance: Option<String>,
    block: String,
) -> Result<Value> {
    if !block.contains("class Model") {
        let mut archive = shared.archive.lock().await;
        archive.rejected_invalid_shape += 1;
        archive.log_event(json!({ "event": "rejected", "agent": agent, "round": round, "reason": "no class Model" }));
        return Ok(json!({
            "status": "rejected",
            "reason": "the generated block does not define `class Model`",
            "preview": truncate(&block, 300),
        }));
    }

    let (id, code_path) = {
        let mut archive = shared.archive.lock().await;
        let sim = archive.max_similarity(&block);
        if sim >= shared.cfg.novelty_reject_threshold {
            archive.rejected_near_duplicates += 1;
            archive.log_event(json!({ "event": "rejected", "agent": agent, "round": round, "reason": "near_duplicate", "similarity": sim }));
            return Ok(json!({
                "status": "rejected",
                "reason": format!("near-duplicate of an archived candidate or the seed (similarity {sim:.3}); make a substantive change"),
            }));
        }
        let id = archive.next_id(agent, round);
        let path = archive.candidates_dir().join(format!("{id}.py"));
        (id, path)
    };

    let source = shared.template.assemble(&block);
    tokio::fs::write(&code_path, &source).await?;

    println!("[{agent}] evaluating {id} ({operator}, parents {parents:?})...");
    let result = shared
        .harness
        .evaluate_file(&shared.task.task, &code_path)
        .await?;

    let performance = fitness::performance(&result, &shared.baseline, &shared.cfg.fitness);
    let valid = fitness::is_valid(&result, &shared.baseline, &shared.cfg.fitness);

    let mut archive = shared.archive.lock().await;
    let sims = archive.novelty_similarities(agent, &block);
    let novelty = fitness::novelty(&sims);
    let score = if valid {
        fitness::rank_score(performance, novelty, &shared.cfg.fitness)
    } else {
        0.0
    };

    let candidate = Candidate {
        id: id.clone(),
        agent: agent.to_string(),
        round,
        operator: operator.to_string(),
        parents,
        guidance,
        block,
        code_path: code_path.to_string_lossy().into_owned(),
        result,
        performance,
        novelty,
        score,
        valid,
        created_at: now_secs(),
    };
    println!(
        "[{agent}] {id}: {} valid={} {} novelty {:.3} score {:.4}",
        candidate.result.status,
        valid,
        prompts::metrics_line(
            candidate.result.quality,
            candidate.result.total_time(),
            performance
        ),
        novelty,
        score
    );
    let summary = candidate.summary();
    archive.log_event(json!({ "event": "candidate", "candidate": summary }));
    archive.insert(candidate)?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use rand::SeedableRng;

    const VECTORISED_KNN: &str = r#"
class Model:
    def __init__(self, n_neighbors=5):
        self.k = int(n_neighbors)

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=np.float64)
        self.y_ = np.asarray(y)
        self.classes_, self.yi_ = np.unique(self.y_, return_inverse=True)
        self.sq_ = np.sum(self.X_ * self.X_, axis=1)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        d2 = np.sum(X * X, axis=1)[:, None] - 2.0 * X @ self.X_.T + self.sq_[None, :]
        idx = np.argpartition(d2, self.k - 1, axis=1)[:, : self.k]
        votes = self.yi_[idx]
        counts = np.zeros((X.shape[0], len(self.classes_)), dtype=np.int64)
        rows = np.arange(X.shape[0])
        for j in range(self.k):
            np.add.at(counts, (rows, votes[:, j]), 1)
        return self.classes_[np.argmax(counts, axis=1)]
"#;

    /// Real harness and archive, no LLM. Skips when Python or scikit-learn are missing.
    #[tokio::test]
    async fn evaluate_block_runs_harness_and_archives() {
        let cfg = Config::default();
        let Ok(harness) = Harness::new(&cfg.harness) else {
            return;
        };
        let task = match harness.describe("knn").await {
            Ok(t) => t,
            Err(e) => {
                eprintln!("skipping: harness unavailable ({e:#})");
                return;
            }
        };
        let source = std::fs::read_to_string(&task.seed_file).unwrap();
        let template = Template::parse(&source).unwrap();
        let baseline_result = harness.evaluate_seed("knn").await.unwrap();
        assert!(baseline_result.is_ok(), "{baseline_result:?}");
        let baseline = Baseline {
            quality: baseline_result.quality.unwrap(),
            time: baseline_result.total_time().unwrap(),
        };

        let dir = std::env::temp_dir().join(format!("ce-evolution-test-{}", now_secs()));
        let archive = Archive::open(&dir, &template.block, vec!["a1".into(), "a2".into()]).unwrap();
        let shared = Shared {
            gemini: Arc::new(GeminiClient::new("dummy".into(), cfg.gemini.clone()).unwrap()),
            harness: Arc::new(harness),
            task,
            template: template.clone(),
            baseline,
            baseline_result,
            sklearn_result: None,
            archive: Arc::new(Mutex::new(archive)),
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(0))),
            goal: "test".into(),
            context_excerpt: None,
            cfg,
        };

        // Near-duplicate of the seed: rejected before the harness runs.
        let dup = evaluate_block(
            &shared,
            "a1",
            1,
            "manual",
            vec!["seed".into()],
            None,
            format!("{}\n# tweak\n", template.block),
        )
        .await
        .unwrap();
        assert_eq!(dup["status"], "rejected", "{dup}");

        let junk = evaluate_block(
            &shared,
            "a1",
            1,
            "manual",
            vec![],
            None,
            "def f():\n    return 1\n".into(),
        )
        .await
        .unwrap();
        assert_eq!(junk["status"], "rejected", "{junk}");

        let out = evaluate_block(
            &shared,
            "a1",
            1,
            "manual",
            vec!["seed".into()],
            None,
            VECTORISED_KNN.into(),
        )
        .await
        .unwrap();
        assert_eq!(out["status"], "ok", "{out}");
        assert_eq!(out["valid"], true, "{out}");
        assert!(out["performance"].as_f64().unwrap() > 0.0);
        assert!(out["novelty"].as_f64().unwrap() > 0.0);

        let archive = shared.archive.lock().await;
        assert_eq!(archive.len(), 1);
        assert_eq!(archive.rejected_near_duplicates, 1);
        assert_eq!(archive.rejected_invalid_shape, 1);
        assert_eq!(archive.best().unwrap().agent, "a1");
        assert_eq!(archive.standings()[0].reward, 1);
        drop(archive);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

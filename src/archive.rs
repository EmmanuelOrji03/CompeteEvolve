//! Persistent archive of every evaluated candidate, appended to
//! `runs/<run-id>/candidates.jsonl`. Reopening a run directory reloads it.

use anyhow::{Context, Result};
use rand::Rng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::fitness::{cosine, embed, reward_for_position};
use crate::harness::{HarnessResult, round4, round5};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    pub agent: String,
    pub round: usize,
    pub operator: String,
    pub parents: Vec<String>,
    pub guidance: Option<String>,
    pub block: String,
    pub code_path: String,
    pub result: HarnessResult,
    pub performance: f64,
    pub novelty: f64,
    pub score: f64,
    pub valid: bool,
    pub created_at: u64,
}

impl Candidate {
    pub fn summary(&self) -> Value {
        let mut s = self.result.summary();
        s["id"] = json!(self.id);
        s["agent"] = json!(self.agent);
        s["round"] = json!(self.round);
        s["operator"] = json!(self.operator);
        s["parents"] = json!(self.parents);
        s["valid"] = json!(self.valid);
        s["performance"] = json!(round4(self.performance));
        s["novelty"] = json!(round4(self.novelty));
        s["score"] = json!(round4(self.score));
        s
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Standing {
    pub agent: String,
    pub rank_position: usize,
    pub reward: u8,
    pub best_score: f64,
    pub best_performance: f64,
    pub best_id: Option<String>,
    pub candidates: usize,
    pub valid_candidates: usize,
}

#[derive(Debug, Clone)]
pub struct Parent {
    pub id: String,
    pub label: String,
    pub block: String,
    pub metrics: Value,
}

pub struct Archive {
    dir: PathBuf,
    seed_block: String,
    seed_embedding: Vec<f32>,
    entries: Vec<Candidate>,
    embeddings: Vec<Vec<f32>>,
    agents: Vec<String>,
    counter: usize,
    pub rejected_near_duplicates: usize,
    pub rejected_invalid_shape: usize,
    candidates_file: File,
    events_file: File,
}

impl Archive {
    pub fn open(dir: &Path, seed_block: &str, agents: Vec<String>) -> Result<Self> {
        fs::create_dir_all(dir.join("candidates"))
            .with_context(|| format!("creating {}", dir.display()))?;
        let candidates_path = dir.join("candidates.jsonl");

        let mut entries = Vec::new();
        if candidates_path.exists() {
            let reader = BufReader::new(File::open(&candidates_path)?);
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<Candidate>(&line) {
                    Ok(c) => entries.push(c),
                    Err(e) => eprintln!("[archive] skipping unreadable line: {e}"),
                }
            }
            if !entries.is_empty() {
                println!(
                    "[archive] resumed {} candidates from {}",
                    entries.len(),
                    candidates_path.display()
                );
            }
        }
        let embeddings = entries.iter().map(|c| embed(&c.block)).collect();
        let counter = entries.len();

        let candidates_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&candidates_path)?;
        let events_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("events.jsonl"))?;

        Ok(Self {
            dir: dir.to_path_buf(),
            seed_block: seed_block.to_string(),
            seed_embedding: embed(seed_block),
            entries,
            embeddings,
            agents,
            counter,
            rejected_near_duplicates: 0,
            rejected_invalid_shape: 0,
            candidates_file,
            events_file,
        })
    }

    pub fn candidates_dir(&self) -> PathBuf {
        self.dir.join("candidates")
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn entries(&self) -> &[Candidate] {
        &self.entries
    }

    pub fn seed_block(&self) -> &str {
        &self.seed_block
    }

    /// Must be called under the archive lock so IDs never collide across agents.
    pub fn next_id(&mut self, agent: &str, round: usize) -> String {
        self.counter += 1;
        format!("{agent}-r{round}-c{:03}", self.counter)
    }

    pub fn log_event(&mut self, mut event: Value) {
        event["ts"] = json!(now_secs());
        let _ = writeln!(self.events_file, "{event}");
    }

    pub fn insert(&mut self, candidate: Candidate) -> Result<()> {
        writeln!(
            self.candidates_file,
            "{}",
            serde_json::to_string(&candidate)?
        )?;
        self.candidates_file.flush()?;
        self.embeddings.push(embed(&candidate.block));
        self.entries.push(candidate);
        Ok(())
    }

    pub fn max_similarity(&self, block: &str) -> f64 {
        let e = embed(block);
        let mut best = cosine(&e, &self.seed_embedding);
        for other in &self.embeddings {
            best = best.max(cosine(&e, other));
        }
        best
    }

    /// The paper's three novelty terms: seed, own previous candidate, rivals' best.
    pub fn novelty_similarities(&self, agent: &str, block: &str) -> Vec<f64> {
        let e = embed(block);
        let mut sims = vec![cosine(&e, &self.seed_embedding)];
        if let Some(prev) = self.latest_for_agent(agent) {
            sims.push(cosine(&e, &embed(&prev.block)));
        }
        let rivals: Vec<f64> = self
            .agents
            .iter()
            .filter(|a| a.as_str() != agent)
            .filter_map(|a| self.best_for_agent(a))
            .map(|c| cosine(&e, &embed(&c.block)))
            .collect();
        if !rivals.is_empty() {
            sims.push(rivals.iter().sum::<f64>() / rivals.len() as f64);
        }
        sims
    }

    fn better(a: &Candidate, b: &Candidate) -> bool {
        (a.valid, a.score, a.performance) > (b.valid, b.score, b.performance)
    }

    pub fn best(&self) -> Option<&Candidate> {
        self.entries
            .iter()
            .filter(|c| c.valid)
            .reduce(|a, b| if Self::better(b, a) { b } else { a })
    }

    pub fn best_for_agent(&self, agent: &str) -> Option<&Candidate> {
        self.entries
            .iter()
            .filter(|c| c.valid && c.agent == agent)
            .reduce(|a, b| if Self::better(b, a) { b } else { a })
    }

    pub fn latest_for_agent(&self, agent: &str) -> Option<&Candidate> {
        self.entries.iter().rev().find(|c| c.agent == agent)
    }

    pub fn top(&self, n: usize, agent: Option<&str>) -> Vec<&Candidate> {
        let mut v: Vec<&Candidate> = self
            .entries
            .iter()
            .filter(|c| agent.is_none_or(|a| c.agent == a))
            .collect();
        v.sort_by(|a, b| {
            (b.valid, b.score, b.performance)
                .partial_cmp(&(a.valid, a.score, a.performance))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v.truncate(n.max(1));
        v
    }

    pub fn standings(&self) -> Vec<Standing> {
        let mut rows: Vec<Standing> = self
            .agents
            .iter()
            .map(|a| {
                let best = self.best_for_agent(a);
                let best_any = self.top(1, Some(a)).first().copied();
                Standing {
                    agent: a.clone(),
                    rank_position: 0,
                    reward: 0,
                    best_score: best.map(|c| c.score).unwrap_or(0.0),
                    best_performance: best.or(best_any).map(|c| c.performance).unwrap_or(0.0),
                    best_id: best.map(|c| c.id.clone()),
                    candidates: self.entries.iter().filter(|c| &c.agent == a).count(),
                    valid_candidates: self
                        .entries
                        .iter()
                        .filter(|c| &c.agent == a && c.valid)
                        .count(),
                }
            })
            .collect();
        rows.sort_by(|x, y| {
            (y.best_id.is_some(), y.best_score, y.best_performance)
                .partial_cmp(&(x.best_id.is_some(), x.best_score, x.best_performance))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| x.agent.cmp(&y.agent))
        });
        let n = rows.len();
        for (i, row) in rows.iter_mut().enumerate() {
            row.rank_position = i + 1;
            row.reward = reward_for_position(i + 1, n);
        }
        rows
    }

    fn seed_parent(&self) -> Parent {
        Parent {
            id: "seed".into(),
            label: "seed (original implementation)".into(),
            block: self.seed_block.clone(),
            metrics: json!({ "note": "baseline; performance 1.0 by definition" }),
        }
    }

    fn to_parent(c: &Candidate) -> Parent {
        Parent {
            id: c.id.clone(),
            label: format!("{} by {} ({})", c.id, c.agent, c.operator),
            block: c.block.clone(),
            metrics: json!({
                "quality": c.result.quality.map(round4),
                "time": c.result.total_time().map(round5),
                "performance": round4(c.performance),
                "novelty": round4(c.novelty),
                "score": round4(c.score),
            }),
        }
    }

    /// Score-weighted sampling over valid candidates; `seed_probability` forces the seed.
    pub fn sample_parent(
        &self,
        rng: &mut StdRng,
        seed_probability: f64,
        only_agents_other_than: Option<&str>,
        exclude_id: Option<&str>,
    ) -> Parent {
        let pool: Vec<&Candidate> = self
            .entries
            .iter()
            .filter(|c| c.valid)
            .filter(|c| only_agents_other_than.is_none_or(|a| c.agent != a))
            .filter(|c| exclude_id.is_none_or(|id| c.id != id))
            .collect();
        if pool.is_empty() || rng.random::<f64>() < seed_probability {
            return self.seed_parent();
        }
        let weights: Vec<f64> = pool.iter().map(|c| c.score + 0.05).collect();
        let total: f64 = weights.iter().sum();
        let mut x = rng.random::<f64>() * total;
        for (c, w) in pool.iter().zip(&weights) {
            if x < *w {
                return Self::to_parent(c);
            }
            x -= w;
        }
        Self::to_parent(pool[pool.len() - 1])
    }

    /// Two distinct parents, the second preferably from a rival. None if only the seed exists.
    pub fn sample_pair(
        &self,
        rng: &mut StdRng,
        seed_probability: f64,
        agent: &str,
    ) -> Option<(Parent, Parent)> {
        let a = self.sample_parent(rng, seed_probability, None, None);
        let mut b = self.sample_parent(rng, 0.0, Some(agent), Some(&a.id));
        if b.id == a.id {
            b = self.sample_parent(rng, 0.0, None, Some(&a.id));
        }
        if b.id == a.id {
            if a.id == "seed" {
                return None;
            }
            b = self.seed_parent();
        }
        Some((a, b))
    }

    pub fn write_summary(&self, extra: Value) -> Result<PathBuf> {
        let mut summary = json!({
            "candidates": self.entries.len(),
            "valid_candidates": self.entries.iter().filter(|c| c.valid).count(),
            "rejected_near_duplicates": self.rejected_near_duplicates,
            "rejected_invalid_shape": self.rejected_invalid_shape,
            "standings": self.standings(),
            "best": self.best().map(Candidate::summary),
            "top_5": self.top(5, None).iter().map(|c| c.summary()).collect::<Vec<_>>(),
        });
        if let (Some(dst), Some(src)) = (summary.as_object_mut(), extra.as_object()) {
            for (k, v) in src {
                dst.insert(k.clone(), v.clone());
            }
        }
        let path = self.dir.join("summary.json");
        fs::write(&path, serde_json::to_string_pretty(&summary)?)?;
        Ok(path)
    }
}

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn cand(id: &str, agent: &str, block: &str, score: f64, valid: bool) -> Candidate {
        Candidate {
            id: id.into(),
            agent: agent.into(),
            round: 1,
            operator: "mutate".into(),
            parents: vec!["seed".into()],
            guidance: None,
            block: block.into(),
            code_path: String::new(),
            result: HarnessResult {
                status: "ok".into(),
                ..Default::default()
            },
            performance: score * 2.0,
            novelty: 0.3,
            score,
            valid,
            created_at: 0,
        }
    }

    #[test]
    fn standings_rank_best_first_and_reward_top_half() {
        let dir = std::env::temp_dir().join(format!("ce-archive-test-{}", now_secs()));
        let mut a = Archive::open(
            &dir,
            "class Model: pass",
            vec!["a1".into(), "a2".into(), "a3".into()],
        )
        .unwrap();
        a.insert(cand("x", "a1", "class Model: v = 1", 0.4, true))
            .unwrap();
        a.insert(cand("y", "a2", "class Model: v = 2", 0.7, true))
            .unwrap();
        a.insert(cand("z", "a3", "class Model: v = 3", 0.9, false))
            .unwrap();
        let s = a.standings();
        assert_eq!(s[0].agent, "a2");
        assert_eq!(s[0].reward, 1);
        assert_eq!(s[1].agent, "a1");
        assert_eq!(s[1].reward, 1);
        assert_eq!(s[2].agent, "a3");
        assert_eq!(s[2].reward, 0);
        assert_eq!(a.best().unwrap().id, "y");

        let mut rng = StdRng::seed_from_u64(1);
        let p = a.sample_parent(&mut rng, 0.0, None, None);
        assert!(
            p.id == "x" || p.id == "y",
            "invalid candidates are never parents"
        );
        let (pa, pb) = a.sample_pair(&mut rng, 0.0, "a1").unwrap();
        assert_ne!(pa.id, pb.id);

        drop(a);
        let b = Archive::open(
            &dir,
            "class Model: pass",
            vec!["a1".into(), "a2".into(), "a3".into()],
        )
        .unwrap();
        assert_eq!(b.len(), 3);
        let _ = fs::remove_dir_all(&dir);
    }
}

//! Fitness, novelty, rank and reward. Unlike the first paper draft, lower time
//! scores higher, novelty is `1 - similarity`, and the top half is rewarded.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::config::FitnessConfig;
use crate::harness::HarnessResult;

/// The seed's measured quality and time (alpha in the paper).
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Baseline {
    pub quality: f64,
    pub time: f64,
}

/// performance = 2^-(SE+RE) * reliability * (Q/Q0)^qe * (T0/T)^se. 1.0 equals the baseline.
pub fn performance(r: &HarnessResult, base: &Baseline, w: &FitnessConfig) -> f64 {
    let (Some(quality), Some(time)) = (r.quality, r.total_time()) else {
        return 0.0;
    };
    if r.n_folds_total == 0 || r.n_folds_ok == 0 {
        return 0.0;
    }
    let reliability = r.n_folds_ok as f64 / r.n_folds_total as f64;
    let quality_ratio = (quality / base.quality.max(1e-9)).max(0.0);
    let speed_ratio = (base.time.max(1e-6) / time.max(1e-6)).clamp(0.01, 100.0);
    let penalty = 2f64.powi(-(r.error_count() as i32));
    let p = penalty
        * reliability
        * quality_ratio.powf(w.quality_exponent)
        * speed_ratio.powf(w.speed_exponent);
    if p.is_finite() { p } else { 0.0 }
}

pub fn is_valid(r: &HarnessResult, base: &Baseline, w: &FitnessConfig) -> bool {
    r.is_ok()
        && r.syntax_error.is_none()
        && r.runtime_errors.is_empty()
        && r.n_folds_total > 0
        && r.n_folds_ok == r.n_folds_total
        && r.quality
            .is_some_and(|q| q >= base.quality - w.quality_tolerance)
}

/// novelty = 1 - mean(similarities)
pub fn novelty(similarities: &[f64]) -> f64 {
    if similarities.is_empty() {
        return 1.0;
    }
    let mean = similarities.iter().sum::<f64>() / similarities.len() as f64;
    (1.0 - mean).clamp(0.0, 1.0)
}

/// Maps [0, inf) to [0, 1); baseline performance 1.0 maps to 0.5.
pub fn squash(x: f64) -> f64 {
    if x.is_finite() && x >= 0.0 {
        x / (1.0 + x)
    } else {
        0.0
    }
}

/// rank score = squash(performance) * (1 - w + w * novelty)
pub fn rank_score(performance: f64, novelty: f64, w: &FitnessConfig) -> f64 {
    squash(performance) * (1.0 - w.novelty_weight + w.novelty_weight * novelty.clamp(0.0, 1.0))
}

/// Top half of `n` agents (1-based position) earns 1.
pub fn reward_for_position(position: usize, n: usize) -> u8 {
    let cutoff = n.div_ceil(2).max(1);
    u8::from(position >= 1 && position <= cutoff)
}

pub const EMBEDDING_DIM: usize = 512;

fn tokens(code: &str) -> impl Iterator<Item = String> + '_ {
    code.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
}

/// Unit-norm hashed unigram and bigram counts.
pub fn embed(code: &str) -> Vec<f32> {
    let mut v = vec![0f32; EMBEDDING_DIM];
    let toks: Vec<String> = tokens(code).collect();
    for (i, t) in toks.iter().enumerate() {
        v[bucket(t)] += 1.0;
        if let Some(next) = toks.get(i + 1) {
            v[bucket(&format!("{t} {next}"))] += 1.0;
        }
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

fn bucket(token: &str) -> usize {
    let mut h = DefaultHasher::new();
    token.hash(&mut h);
    (h.finish() as usize) % EMBEDDING_DIM
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x * y) as f64)
        .sum::<f64>()
        .clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> FitnessConfig {
        FitnessConfig::default()
    }

    fn result(quality: f64, time: f64) -> HarnessResult {
        HarnessResult {
            status: "ok".into(),
            n_folds_total: 6,
            n_folds_ok: 6,
            fit_time: Some(time),
            predict_time: Some(0.0),
            quality: Some(quality),
            quality_std: Some(0.0),
            ..Default::default()
        }
    }

    #[test]
    fn faster_equally_accurate_beats_baseline() {
        let base = Baseline {
            quality: 0.9,
            time: 1.0,
        };
        let same = performance(&result(0.9, 1.0), &base, &cfg());
        let faster = performance(&result(0.9, 0.5), &base, &cfg());
        let slower = performance(&result(0.9, 2.0), &base, &cfg());
        assert!((same - 1.0).abs() < 1e-9);
        assert!(faster > same, "2x speedup must score higher");
        assert!(slower < same, "2x slowdown must score lower");
    }

    #[test]
    fn accuracy_loss_is_expensive() {
        let base = Baseline {
            quality: 0.9,
            time: 1.0,
        };
        // 10x faster but 10% less accurate should not beat a modest, lossless speedup.
        let fast_lossy = performance(&result(0.81, 0.1), &base, &cfg());
        let modest = performance(&result(0.9, 0.5), &base, &cfg());
        assert!(!is_valid(&result(0.81, 0.1), &base, &cfg()));
        assert!(is_valid(&result(0.9, 0.5), &base, &cfg()));
        assert!(
            modest > fast_lossy * 0.3,
            "valid modest gain must remain competitive"
        );
    }

    #[test]
    fn errors_penalise_and_invalidate() {
        let base = Baseline {
            quality: 0.9,
            time: 1.0,
        };
        let mut r = result(0.9, 0.5);
        r.runtime_errors.push("boom".into());
        r.n_folds_ok = 3;
        let p = performance(&r, &base, &cfg());
        assert!(p < performance(&result(0.9, 0.5), &base, &cfg()) / 2.0);
        assert!(!is_valid(&r, &base, &cfg()));
        assert_eq!(
            performance(&HarnessResult::failed("timeout", "x".into()), &base, &cfg()),
            0.0
        );
    }

    #[test]
    fn verbatim_copy_has_zero_novelty() {
        let code = "class Model:\n    def fit(self, X, y):\n        return self\n";
        let e = embed(code);
        let sim = cosine(&e, &embed(code));
        assert!(sim > 0.999);
        assert!(novelty(&[sim]) < 1e-3);
        let other = "def totally_different(a, b):\n    return a * b + 42\n";
        assert!(cosine(&e, &embed(other)) < 0.5);
        assert!(novelty(&[]) == 1.0);
    }

    #[test]
    fn rank_score_prefers_performance_and_novelty() {
        let w = cfg();
        assert!(rank_score(2.0, 0.5, &w) > rank_score(1.0, 0.5, &w));
        assert!(rank_score(1.0, 0.8, &w) > rank_score(1.0, 0.2, &w));
        assert_eq!(rank_score(0.0, 1.0, &w), 0.0);
        assert!((squash(1.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn best_agent_is_rewarded() {
        assert_eq!(reward_for_position(1, 3), 1);
        assert_eq!(reward_for_position(2, 3), 1);
        assert_eq!(reward_for_position(3, 3), 0);
        assert_eq!(reward_for_position(1, 1), 1);
        assert_eq!(reward_for_position(2, 2), 0);
        assert_eq!(reward_for_position(3, 4), 0);
    }
}

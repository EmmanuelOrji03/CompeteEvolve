//! Bridge to `harness/evaluate.py`, the only source of measurements, and the
//! EVOLVE-BLOCK template.

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use crate::config::HarnessConfig;
use crate::gemini::truncate;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct HarnessResult {
    pub status: String,
    pub syntax_error: Option<String>,
    pub runtime_errors: Vec<String>,
    pub n_folds_total: usize,
    pub n_folds_ok: usize,
    pub fit_time: Option<f64>,
    pub predict_time: Option<f64>,
    pub quality: Option<f64>,
    pub quality_std: Option<f64>,
    pub per_dataset: Value,
    pub code_length: usize,
}

impl HarnessResult {
    pub fn total_time(&self) -> Option<f64> {
        Some(self.fit_time? + self.predict_time?)
    }

    pub fn is_ok(&self) -> bool {
        self.status == "ok"
    }

    /// `SE + RE` in the paper.
    pub fn error_count(&self) -> usize {
        usize::from(self.syntax_error.is_some()) + self.runtime_errors.len()
    }

    pub fn failed(status: &str, message: String) -> Self {
        Self {
            status: status.to_string(),
            runtime_errors: vec![message],
            ..Default::default()
        }
    }

    pub fn summary(&self) -> Value {
        let mut errors: Vec<String> = Vec::new();
        if let Some(e) = &self.syntax_error {
            errors.push(format!("syntax error: {e}"));
        }
        errors.extend(self.runtime_errors.iter().take(3).map(|e| truncate(e, 400)));
        json!({
            "status": self.status,
            "quality": self.quality.map(round4),
            "quality_std": self.quality_std.map(round4),
            "fit_time": self.fit_time.map(round5),
            "predict_time": self.predict_time.map(round5),
            "folds_ok": format!("{}/{}", self.n_folds_ok, self.n_folds_total),
            "errors": errors,
        })
    }
}

pub fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

pub fn round5(x: f64) -> f64 {
    (x * 100_000.0).round() / 100_000.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskInfo {
    pub task: String,
    pub kind: String,
    pub datasets: Vec<String>,
    pub description: String,
    pub seed_file: String,
}

impl TaskInfo {
    pub fn contract(&self) -> String {
        if self.kind == "clustering" {
            "The file must define `class Model` constructed as `Model(n_clusters=k)`, with \
             `fit(X)` (returns self) and `predict(X)` returning a 1-D integer array of cluster \
             labels in [0, k)."
                .into()
        } else {
            "The file must define `class Model` constructed as `Model()`, with `fit(X, y)` \
             (returns self; y holds integer class labels) and `predict(X)` returning a 1-D \
             array of predicted class labels."
                .into()
        }
    }
}

pub struct Harness {
    python: String,
    script: PathBuf,
    cfg: HarnessConfig,
}

impl Harness {
    pub fn new(cfg: &HarnessConfig) -> Result<Self> {
        let script = locate_script(&cfg.script)?;
        Ok(Self {
            python: cfg.python.clone(),
            script,
            cfg: cfg.clone(),
        })
    }

    pub fn script_path(&self) -> &Path {
        &self.script
    }

    /// `Ok(None)` means the harness timed out and was killed.
    async fn run(&self, args: &[String], timeout: Duration) -> Result<Option<Value>> {
        let mut cmd = tokio::process::Command::new(&self.python);
        cmd.arg(&self.script)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let child = cmd.spawn().with_context(|| {
            format!(
                "spawning `{}` for the harness (set PYTHON in .env if needed)",
                self.python
            )
        })?;
        let output = match tokio::time::timeout(timeout, child.wait_with_output()).await {
            Ok(out) => out.context("waiting for the harness process")?,
            Err(_) => return Ok(None),
        };
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let last_line = stdout
            .lines()
            .rev()
            .find(|l| l.trim_start().starts_with('{') || l.trim_start().starts_with('['));
        match last_line.and_then(|l| serde_json::from_str::<Value>(l).ok()) {
            Some(v) => Ok(Some(v)),
            None => Err(anyhow!(
                "harness produced no JSON (exit {:?}).\nstdout: {}\nstderr: {}",
                output.status.code(),
                truncate(stdout.trim(), 500),
                truncate(stderr.trim(), 1500)
            )),
        }
    }

    fn common_args(&self, task: &str) -> Vec<String> {
        vec![
            "--task".into(),
            task.into(),
            "--folds".into(),
            self.cfg.folds.to_string(),
            "--repeats".into(),
            self.cfg.repeats.to_string(),
            "--seed".into(),
            self.cfg.seed.to_string(),
        ]
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(self.cfg.timeout_seconds.max(5))
    }

    pub async fn list_tasks(&self) -> Result<Vec<TaskInfo>> {
        let v = self
            .run(&["--list-tasks".into()], Duration::from_secs(60))
            .await?
            .ok_or_else(|| anyhow!("harness timed out listing tasks"))?;
        Ok(serde_json::from_value(v)?)
    }

    pub async fn describe(&self, task: &str) -> Result<TaskInfo> {
        let args = vec![
            "--task".to_string(),
            task.to_string(),
            "--seed-file".into(),
            "--describe".into(),
        ];
        let v = self
            .run(&args, Duration::from_secs(60))
            .await?
            .ok_or_else(|| anyhow!("harness timed out"))?;
        serde_json::from_value(v.clone())
            .with_context(|| format!("unexpected describe output: {v}"))
    }

    async fn evaluate_with(&self, task: &str, extra: &[String]) -> Result<HarnessResult> {
        let mut args = self.common_args(task);
        args.extend_from_slice(extra);
        match self.run(&args, self.timeout()).await {
            Ok(Some(v)) => {
                let r: HarnessResult = serde_json::from_value(v.clone())
                    .with_context(|| format!("unexpected harness output: {v}"))?;
                Ok(r)
            }
            Ok(None) => Ok(HarnessResult::failed(
                "timeout",
                format!(
                    "evaluation exceeded {} seconds and was killed",
                    self.cfg.timeout_seconds
                ),
            )),
            Err(e) => Ok(HarnessResult::failed("harness_error", format!("{e:#}"))),
        }
    }

    pub async fn evaluate_file(&self, task: &str, path: &Path) -> Result<HarnessResult> {
        self.evaluate_with(
            task,
            &["--candidate".into(), path.to_string_lossy().into_owned()],
        )
        .await
    }

    pub async fn evaluate_seed(&self, task: &str) -> Result<HarnessResult> {
        self.evaluate_with(task, &["--seed-file".into()]).await
    }

    pub async fn evaluate_sklearn(&self, task: &str) -> Result<HarnessResult> {
        self.evaluate_with(task, &["--sklearn-reference".into()])
            .await
    }
}

fn locate_script(configured: &str) -> Result<PathBuf> {
    let direct = PathBuf::from(configured);
    if direct.exists() {
        // Avoid canonicalize: it adds a `\\?\` prefix on Windows.
        return Ok(std::env::current_dir()
            .map(|cwd| cwd.join(&direct))
            .unwrap_or(direct));
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join(configured);
    if manifest.exists() {
        return Ok(manifest);
    }
    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.ancestors().take(4) {
            let p = ancestor.join(configured);
            if p.exists() {
                return Ok(p);
            }
        }
    }
    bail!(
        "cannot find the harness script '{configured}'; run from the repository root or set harness.script in the config"
    )
}

pub const BLOCK_START: &str = "# EVOLVE-BLOCK-START";
pub const BLOCK_END: &str = "# EVOLVE-BLOCK-END";

/// A seed file split at the EVOLVE markers. Only `block` is ever regenerated.
#[derive(Debug, Clone)]
pub struct Template {
    pub header: String,
    pub block: String,
    pub footer: String,
}

impl Template {
    pub fn parse(source: &str) -> Result<Self> {
        let start = source
            .find(BLOCK_START)
            .ok_or_else(|| anyhow!("seed file has no {BLOCK_START} marker"))?;
        let end = source
            .find(BLOCK_END)
            .ok_or_else(|| anyhow!("seed file has no {BLOCK_END} marker"))?;
        if end < start {
            bail!("{BLOCK_END} appears before {BLOCK_START}");
        }
        let header = source[..start].to_string();
        let block = source[start + BLOCK_START.len()..end]
            .trim_matches('\n')
            .to_string();
        let footer = source[end + BLOCK_END.len()..].to_string();
        Ok(Self {
            header,
            block,
            footer,
        })
    }

    pub fn assemble(&self, block: &str) -> String {
        format!(
            "{}{BLOCK_START}\n{}\n{BLOCK_END}{}",
            self.header,
            block.trim_matches('\n'),
            self.footer
        )
    }
}

/// Strips fences; if the model echoed the whole file, keeps only the block.
pub fn extract_block(text: &str) -> String {
    let cleaned = strip_code_fences(text);
    if let (Some(s), Some(e)) = (cleaned.find(BLOCK_START), cleaned.find(BLOCK_END)) {
        if e > s {
            return cleaned[s + BLOCK_START.len()..e]
                .trim_matches('\n')
                .to_string();
        }
    }
    cleaned.trim_matches('\n').to_string()
}

pub fn strip_code_fences(text: &str) -> String {
    let trimmed = text.trim();
    if let Some(rest) = trimmed.strip_prefix("```") {
        let rest = rest.trim_start_matches(|c: char| c.is_alphanumeric() || c == '_' || c == '-');
        let rest = rest.strip_prefix('\n').unwrap_or(rest);
        if let Some(end) = rest.rfind("```") {
            return rest[..end].trim_end().to_string();
        }
        return rest.trim_end().to_string();
    }
    // Prose around fences: take the largest fenced block.
    let mut best: Option<&str> = None;
    let mut rest = trimmed;
    while let Some(open) = rest.find("```") {
        let after = &rest[open + 3..];
        let after = after.trim_start_matches(|c: char| c.is_alphanumeric() || c == '_' || c == '-');
        let after = after.strip_prefix('\n').unwrap_or(after);
        let Some(close) = after.find("```") else {
            break;
        };
        let body = &after[..close];
        if best.is_none_or(|b| body.len() > b.len()) {
            best = Some(body);
        }
        rest = &after[close + 3..];
    }
    best.map(|b| b.trim_end().to_string())
        .unwrap_or_else(|| trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: &str = "import numpy as np\n\n# EVOLVE-BLOCK-START\nclass Model:\n    pass\n# EVOLVE-BLOCK-END\n\nassert Model\n";

    #[test]
    fn template_roundtrip() {
        let t = Template::parse(SEED).unwrap();
        assert_eq!(t.block, "class Model:\n    pass");
        assert_eq!(t.assemble(&t.block), SEED);
    }

    #[test]
    fn extract_handles_fences_and_markers() {
        let fenced = "```python\nclass Model:\n    x = 1\n```";
        assert_eq!(extract_block(fenced), "class Model:\n    x = 1");

        let echoed = "Here you go:\n```python\nimport numpy as np\n# EVOLVE-BLOCK-START\nclass Model:\n    y = 2\n# EVOLVE-BLOCK-END\n```\nDone.";
        assert_eq!(extract_block(echoed), "class Model:\n    y = 2");

        assert_eq!(
            extract_block("class Model:\n    z = 3\n"),
            "class Model:\n    z = 3"
        );
    }
}

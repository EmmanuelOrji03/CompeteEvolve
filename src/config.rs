//! Run configuration (`competeevolve.json`, every field defaulted) and the CLI.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub agents: usize,
    pub rounds: usize,
    pub samples_per_evolve: usize,
    pub max_tool_calls_per_round: usize,
    /// Stop early once squash(best performance) reaches this value.
    pub mu_threshold: f64,
    /// Chance a mutation starts from the seed instead of an archived parent.
    pub seed_parent_probability: f64,
    /// Candidates at least this similar to the archive are rejected unevaluated.
    pub novelty_reject_threshold: f64,
    pub context_max_chars: usize,
    pub fitness: FitnessConfig,
    pub harness: HarnessConfig,
    pub gemini: GeminiConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FitnessConfig {
    /// Exponent on quality ratio. Larger makes accuracy loss costlier than speed gain.
    pub quality_exponent: f64,
    pub speed_exponent: f64,
    pub novelty_weight: f64,
    /// Valid only if quality >= baseline - tolerance.
    pub quality_tolerance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HarnessConfig {
    pub python: String,
    pub script: String,
    pub folds: usize,
    pub repeats: usize,
    pub seed: u64,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GeminiConfig {
    pub agent_model: String,
    pub generation_model: String,
    /// Shared by all agents. Free tier is about 10.
    pub requests_per_minute: f64,
    pub max_retries: u32,
    pub temperature: f64,
    pub request_timeout_seconds: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            agents: 3,
            rounds: 3,
            samples_per_evolve: 3,
            max_tool_calls_per_round: 5,
            mu_threshold: 0.7,
            seed_parent_probability: 0.2,
            novelty_reject_threshold: 0.98,
            context_max_chars: 12_000,
            fitness: FitnessConfig::default(),
            harness: HarnessConfig::default(),
            gemini: GeminiConfig::default(),
        }
    }
}

impl Default for FitnessConfig {
    fn default() -> Self {
        Self {
            quality_exponent: 4.0,
            speed_exponent: 1.0,
            novelty_weight: 0.5,
            quality_tolerance: 0.01,
        }
    }
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            python: "python".into(),
            script: "harness/evaluate.py".into(),
            folds: 3,
            repeats: 2,
            seed: 0,
            timeout_seconds: 180,
        }
    }
}

impl Default for GeminiConfig {
    fn default() -> Self {
        Self {
            agent_model: "gemini-3.7-flash".into(),
            generation_model: "gemini-3.7-flash".into(),
            requests_per_minute: 8.0,
            max_retries: 6,
            temperature: 0.9,
            request_timeout_seconds: 120,
        }
    }
}

impl Config {
    /// `GEMINI_MODEL` and `PYTHON` env vars override the file.
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let mut cfg = match path {
            Some(p) => Self::from_file(p)?,
            None => {
                let default = Path::new("competeevolve.json");
                if default.exists() {
                    Self::from_file(default)?
                } else {
                    Self::default()
                }
            }
        };
        if let Ok(model) = std::env::var("GEMINI_MODEL") {
            if !model.trim().is_empty() {
                cfg.gemini.agent_model = model.clone();
                cfg.gemini.generation_model = model;
            }
        }
        if let Ok(python) = std::env::var("PYTHON") {
            if !python.trim().is_empty() {
                cfg.harness.python = python;
            }
        }
        cfg.validate()?;
        Ok(cfg)
    }

    fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing config {}", path.display()))
    }

    fn validate(&self) -> Result<()> {
        if self.agents == 0 || self.rounds == 0 || self.samples_per_evolve == 0 {
            bail!("agents, rounds and samples_per_evolve must all be at least 1");
        }
        if !(0.0..=1.0).contains(&self.fitness.novelty_weight) {
            bail!("fitness.novelty_weight must be in [0, 1]");
        }
        if self.gemini.requests_per_minute <= 0.0 {
            bail!("gemini.requests_per_minute must be positive");
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct Cli {
    pub task: Option<String>,
    pub agents: Option<usize>,
    pub rounds: Option<usize>,
    pub config: Option<PathBuf>,
    pub goal: Option<String>,
    pub context_file: Option<PathBuf>,
    pub run_id: Option<String>,
    pub list_tasks: bool,
    pub baseline_only: bool,
    pub help: bool,
}

pub const USAGE: &str = "\
CompeteEvolve - competing LLM agents evolving ML algorithm implementations

USAGE:
    competeevolve [--task NAME] [OPTIONS]

OPTIONS:
    --task NAME          Task to evolve (kmeans | logistic_regression | knn). Prompts if omitted.
    --agents N           Number of competing agents (default from config, 3).
    --rounds N           Number of rounds (default from config, 3).
    --goal TEXT          Optimisation goal shown to the agents.
    --context FILE       Reference file (e.g. context/kmeans.md) excerpted into the first prompt.
    --config FILE        JSON config (default: competeevolve.json if present).
    --run-id ID          Name of the run directory under runs/ (default: <task>-<timestamp>).
    --list-tasks         Print the available tasks and exit.
    --baseline-only      Evaluate the seed and scikit-learn reference, then exit.
    -h, --help           Show this help.

ENVIRONMENT:
    GEMINI_API_KEY       Required.
    GEMINI_MODEL         Overrides both model names in the config.
    PYTHON               Python interpreter used for the harness (default: python).
";

impl Cli {
    pub fn parse<I: IntoIterator<Item = String>>(args: I) -> Result<Self> {
        let mut cli = Cli::default();
        let mut it = args.into_iter();
        while let Some(arg) = it.next() {
            let mut value = |name: &str| -> Result<String> {
                it.next()
                    .with_context(|| format!("{name} requires a value"))
            };
            match arg.as_str() {
                "--task" => cli.task = Some(value("--task")?),
                "--agents" => {
                    cli.agents = Some(
                        value("--agents")?
                            .parse()
                            .context("--agents must be an integer")?,
                    )
                }
                "--rounds" => {
                    cli.rounds = Some(
                        value("--rounds")?
                            .parse()
                            .context("--rounds must be an integer")?,
                    )
                }
                "--goal" => cli.goal = Some(value("--goal")?),
                "--context" => cli.context_file = Some(PathBuf::from(value("--context")?)),
                "--config" => cli.config = Some(PathBuf::from(value("--config")?)),
                "--run-id" => cli.run_id = Some(value("--run-id")?),
                "--list-tasks" => cli.list_tasks = true,
                "--baseline-only" => cli.baseline_only = true,
                "-h" | "--help" => cli.help = true,
                other => {
                    // Old CLI: a bare integer is the agent count.
                    if let Ok(n) = other.parse::<usize>() {
                        cli.agents = Some(n);
                    } else {
                        bail!("unknown argument '{other}'\n\n{USAGE}");
                    }
                }
            }
        }
        Ok(cli)
    }

    pub fn apply(&self, cfg: &mut Config) {
        if let Some(n) = self.agents {
            cfg.agents = n.max(1);
        }
        if let Some(n) = self.rounds {
            cfg.rounds = n.max(1);
        }
    }
}

mod agent;
mod archive;
mod config;
mod evolution;
mod fitness;
mod gemini;
mod harness;
mod orchestrator;
mod prompts;

use anyhow::Result;
use std::io::{self, Write};

use config::{Cli, Config, USAGE};
use harness::Harness;
use orchestrator::RunOptions;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let cli = Cli::parse(std::env::args().skip(1))?;
    if cli.help {
        print!("{USAGE}");
        return Ok(());
    }

    let mut cfg = Config::load(cli.config.as_deref())?;
    cli.apply(&mut cfg);

    let harness = Harness::new(&cfg.harness)?;
    let tasks = harness.list_tasks().await?;

    if cli.list_tasks {
        for t in &tasks {
            println!("{:<22} {:<15} {}", t.task, t.kind, t.description);
        }
        return Ok(());
    }

    let task = match cli.task.clone() {
        Some(t) => t,
        None => {
            println!("Select a task:");
            for (i, t) in tasks.iter().enumerate() {
                println!("  {}) {:<22} {}", i + 1, t.task, t.description);
            }
            print!("> ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let idx: usize = input.trim().parse().unwrap_or(1);
            tasks
                .get(idx.saturating_sub(1))
                .map(|t| t.task.clone())
                .unwrap_or_else(|| tasks[0].task.clone())
        }
    };

    let api_key = std::env::var("GEMINI_API_KEY")
        .ok()
        .filter(|k| !k.trim().is_empty());

    orchestrator::run(
        cfg,
        api_key,
        RunOptions {
            task,
            goal: cli.goal,
            context_file: cli.context_file,
            run_id: cli.run_id,
            baseline_only: cli.baseline_only,
        },
    )
    .await
}

mod agent;
mod benchmark;
mod context;
mod metaprompt;

use agent::database::{self, Connection};
use agent::{Agent, SharedTools};
use std::collections::HashMap;
use std::env;
use std::io::{self, Write};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    // `GEMINI_API_KEY` must be set (e.g. in a .env file).
    let api_key = env::var("GEMINI_API_KEY")?;

    // Let the user pick which markdown file to feed in as context; the
    // file's name (e.g. "kmeans") becomes the algorithm tag used to keep
    // this run's code/training/test data separate from any other
    // algorithm's data in the database.
    let (algorithm, context) = context::choose_markdown_file("context")?;
    println!("Optimizing algorithm: {algorithm}");

    let shared = SharedTools {
        db: Arc::new(Mutex::new(Connection::new())),
        agent_evaluations: Arc::new(Mutex::new(HashMap::new())),
    };

    maybe_upload_seed_data(&shared, &algorithm).await?;

    let refined_prompt = metaprompt::run_metaprompt(&context).await?;
    println!("The meta prompt is ready: {}", refined_prompt);

    let num_agents = match env::args().nth(1) {
        Some(arg) => arg.parse::<usize>()?,
        None => read_agent_count(),
    };

    let mut agents: Vec<Agent> = (0..num_agents)
        .map(|i| {
            let name = format!("agent-{}", i + 1);
            Agent::new(name, algorithm.clone(), api_key.clone(), shared.clone())
        })
        .collect();

    let handles = agents
        .iter_mut()
        .map(|agent| agent.send_owned(refined_prompt.clone()));

    let results = futures::future::join_all(handles).await;

    for (agent, result) in agents.iter().zip(results) {
        match result {
            Ok(reply) => println!("{}: {}", agent.name, reply),
            Err(e) => println!("{}: error - {e}", agent.name),
        }
    }

    // Report each agent's evolved code, the overall top performer, and
    // that top performer against the original baseline — as tables and
    // a saved chart. Only possible once at least one agent has actually
    // been evaluated; report why, rather than failing the whole run, if
    // that hasn't happened (e.g. the model never called the tools).
    match benchmark::run_and_print(&shared, &algorithm).await {
        Ok(_) => {}
        Err(e) => println!("\nSkipping benchmark report: {e}"),
    }

    Ok(())
}

/// Lets the user seed the chosen algorithm's baseline code and
/// training/test CSV data before agents start. `evolution` needs the
/// baseline + training/test data, `evaluator` needs all three, and the
/// benchmark report's "top evolved vs original" comparison needs the
/// baseline specifically — agents can also write any of these themselves
/// via the `database` tool if told to, but seeding them here guarantees
/// they exist for a reliable benchmark.
async fn maybe_upload_seed_data(shared: &SharedTools, algorithm: &str) -> anyhow::Result<()> {
    print!("Upload the original '{algorithm}' code and its training/test CSV data now? (y/n) ");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    if !input.trim().eq_ignore_ascii_case("y") {
        println!(
            "Skipping upload — make sure a baseline 'code' entry and \
             '{algorithm}' training_data/test_data already exist in the \
             database, or ask an agent to write them via the database tool."
        );
        return Ok(());
    }

    print!("Path to the original {algorithm} implementation (Python file): ");
    io::stdout().flush().unwrap();
    let mut code_path = String::new();
    io::stdin().read_line(&mut code_path)?;
    let code_path = code_path.trim();

    if code_path.is_empty() {
        println!("  skipped (no path given)");
    } else {
        let mut conn = shared.db.lock().await;
        // No "agent" field: this is what distinguishes the baseline from
        // any agent's evolved submission later (see benchmark.rs).
        let result = database::run(
            &mut conn,
            &serde_json::json!({
                "action": "write",
                "path": code_path,
                "category": "code",
                "algorithm": algorithm,
            }),
        );
        match result {
            Ok(r) => println!("  uploaded: {r}"),
            Err(e) => println!("  failed to upload original code: {e}"),
        }
    }

    for category in ["training_data", "test_data"] {
        print!("Path to {category} CSV for '{algorithm}': ");
        io::stdout().flush().unwrap();
        let mut path = String::new();
        io::stdin().read_line(&mut path)?;
        let path = path.trim();

        if path.is_empty() {
            println!("  skipped (no path given)");
            continue;
        }

        let mut conn = shared.db.lock().await;
        match database::upload_csv_file(&mut conn, path, category, algorithm) {
            Ok(result) => println!("  uploaded: {result}"),
            Err(e) => println!("  failed to upload {category}: {e}"),
        }
    }

    Ok(())
}

fn read_agent_count() -> usize {
    print!("How many agents? ");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().parse().unwrap_or(1)
}

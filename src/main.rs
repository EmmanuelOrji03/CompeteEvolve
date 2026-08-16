mod agent;
mod context;
mod metaprompt;

use agent::{Agent, SharedTools};
use agent::database::Connection;
use std::env;
use std::io::{self, Write};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let api_key = env::var("GEMINI_API_KEY")?;

    // Let the user pick which markdown file to feed in as context.
    let context = context::choose_markdown_file("context")?;

    let refined_prompt = metaprompt::run_metaprompt(&context).await?;
    println!("The meta prompt is ready: {}", refined_prompt);

    let num_agents = match env::args().nth(1) {
        Some(arg) => arg.parse::<usize>()?,
        None => read_agent_count(),
    };

    let shared = SharedTools {
        db: Arc::new(Mutex::new(Connection::new())),
    };

    let mut agents: Vec<Agent> = (0..num_agents)
        .map(|i| {
            let name = format!("agent-{}", i + 1);
            Agent::new(name, api_key.clone(), shared.clone())
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

    Ok(())
}

fn read_agent_count() -> usize {
    print!("How many agents? ");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().parse().unwrap_or(1)
}
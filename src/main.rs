mod metaprompt;
mod agent;
mod evolution;
mod benchmark;
mod evaluator;
mod reinforcement;
mod database;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let refined_prompt = metaprompt::run_metaprompt().await?;
    println!("The meta prompt ready: {}", refined_prompt);
    Ok(())
}

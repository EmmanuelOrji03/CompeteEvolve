use anyhow::Result;
use serde_json::Value;

/// Scores a candidate (e.g. a genome, policy, or model output).
pub fn run(args: &Value) -> Result<String> {
    let candidate = args["candidate"].as_str().unwrap_or_default();
    // ... actual evaluation logic here
    Ok(format!("evaluated '{candidate}': score=0.0"))
}
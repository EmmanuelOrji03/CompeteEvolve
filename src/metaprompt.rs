/// Represents a system for improving the quality of a user's prompt.
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};

#[derive(Serialize)]
struct GeminiRequest {
    #[serde(rename = "systemInstruction")]
    system_instruction: SystemInstruction,
    contents: Vec<Content>,
}

#[derive(Serialize)]
struct SystemInstruction {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct Part {
    text: String,
}

#[derive(Deserialize, Debug)]
struct GeminiResponse {
    candidates: Vec<Candidate>,
}

#[derive(Deserialize, Debug)]
struct Candidate {
    content: ResponseContent,
}

#[derive(Deserialize, Debug)]
struct ResponseContent {
    parts: Vec<ResponsePart>,
}

#[derive(Deserialize, Debug)]
struct ResponsePart {
    text: String,
}

/// Gets the raw prompt from the user.
pub fn get_prompt() -> String {
    println!("Enter your prompt");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to accept input");
    input.trim().to_string()
}

/// Takes the user's raw prompt plus optional markdown context and returns
/// a refined version via Gemini. Pass an empty string for `context` if
/// there's none to include.
pub async fn run_metaprompt(context: &str) -> Result<String> {
    let api_key = std::env::var("GEMINI_API_KEY")
        .map_err(|_| anyhow::anyhow!("GEMINI_API_KEY is not set"))?;
    let model = "gemini-3.6-flash";
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model, api_key
    );

    let prompt = get_prompt();

    let full_prompt = if context.trim().is_empty() {
        prompt
    } else {
        format!(
            "Reference material:\n{}\n\nUser prompt to refine:\n{}",
            context, prompt
        )
    };

    let request_body = GeminiRequest {
        system_instruction: SystemInstruction {
            parts: vec![Part {
                text: "You are a highly experienced prompt engineer. \
                       Use any reference material provided to ground your \
                       understanding, then polish and refine the user's prompt.
                       PROMPT FORMAT:
                       Return only a single straightforward response. Don't add extra additions to the polished prompt. 
                       Your response must be a SINGLE polished prompt. No multiple versions just the best

                       "
                    .to_string(),
            }],
        },
        contents: vec![Content {
            parts: vec![Part { text: full_prompt }],
        }],
    };

    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .json(&request_body)
        .send()
        .await?
        .json::<GeminiResponse>()
        .await?;

    let refined = response
        .candidates
        .get(0)
        .and_then(|c| c.content.parts.get(0))
        .map(|p| p.text.clone())
        .ok_or_else(|| anyhow::anyhow!("Gemini returned no candidates"))?;

    Ok(refined)
}

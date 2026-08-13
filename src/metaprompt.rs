/// Represents a system for improving the quality of users prompt
use serde::{Deserialize, Serialize};
use std::io::{self, Write};
# [derive (Serialize)]
struct GeminiRequest{
    #[serde(rename = "systemInstruction")]
    system_instruction : SystemInstruction,
    contents: Vec<Content>,
}
#[derive(Serialize)]
struct SystemInstruction{
    parts: Vec<Part>
}
#[derive(Serialize)]
struct Content{
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct Part{
    text: String
}
#[derive(Deserialize, Debug)]
struct GeminiResponse{
    candidates: Vec<Candidate>
}
#[derive(Deserialize, Debug)]
struct Candidate{
    content: Responsecontent
}
#[derive(Deserialize, Debug)]
struct Responsecontent{
    parts: Vec<Responsepart>
}
#[derive(Deserialize, Debug)]
struct Responsepart{
    text: String
}

/// Gets the prompt from the user
pub fn get_prompt() -> String{
    println!("Enter your prompt");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to accept input");
    input.trim().to_string()
}


/// The main function of the code 
pub async fn run_metaprompt() -> Result<String, Box< dyn std::error::Error>>{
 
    dotenvy::dotenv()?;
    /// Stores the API key
    let api_key= std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY is not set");
    /// Stores the Gemini Model we are using
    let model = "gemini-3.6-flash";
    /// The url to call the AI model via an API
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model, api_key
    );
    let prompt:String = get_prompt();

    /// The API request JSON
    let request_body = GeminiRequest{
        /// System Prompt
        system_instruction: SystemInstruction {
             parts: vec![Part{
                text: "You are a highly experienced prompt enginner. 
                I want you to polish and refine the input prompt from the user".to_string(),
             }], 
        },
        /// User Prompt
        contents: vec![Content{
            parts: vec![Part{
                text: prompt
            }],
        }],
    };
    let client = reqwest::Client::new();
    /// Makes the API request using the url
    let response = client
    .post(&url)
    .json(&request_body)
    .send()
    .await?
    .json::<GeminiResponse>()
    .await?;

    let refined =  response.candidates[0].content.parts[0].text.clone();
    /// Prints Gemini Model response
    println!("{}",refined);
    Ok(refined)
}



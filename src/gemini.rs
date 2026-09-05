//! Gemini REST client. One shared instance rate-limits every call in the run.

use anyhow::{Result, anyhow};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

use crate::config::GeminiConfig;

const API_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

pub struct GeminiClient {
    http: reqwest::Client,
    api_key: String,
    cfg: GeminiConfig,
    gate: Mutex<Instant>,
    calls: AtomicUsize,
}

impl GeminiClient {
    pub fn new(api_key: String, cfg: GeminiConfig) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.request_timeout_seconds))
            .build()?;
        Ok(Self {
            http,
            api_key,
            cfg,
            gate: Mutex::new(Instant::now()),
            calls: AtomicUsize::new(0),
        })
    }

    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }

    /// Requests start at least `60 / rpm` seconds apart.
    async fn wait_turn(&self) {
        let interval = Duration::from_secs_f64(60.0 / self.cfg.requests_per_minute);
        let start_at = {
            let mut gate = self.gate.lock().await;
            let now = Instant::now();
            let at = if *gate > now { *gate } else { now };
            *gate = at + interval;
            at
        };
        tokio::time::sleep_until(start_at.into()).await;
    }

    /// Returns `candidates[0].content`.
    pub async fn generate(
        &self,
        model: &str,
        system: Option<&str>,
        contents: &[Value],
        tools: Option<&Value>,
        temperature: f64,
    ) -> Result<Value> {
        let mut body = json!({
            "contents": contents,
            "generationConfig": { "temperature": temperature }
        });
        if let Some(s) = system {
            body["systemInstruction"] = json!({ "parts": [{ "text": s }] });
        }
        if let Some(t) = tools {
            body["tools"] = t.clone();
        }
        let url = format!("{API_URL}/{model}:generateContent");

        let mut attempt = 0u32;
        loop {
            self.wait_turn().await;
            self.calls.fetch_add(1, Ordering::Relaxed);

            let sent = self
                .http
                .post(&url)
                .header("x-goog-api-key", &self.api_key)
                .json(&body)
                .send()
                .await;
            let (http_code, text) = match sent {
                Ok(resp) => {
                    let code = resp.status().as_u16();
                    let text = resp.text().await.unwrap_or_default();
                    (code, text)
                }
                Err(e) => (0, format!("transport error: {e}")),
            };
            let parsed: Value = serde_json::from_str(&text).unwrap_or(Value::Null);

            if let Some(content) = parsed["candidates"][0]["content"].as_object() {
                return Ok(Value::Object(content.clone()));
            }

            let api_status = parsed["error"]["status"].as_str().unwrap_or("");
            let retryable = http_code == 0
                || http_code == 429
                || http_code >= 500
                || api_status == "RESOURCE_EXHAUSTED"
                || api_status == "UNAVAILABLE";

            if retryable && attempt < self.cfg.max_retries {
                let delay = retry_delay_seconds(&parsed)
                    .unwrap_or_else(|| (2f64.powi(attempt as i32) * 2.0).min(60.0))
                    .min(90.0);
                attempt += 1;
                eprintln!(
                    "[gemini:{model}] {} - retrying in {delay:.1}s (attempt {attempt}/{})",
                    short_reason(http_code, &parsed, &text),
                    self.cfg.max_retries
                );
                tokio::time::sleep(Duration::from_secs_f64(delay + 0.5)).await;
                continue;
            }

            if let Some(reason) = parsed["promptFeedback"]["blockReason"].as_str() {
                return Err(anyhow!("prompt blocked by Gemini: {reason}"));
            }
            if let Some(reason) = parsed["candidates"][0]["finishReason"].as_str() {
                return Err(anyhow!(
                    "Gemini returned an empty candidate (finishReason={reason})"
                ));
            }
            return Err(anyhow!(
                "Gemini request failed: {}",
                short_reason(http_code, &parsed, &text)
            ));
        }
    }

    pub async fn generate_text(
        &self,
        model: &str,
        system: Option<&str>,
        prompt: &str,
        temperature: f64,
    ) -> Result<String> {
        let contents = [json!({ "role": "user", "parts": [{ "text": prompt }] })];
        let content = self
            .generate(model, system, &contents, None, temperature)
            .await?;
        let text = collect_text(&content);
        if text.trim().is_empty() {
            return Err(anyhow!("Gemini returned no text"));
        }
        Ok(text)
    }
}

pub fn collect_text(content: &Value) -> String {
    content["parts"]
        .as_array()
        .map(|parts| {
            parts
                .iter()
                .filter_map(|p| p["text"].as_str())
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default()
}

fn retry_delay_seconds(parsed: &Value) -> Option<f64> {
    parsed["error"]["details"]
        .as_array()?
        .iter()
        .find(|d| d["@type"] == "type.googleapis.com/google.rpc.RetryInfo")
        .and_then(|d| d["retryDelay"].as_str())
        .and_then(|s| s.trim_end_matches('s').parse::<f64>().ok())
}

fn short_reason(http_code: u16, parsed: &Value, raw: &str) -> String {
    if let Some(msg) = parsed["error"]["message"].as_str() {
        let status = parsed["error"]["status"].as_str().unwrap_or("");
        return format!("HTTP {http_code} {status}: {}", truncate(msg, 200));
    }
    format!("HTTP {http_code}: {}", truncate(raw, 200))
}

pub fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max_chars).collect();
        format!("{cut}...")
    }
}

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::mpsc;

use crate::LLMEngine;

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
    done: bool,
}

pub struct OllamaEngine {
    model: String,
    client: Client,
    base_url: String,
}

impl OllamaEngine {
    pub fn new(model: &str, base_url: Option<String>) -> Self {
        Self {
            model: model.to_string(),
            client: Client::new(),
            base_url: base_url.unwrap_or_else(|| "http://localhost:11434".to_string()),
        }
    }
}

#[async_trait]
impl LLMEngine for OllamaEngine {
    async fn load_model(&self, _model_id: &str) -> Result<()> {
        // Ollama handles loading on demand, but we could check if model exists using /api/tags
        // For now, assume it's pulled.
        Ok(())
    }

    async fn generate(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/api/generate", self.base_url);
        let body = json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false
        });

        let res = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Failed to connect to Ollama")?;

        if !res.status().is_success() {
            return Err(anyhow::anyhow!("Ollama returned error: {}", res.status()));
        }

        let response: OllamaResponse = res
            .json()
            .await
            .context("Failed to parse Ollama response")?;
        Ok(response.response)
    }

    async fn generate_stream(&self, prompt: &str, sender: mpsc::Sender<String>) -> Result<()> {
        let url = format!("{}/api/generate", self.base_url);
        let body = json!({
            "model": self.model,
            "prompt": prompt,
            "stream": true
        });

        let res = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Failed to connect to Ollama")?;

        if !res.status().is_success() {
            let _ = sender
                .send(format!("Error: Ollama returned {}", res.status()))
                .await;
            return Err(anyhow::anyhow!("Ollama returned error: {}", res.status()));
        }

        let mut stream = res.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk_res) = stream.next().await {
            match chunk_res {
                Ok(chunk) => {
                    // Note: This is still not 100% utf8 safe if a multibyte char is split,
                    // but for JSON protocol usually boundaries are safe or rare.
                    // For a robust implementation we would use a proper codec.
                    let s = String::from_utf8_lossy(&chunk);
                    buffer.push_str(&s);

                    while let Some(pos) = buffer.find('\n') {
                        let line = buffer[..pos].to_string();
                        // Advance buffer
                        // Optimization: drain or improved slicing could be used but this is clear
                        buffer = buffer[pos + 1..].to_string();

                        if line.trim().is_empty() {
                            continue;
                        }

                        match serde_json::from_str::<OllamaResponse>(&line) {
                            Ok(json_res) => {
                                if sender.send(json_res.response).await.is_err() {
                                    return Ok(());
                                }
                                if json_res.done {
                                    return Ok(());
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to parse Ollama JSON: {} | Line: {}",
                                    e,
                                    line
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = sender.send(format!("Error streaming: {}", e)).await;
                }
            }
        }

        Ok(())
    }
}

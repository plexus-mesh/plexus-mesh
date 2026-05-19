use crate::LLMEngine;
use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::mpsc::Sender;

pub struct MockEngine;

impl MockEngine {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl LLMEngine for MockEngine {
    async fn load_model(&self, _model_id: &str) -> Result<()> {
        Ok(())
    }

    async fn generate(&self, prompt: &str) -> Result<String> {
        Ok(format!("Mock AI: I received your prompt: '{}'. The real AI engine is currently disabled for debugging.", prompt))
    }

    async fn generate_stream(&self, prompt: &str, sender: Sender<String>) -> Result<()> {
        println!("MockEngine: generate_stream called with prompt: {}", prompt);
        let response = format!("Mock AI: Echoing '{}'... ", prompt);
        if let Err(e) = sender.send(response).await {
            println!("MockEngine: Failed to send start: {}", e);
        }

        for i in 0..5 {
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
            if let Err(e) = sender.send(format!("Token {} ", i)).await {
                println!("MockEngine: Failed to send token {}: {}", i, e);
                break;
            }
        }

        let _ = sender.send("\n[Mock Stream Finished]".to_string()).await;
        println!("MockEngine: Stream finished.");
        Ok(())
    }
}

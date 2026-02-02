mod engine;
mod lance_store;
mod memory;
pub use engine::TinyLlamaEngine;
pub use lance_store::LanceDbStore;
pub use memory::{BertEmbedder, QdrantStore, SimpleVectorStore};
pub mod chat;
pub use chat::{ChatHistory, ChatMessage, Role};
pub mod voice;
use anyhow::Result;
use async_trait::async_trait;

#[async_trait]
pub trait LLMEngine: Send + Sync {
    /// Load a model from a path or identifier
    async fn load_model(&self, model_id: &str) -> Result<()>;

    /// Generate text completion
    async fn generate(&self, prompt: &str) -> Result<String>;

    /// Generate text completion with streaming
    async fn generate_stream(
        &self,
        prompt: &str,
        sender: tokio::sync::mpsc::Sender<String>,
    ) -> Result<()>;
}

#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Add a vector to the store
    async fn add(&self, id: &str, vector: Vec<f32>) -> Result<()>;
    async fn add_document(&self, id: &str, text: &str, vector: Vec<f32>) -> Result<()>;

    /// Search for nearest neighbors
    async fn search(&self, query_vector: Vec<f32>, k: usize) -> Result<Vec<(String, f32)>>;
}

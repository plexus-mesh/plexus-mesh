mod engine;
#[cfg(feature = "lancedb")]
mod lance_store;
mod memory;
mod mock;
mod ollama;
pub use engine::TinyLlamaEngine;
#[cfg(feature = "lancedb")]
pub use lance_store::LanceDbStore;
pub use memory::{BertEmbedder, QdrantStore, SimpleVectorStore};

/// Process-wide serialization of on-device GPU/Metal work.
///
/// The BERT embedder (candle/Metal, [`BertEmbedder::embed`]) and the app's
/// Gemma decoder (llama.cpp/Metal, in `rust_lib_mindora`) are two independent
/// runtimes that each allocate Metal buffers. Running them at the same time
/// exceeds the per-process Metal budget and hard-crashes the app (milder
/// overlaps merely fail the embed/inference). Every on-device GPU operation —
/// in *either* crate — must `COMPUTE_GUARD.lock().await` for the duration of its
/// compute so the two can never touch the allocator concurrently.
///
/// Deadlock-free by construction: no GPU op nests another (embed never infers,
/// inference never embeds), so a holder never re-acquires the guard.
pub static COMPUTE_GUARD: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

pub use mock::MockEngine;
pub use ollama::OllamaEngine;
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

    /// Search for nearest neighbors, restricted by optional metadata filters.
    ///
    /// Stores that understand `source_type` / `date_key` metadata (e.g. the
    /// SQLCipher-backed store) should override this to filter *inside* the
    /// index, where the query embedding and any over-fetch/fallback logic
    /// live — that's far more accurate than letting the caller post-filter a
    /// fixed top-k after the fact. Stores without metadata fall back to the
    /// unfiltered [`VectorStore::search`] via this default.
    async fn search_filtered(
        &self,
        query_vector: Vec<f32>,
        k: usize,
        _source_type: Option<String>,
        _date_from: Option<String>,
        _date_to: Option<String>,
    ) -> Result<Vec<(String, f32)>> {
        self.search(query_vector, k).await
    }
}

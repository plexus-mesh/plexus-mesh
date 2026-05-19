use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::tokio::Api, Repo, RepoType};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokenizers::{PaddingParams, Tokenizer};
use tokio::sync::Mutex as AsyncMutex;

use crate::VectorStore;
use std::convert::TryInto; // For payload conversion
use std::path::PathBuf;

const BERT_REPO: &str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2";

pub struct BertEmbedder {
    model: Arc<Mutex<Option<BertModel>>>,
    tokenizer: Arc<Mutex<Option<Tokenizer>>>,
    loading: Arc<AsyncMutex<bool>>,
    data_dir: Option<std::path::PathBuf>,
}

impl BertEmbedder {
    pub fn new(data_dir: Option<std::path::PathBuf>) -> Self {
        Self {
            model: Arc::new(Mutex::new(None)),
            tokenizer: Arc::new(Mutex::new(None)),
            loading: Arc::new(AsyncMutex::new(false)),
            data_dir,
        }
    }

    async fn ensure_loaded(&self) -> Result<()> {
        {
            if self.model.lock().unwrap().is_some() {
                return Ok(());
            }
        }

        let mut loading_guard = self.loading.lock().await;
        if *loading_guard {
            return Ok(());
        }
        *loading_guard = true;

        // --- Strategy: Try LOCAL files first (pre-downloaded by Dart), then fallback to HF ---
        let (config_path, tokenizer_path, weights_path) = if let Some(ref dir) = self.data_dir {
            let embedder_dir = dir.join("embedder");
            let local_config = embedder_dir.join("config.json");
            let local_tokenizer = embedder_dir.join("tokenizer.json");
            let local_weights = embedder_dir.join("model.safetensors");

            if local_config.exists() && local_tokenizer.exists() && local_weights.exists() {
                println!(
                    "BertEmbedder: Loading from pre-downloaded local files at {:?}",
                    embedder_dir
                );
                (local_config, local_tokenizer, local_weights)
            } else {
                println!("BertEmbedder: Local files not found. Downloading from HuggingFace...");
                match self.download_from_hf().await {
                    Ok(paths) => paths,
                    Err(e) => {
                        *loading_guard = false;
                        return Err(e);
                    }
                }
            }
        } else {
            println!("BertEmbedder: No data_dir. Downloading from HuggingFace...");
            match self.download_from_hf().await {
                Ok(paths) => paths,
                Err(e) => {
                    *loading_guard = false;
                    return Err(e);
                }
            }
        };

        // --- Load model from resolved paths ---
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
            *loading_guard = false;
            E::msg(format!("Failed to read config: {}", e))
        })?;

        let config: Config = serde_json::from_str(&config_str).map_err(|e| {
            *loading_guard = false;
            E::msg(format!("Failed to parse config: {}", e))
        })?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            *loading_guard = false;
            E::msg(format!("Failed to load tokenizer: {}", e))
        })?;

        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &Device::Cpu) }
                .map_err(|e| {
                    *loading_guard = false;
                    E::msg(format!("Failed to load weights: {}", e))
                })?;

        let model = BertModel::load(vb, &config).map_err(|e| {
            *loading_guard = false;
            E::msg(format!("Failed to init BertModel: {}", e))
        })?;

        {
            let mut m_guard = self.model.lock().unwrap();
            *m_guard = Some(model);
            let mut t_guard = self.tokenizer.lock().unwrap();
            *t_guard = Some(tokenizer);
        }

        println!("Embedding Model Loaded.");
        *loading_guard = false;
        Ok(())
    }

    /// Fallback: Download model files from HuggingFace (works on non-sandboxed environments)
    async fn download_from_hf(&self) -> Result<(PathBuf, PathBuf, PathBuf)> {
        println!("Downloading Embedding Model ({})...", BERT_REPO);
        let mut builder = hf_hub::api::tokio::ApiBuilder::new();
        if let Some(ref dir) = self.data_dir {
            builder = builder.with_cache_dir(dir.clone());
        }

        let api = builder
            .build()
            .map_err(|e| E::msg(format!("Failed to init HF API: {}", e)))?;
        let repo = api.repo(Repo::new(BERT_REPO.to_string(), RepoType::Model));

        let config_path = repo
            .get("config.json")
            .await
            .map_err(|e| E::msg(format!("Failed to download config.json: {}", e)))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .await
            .map_err(|e| E::msg(format!("Failed to download tokenizer.json: {}", e)))?;
        let weights_path = repo
            .get("model.safetensors")
            .await
            .map_err(|e| E::msg(format!("Failed to download model.safetensors: {}", e)))?;

        Ok((config_path, tokenizer_path, weights_path))
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.ensure_loaded().await?;

        // We hold the lock during inference for simplicity since BertModel isn't Clone
        let mut guard = self.model.lock().unwrap();
        let model = guard.as_mut().unwrap();

        let tokenizer_guard = self.tokenizer.lock().unwrap();
        let tokenizer = tokenizer_guard.as_ref().unwrap();

        let tokens = tokenizer.encode(text, true).map_err(E::msg)?;
        let token_ids = tokens.get_ids();
        let token_ids = Tensor::new(token_ids, &Device::Cpu)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;

        // Calculate embeddings
        // forward(input_ids, token_type_ids, position_ids)
        // Check signature: error said arg #3 is Option<&Tensor>
        let embeddings = model.forward(&token_ids, &token_type_ids, None)?;

        // Mean pooling
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = normalize_l2(&embeddings)?;

        let vec = embeddings.get(0)?.to_vec1::<f32>()?;
        Ok(vec)
    }
}

fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    let norm = v.sqr()?.sum_keepdim(1)?.sqrt()?;
    Ok(v.broadcast_div(&norm)?)
}

pub struct SimpleVectorStore {
    data: Arc<Mutex<HashMap<String, (Vec<f32>, String)>>>,
    path: Option<std::path::PathBuf>,
}

impl SimpleVectorStore {
    pub fn new(path: Option<std::path::PathBuf>) -> Self {
        let mut map = HashMap::new();
        if let Some(p) = &path {
            if p.exists() {
                if let Ok(file) = std::fs::File::open(p) {
                    if let Ok(loaded) = serde_json::from_reader(file) {
                        map = loaded;
                    }
                }
            }
        }
        Self {
            data: Arc::new(Mutex::new(map)),
            path,
        }
    }

    fn save_to_disk(&self) {
        if let Some(p) = &self.path {
            if let Ok(data) = self.data.lock() {
                if let Ok(file) = std::fs::File::create(p) {
                    let _ = serde_json::to_writer(file, &*data);
                }
            }
        }
    }
}

#[async_trait::async_trait]
impl VectorStore for SimpleVectorStore {
    async fn add(&self, _id: &str, _vector: Vec<f32>) -> Result<()> {
        Err(anyhow::anyhow!("Use add_document instead"))
    }

    async fn add_document(&self, id: &str, text: &str, vector: Vec<f32>) -> Result<()> {
        {
            let mut data = self.data.lock().unwrap();
            data.insert(id.to_string(), (vector, text.to_string()));
        }
        self.save_to_disk();
        Ok(())
    }

    async fn search(&self, query_vector: Vec<f32>, k: usize) -> Result<Vec<(String, f32)>> {
        let data = self.data.lock().unwrap();
        let mut results = vec![];

        for (_id, (vec, text)) in data.iter() {
            let similarity = cosine_similarity(&query_vector, vec);
            results.push((text.clone(), similarity));
        }

        // Sort by similarity descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        Ok(results)
    }
}

fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm1 == 0.0 || norm2 == 0.0 {
        return 0.0;
    }

    dot_product / (norm1 * norm2)
}

use qdrant_client::prelude::*;
use qdrant_client::qdrant::vectors_config::Config as VectorsConfig;
use qdrant_client::qdrant::with_payload_selector::SelectorOptions;
use qdrant_client::qdrant::{CreateCollection, PointStruct, SearchPoints, VectorParams};

pub struct QdrantStore {
    client: Arc<QdrantClient>,
    collection_name: String,
}

impl QdrantStore {
    pub async fn new(url: &str) -> Result<Self> {
        let client = QdrantClient::from_url(url).build()?;
        let store = Self {
            client: Arc::new(client),
            collection_name: "plexus_memory".to_string(),
        };
        store.init().await?;
        Ok(store)
    }

    async fn init(&self) -> Result<()> {
        let collections = self.client.list_collections().await?;
        if !collections
            .collections
            .iter()
            .any(|c| c.name == self.collection_name)
        {
            println!("Creating Qdrant collection: {}", self.collection_name);
            self.client
                .create_collection(&CreateCollection {
                    collection_name: self.collection_name.clone(),
                    vectors_config: Some(
                        VectorParams {
                            size: 384, // MiniLM-L6-v2 dimension
                            distance: Distance::Cosine.into(),
                            ..Default::default()
                        }
                        .into(),
                    ),
                    ..Default::default()
                })
                .await?;
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl VectorStore for QdrantStore {
    async fn add(&self, _id: &str, _vector: Vec<f32>) -> Result<()> {
        Err(anyhow::anyhow!("Use add_document instead"))
    }

    async fn add_document(&self, id: &str, text: &str, vector: Vec<f32>) -> Result<()> {
        let payload: std::collections::HashMap<String, qdrant_client::qdrant::Value> =
            serde_json::from_value(serde_json::json!({
                "text": text
            }))?;

        let point = PointStruct::new(id.to_string(), vector, payload);

        self.client
            .upsert_points(self.collection_name.clone(), None, vec![point], None)
            .await?;
        Ok(())
    }

    async fn search(&self, query_vector: Vec<f32>, k: usize) -> Result<Vec<(String, f32)>> {
        let search_result = self
            .client
            .search_points(&SearchPoints {
                collection_name: self.collection_name.clone(),
                vector: query_vector,
                limit: k as u64,
                with_payload: Some(SelectorOptions::Enable(true).into()),
                ..Default::default()
            })
            .await?;

        let mut results = vec![];
        for point in search_result.result {
            // point.payload is a HashMap<String, Value> in newer versions
            let payload = point.payload;
            if let Some(json_val) = payload.get("text") {
                let text = format!("{}", json_val);
                results.push((text, point.score));
            }
        }
        Ok(results)
    }
}

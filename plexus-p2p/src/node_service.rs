use crate::{
    build_swarm_safe,
    protocol::{Heartbeat, NodeCapabilities},
    swarm::PlexusBehaviourEvent,
    GenerateRequest, GenerateResponse, IdentityStore, PlexusBehaviour,
};
use anyhow::{Context, Result};
use futures::StreamExt;
use libp2p::{
    dcutr,
    gossipsub::{self, IdentTopic},
    multiaddr::Protocol,
    relay,
    request_response::{self, OutboundRequestId},
    swarm::SwarmEvent,
    Multiaddr, PeerId, Swarm,
};
#[cfg(feature = "lancedb")]
use plexus_ai::LanceDbStore;
use plexus_ai::{
    voice::WhisperEngine, BertEmbedder, ChatHistory, LLMEngine, QdrantStore, SimpleVectorStore,
    TinyLlamaEngine, VectorStore,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use sysinfo::{Networks, System};
use tokio::sync::mpsc;
use tokio::time::{interval, Duration, Instant};
use tracing::{error, info, warn};

#[derive(Debug, Clone, serde::Serialize)]
pub struct SystemCapabilities {
    pub cpu_model: String,
    pub total_memory_gb: u64,
    pub used_memory_gb: u64,
    pub cpu_cores: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct NodeStatus {
    pub peer_id: String,
    pub connected_peers: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
struct PairingResponse {
    code: String,
    addresses: Vec<String>,
}

#[derive(Debug)]
pub enum NodeCommand {
    Shutdown,
    Generate {
        prompt: String,
        respond_to: mpsc::Sender<String>,
    },
    GetStatus {
        respond_to: mpsc::Sender<NodeStatus>,
    },
    GetMeshState {
        respond_to: mpsc::Sender<Vec<Heartbeat>>,
    },
    SetSystemPrompt {
        prompt: String,
        respond_to: mpsc::Sender<()>,
    },
    Transcribe {
        audio_data: Vec<f32>,
        respond_to: mpsc::Sender<String>,
    },
    GetSystemInfo {
        respond_to: mpsc::Sender<SystemCapabilities>,
    },
    StartPairing {
        respond_to: mpsc::Sender<String>,
    },
    Search {
        query: String,
        limit: usize,
        respond_to: mpsc::Sender<Result<Vec<(String, f32)>, String>>,
    },
    /// Connect to a paired device by its multiaddr (supports relay addresses).
    ConnectPeer {
        address: String,
        respond_to: mpsc::Sender<Result<(), String>>,
    },
    /// Register a relay node for NAT traversal (used when on mobile data).
    SetRelay {
        relay_address: String,
        respond_to: mpsc::Sender<Result<(), String>>,
    },
    /// Offload an AI query to the most capable connected peer.
    /// Used when the local model is too small or the device is constrained.
    OffloadInference {
        query: String,
        respond_to: mpsc::Sender<Result<String, String>>,
    },
}

use tokio::sync::Mutex;

// ...

pub struct NodeService {
    swarm: Swarm<PlexusBehaviour>,
    command_rx: mpsc::Receiver<NodeCommand>,
    ai_engine: Box<dyn LLMEngine>,
    whisper_engine: Arc<Mutex<WhisperEngine>>,
    pending_requests: HashMap<OutboundRequestId, mpsc::Sender<String>>,
    /// Pending offload inference requests (mapped from request_id to response channel)
    pending_offloads: HashMap<OutboundRequestId, mpsc::Sender<Result<String, String>>>,
    chat_history: ChatHistory,
    history_path: PathBuf,
    embedder: BertEmbedder,
    vector_store: Arc<dyn VectorStore>,
    system: System,
    mesh_state: crate::crdt::MeshState,
    heartbeat_topic: IdentTopic,
    active_model: String,
    /// Known paired peers for auto-reconnection
    paired_peers: Vec<(PeerId, Multiaddr)>,
    /// Active relay address (for NAT traversal over mobile data)
    relay_addr: Option<Multiaddr>,
    /// Last successful connection time per peer (for exponential backoff)
    last_connect_attempt: HashMap<PeerId, Instant>,
    /// Reconnection backoff counter per peer
    reconnect_backoff: HashMap<PeerId, u32>,
}

impl NodeService {
    pub async fn new(
        identity_path: PathBuf,
        command_rx: mpsc::Receiver<NodeCommand>,
        model_id: String,
        ai_endpoint: Option<String>,
        bootstrap_peers: Vec<libp2p::Multiaddr>,
        data_dir: Option<PathBuf>,
        injected_vector_store: Option<Arc<dyn VectorStore>>, // Added for SQLCipher
    ) -> Result<Self> {
        info!("NodeService: Initializing...");
        info!("NodeService: Selected Model: {}", model_id);

        let identity_store = IdentityStore::new(identity_path.clone());
        info!("NodeService: Loading/Generating identity...");
        let keypair = identity_store
            .load_or_generate()
            .context("Failed to load identity")?;
        info!("NodeService: Identity loaded.");

        info!("NodeService: Building Swarm (SAFE MODE)...");
        let mut swarm = build_swarm_safe(keypair)
            .await
            .context("Failed to build swarm")?;
        info!("NodeService: Swarm built.");

        // Bootstrap Kademlia
        if !bootstrap_peers.is_empty() {
            info!(
                "NodeService: Bootstrapping Kademlia with {} peers...",
                bootstrap_peers.len()
            );
            for peer in bootstrap_peers {
                // For now, we don't have the peer_id in the Multiaddr usually unless it's /p2p/...,
                // but Kademlia needs a PeerId.
                // If the Multiaddr contains /p2p/<ID>, we can extract it.
                // Simplification: We assume the multiaddr ends with /p2p/Qm...
                if let Some(Protocol::P2p(peer_id)) =
                    peer.iter().find(|p| matches!(p, Protocol::P2p(_)))
                {
                    info!("Adding bootstrap peer: {}", peer);
                    swarm.behaviour_mut().kademlia.add_address(&peer_id, peer);
                } else {
                    tracing::warn!("Bootstrap peer address must include /p2p/<ID>: {}", peer);
                }
            }
            if let Err(e) = swarm.behaviour_mut().kademlia.bootstrap() {
                tracing::warn!("Kademlia bootstrap failed: {}", e);
            }
        }

        // Determine Data Directory early
        let app_data_dir = if let Some(path) = data_dir.clone() {
            path
        } else {
            let project_dirs = directories_next::ProjectDirs::from("com", "plexus", "mesh")
                .context("Could not determine data directory")?;
            project_dirs.data_dir().to_path_buf()
        };
        std::fs::create_dir_all(&app_data_dir).context("Failed to create data directory")?;

        info!("NodeService: Initializing AI Engine...");
        let ai_engine: Box<dyn plexus_ai::LLMEngine> = match model_id.as_str() {
            "tinyllama" => Box::new(TinyLlamaEngine::new(Some(app_data_dir.clone()))),
            "ollama" => Box::new(plexus_ai::OllamaEngine::new("llama3", None)),
            "mistral" => Box::new(plexus_ai::OllamaEngine::new("mistral", None)),
            "mock" => Box::new(plexus_ai::MockEngine::new()),
            "phi" => {
                tracing::warn!("Phi engine requested, using Ollama/Phi instead.");
                Box::new(plexus_ai::OllamaEngine::new("phi", None))
            }
            _ => {
                tracing::warn!("Unknown model '{}'. Defaulting to TinyLlama.", model_id);
                Box::new(TinyLlamaEngine::new(Some(app_data_dir.clone())))
            }
        };

        info!("NodeService: Initializing Whisper Engine...");
        let whisper_engine = Arc::new(Mutex::new(WhisperEngine::new())); // Wrapped

        info!("NodeService: Initializing Embedder...");
        let embedder = BertEmbedder::new(Some(app_data_dir.clone()));

        // Database
        info!("NodeService: Initializing Local Vector Database...");
        let vector_store: Arc<dyn VectorStore> = if let Some(store) = injected_vector_store {
            info!("Using injected Vector Store (SQLCipher)");
            store
        } else {
            let qdrant_future = async { QdrantStore::new("http://localhost:6334").await };

            // 2-second timeout for Qdrant connection
            match tokio::time::timeout(Duration::from_secs(2), qdrant_future).await {
                Ok(Ok(store)) => {
                    info!("Connected to Qdrant Vector Database.");
                    Arc::new(store)
                }
                Ok(Err(e)) => {
                    error!(
                        "Failed to connect to Qdrant ({}). Falling back to Embedded Store.",
                        e
                    );
                    #[cfg(feature = "lancedb")]
                    {
                        let lance_path = app_data_dir.join("vectors.lance");
                        match LanceDbStore::new(&lance_path).await {
                            Ok(store) => {
                                info!("Connected to Embedded LanceDB at {:?}", lance_path);
                                Arc::new(store)
                            }
                            Err(e) => {
                                error!("Failed to init LanceDB: {}. Falling back to In-Memory.", e);
                                Arc::new(SimpleVectorStore::new(Some(
                                    app_data_dir.join("vectors.json"),
                                )))
                            }
                        }
                    }
                    #[cfg(not(feature = "lancedb"))]
                    {
                        info!("LanceDB disabled. Falling back to In-Memory SimpleVectorStore.");
                        Arc::new(SimpleVectorStore::new(Some(
                            app_data_dir.join("vectors.json"),
                        )))
                    }
                }
                Err(_) => {
                    error!("Qdrant connection timed out. Falling back to Embedded Store.");
                    #[cfg(feature = "lancedb")]
                    {
                        let lance_path = app_data_dir.join("vectors.lance");
                        match LanceDbStore::new(&lance_path).await {
                            Ok(store) => {
                                info!("Connected to Embedded LanceDB at {:?}", lance_path);
                                Arc::new(store)
                            }
                            Err(e) => {
                                error!("Failed to init LanceDB: {}. Falling back to In-Memory.", e);
                                Arc::new(SimpleVectorStore::new(Some(
                                    app_data_dir.join("vectors.json"),
                                )))
                            }
                        }
                    }
                    #[cfg(not(feature = "lancedb"))]
                    {
                        info!("LanceDB disabled. Falling back to In-Memory SimpleVectorStore.");
                        Arc::new(SimpleVectorStore::new(Some(
                            app_data_dir.join("vectors.json"),
                        )))
                    }
                }
            }
        };
        info!("NodeService: Vector Store initialized.");

        // Load Whisper Model (Async & Non-blocking)
        let we_clone = whisper_engine.clone();
        tokio::spawn(async move {
            info!("Starting background load of Whisper model...");
            if let Err(e) = we_clone.lock().await.load_model().await {
                error!("Failed to load Whisper model in background: {}", e);
            } else {
                info!("Whisper model loaded in background.");
            }
        });

        // Load Chat History
        info!("NodeService: Loading Chat History...");
        let history_path = identity_path
            .parent()
            .unwrap_or(&PathBuf::from("."))
            .join("chat_history.json");
        let chat_history =
            ChatHistory::load_from_file(&history_path).unwrap_or_else(|_| ChatHistory::new(10));

        // Capabilities & Gossipsub
        info!("NodeService: Refreshing System Stats...");
        let mut system = System::new();
        system.refresh_cpu_all();
        system.refresh_memory();

        info!("NodeService: Subscribing to gossipsub...");
        let heartbeat_topic = IdentTopic::new("plexus-mesh/capabilities/1.0.0");
        swarm
            .behaviour_mut()
            .gossipsub
            .subscribe(&heartbeat_topic)?;

        info!("NodeService: Initializing Persistence...");

        let data_dir = if let Some(path) = data_dir {
            path
        } else {
            let project_dirs = directories_next::ProjectDirs::from("com", "plexus", "mesh")
                .context("Could not determine data directory")?;
            project_dirs.data_dir().to_path_buf()
        };

        std::fs::create_dir_all(&data_dir).context("Failed to create data directory")?;
        let db_path = data_dir.join("mesh_state.db");
        info!("NodeService: Mesh DB Path: {:?}", db_path);

        let mesh_state =
            crate::crdt::MeshState::new(db_path).context("Failed to initialize MeshState DB")?;

        info!("NodeService: Initialization Complete.");
        Ok(Self {
            swarm,
            command_rx,
            ai_engine,
            whisper_engine,
            pending_requests: HashMap::new(),
            pending_offloads: HashMap::new(),
            chat_history,
            history_path,
            embedder,
            vector_store,
            system,
            mesh_state,
            heartbeat_topic,
            active_model: model_id,
            paired_peers: Vec::new(),
            relay_addr: None,
            last_connect_attempt: HashMap::new(),
            reconnect_backoff: HashMap::new(),
        })
    }

    fn save_history(&self) {
        let _ = self.chat_history.save_to_file(&self.history_path);
    }

    /// Select the most capable connected peer for offloading.
    /// Prefers peers with more RAM and CPU cores.
    fn select_best_peer_for_offload(&self) -> Option<PeerId> {
        let connected: Vec<PeerId> = self.swarm.connected_peers().cloned().collect();
        if connected.is_empty() {
            return None;
        }

        // Find the peer with the best capabilities from mesh state
        let all_heartbeats = self.mesh_state.get_all();
        let mut best_peer: Option<(PeerId, u64)> = None;

        for hb in &all_heartbeats {
            if let Ok(pid) = PeerId::from_str(&hb.peer_id) {
                if connected.contains(&pid) {
                    let score = hb.capabilities.total_memory / (1024 * 1024); // MB
                    if best_peer.is_none() || score > best_peer.unwrap().1 {
                        best_peer = Some((pid, score));
                    }
                }
            }
        }

        best_peer
            .map(|(pid, _)| pid)
            .or_else(|| connected.into_iter().next())
    }

    /// Attempt to reconnect to paired peers with exponential backoff.
    async fn try_reconnect_paired_peers(&mut self) {
        let now = Instant::now();
        let peers_to_try: Vec<(PeerId, Multiaddr)> = self.paired_peers.clone();

        for (peer_id, addr) in peers_to_try {
            // Skip if already connected
            if self.swarm.is_connected(&peer_id) {
                self.reconnect_backoff.remove(&peer_id);
                continue;
            }

            // Exponential backoff: 5s, 10s, 20s, 40s, 80s, max 300s
            let backoff = self.reconnect_backoff.entry(peer_id).or_insert(0);
            let delay_secs = (5u64 * 2u64.pow((*backoff).min(6))).min(300);

            if let Some(last) = self.last_connect_attempt.get(&peer_id) {
                if now.duration_since(*last) < Duration::from_secs(delay_secs) {
                    continue; // Too soon, skip this peer
                }
            }

            info!(
                "Attempting reconnection to paired peer {} (backoff={}s)",
                peer_id, delay_secs
            );
            self.last_connect_attempt.insert(peer_id, now);

            // Try direct connection first, then relay if available
            let dial_addr = if let Some(ref relay) = self.relay_addr {
                // Construct relay circuit address: /relay_addr/p2p-circuit/p2p/peer_id
                let mut circuit_addr = relay.clone();
                circuit_addr.push(Protocol::P2pCircuit);
                circuit_addr.push(Protocol::P2p(peer_id));
                circuit_addr
            } else {
                addr.clone()
            };

            match self.swarm.dial(dial_addr) {
                Ok(_) => info!("Dial initiated to {}", peer_id),
                Err(e) => {
                    warn!("Failed to dial {}: {}", peer_id, e);
                    *self.reconnect_backoff.entry(peer_id).or_insert(0) += 1;
                }
            }
        }
    }

    pub async fn run(mut self) -> Result<()> {
        // Listen on all interfaces — TCP and QUIC
        if let Err(e) = self.swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?) {
            warn!("Failed to bind TCP listener: {}", e);
        }
        if let Err(e) = self.swarm.listen_on("/ip4/0.0.0.0/udp/0/quic-v1".parse()?) {
            warn!("Failed to bind QUIC listener (non-fatal): {}", e);
        }

        let mut heartbeat_interval = interval(Duration::from_secs(15));
        let mut reconnect_interval = interval(Duration::from_secs(30));

        loop {
            tokio::select! {
                _ = reconnect_interval.tick() => {
                    self.try_reconnect_paired_peers().await;
                }
                _ = heartbeat_interval.tick() => {
                    // Refresh stats
                    self.system.refresh_cpu_all();
                    self.system.refresh_memory();

                    let capabilities = NodeCapabilities {
                        cpu_cores: self.system.cpus().len(),
                        total_memory: self.system.total_memory(),
                        gpu_info: None,
                        model_loaded: true,
                    };

                    let heartbeat = Heartbeat {
                        peer_id: self.swarm.local_peer_id().to_string(),
                        model: self.active_model.clone(),
                        capabilities: capabilities.clone(),
                        timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                    };

                    // Update local state
                    self.update_mesh_state(heartbeat.clone());

                    if let Ok(data) = serde_json::to_vec(&heartbeat) {
                        if let Err(e) = self.swarm.behaviour_mut().gossipsub.publish(self.heartbeat_topic.clone(), data) {
                             error!("Failed to publish heartbeat: {}", e);
                        }
                    }
                }
                event = self.swarm.select_next_some() => {
                    match event {
                        SwarmEvent::NewListenAddr { address, .. } => {
                            info!("Listening on {:?}", address);
                        }
                        SwarmEvent::Behaviour(PlexusBehaviourEvent::Gossipsub(gossipsub::Event::Message { propagation_source: _peer_id, message_id: _id, message })) => {
                            if let Ok(heartbeat) = serde_json::from_slice::<Heartbeat>(&message.data) {
                                info!("Received Heartbeat from {}: {} Cores, {} MB RAM",
                                    heartbeat.peer_id,
                                    heartbeat.capabilities.cpu_cores,
                                    heartbeat.capabilities.total_memory / 1024 / 1024
                                );
                                self.update_mesh_state(heartbeat);
                            }
                        }
                        // mDNS removed for "Safe Mode" debugging
                        /*
                        SwarmEvent::Behaviour(PlexusBehaviourEvent::Mdns(mdns::Event::Discovered(peers))) => {
                            for (peer, addr) in peers {
                                info!("MDNS Discovered: {} at {}", peer, addr);
                                self.swarm.behaviour_mut().kademlia.add_address(&peer, addr.clone());
                                if let Err(e) = self.swarm.dial(addr) {
                                     info!("Failed to dial {}: {}", peer, e);
                                }
                            }
                        }
                        */
                        SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                            info!("Connection established with {}", peer_id);
                            // Reset backoff on successful connection
                            self.reconnect_backoff.remove(&peer_id);
                        }
                        SwarmEvent::ConnectionClosed { peer_id, .. } => {
                            info!("Connection closed with {}", peer_id);
                        }
                        // Relay events
                        SwarmEvent::Behaviour(PlexusBehaviourEvent::RelayClient(
                            relay::client::Event::ReservationReqAccepted { relay_peer_id, .. }
                        )) => {
                            info!("Relay reservation accepted by {}", relay_peer_id);
                        }
                        // DCUtR: direct connection upgrade successful
                        SwarmEvent::Behaviour(PlexusBehaviourEvent::Dcutr(
                            dcutr::Event { remote_peer_id, result }
                        )) => {
                            match result {
                                Ok(_) => info!("DCUtR: Direct connection established with {}", remote_peer_id),
                                Err(ref e) => warn!("DCUtR: Hole-punch to {} failed: {:?} (relay still active)", remote_peer_id, e),
                            }
                        }
                        SwarmEvent::Behaviour(PlexusBehaviourEvent::RequestResponse(
                            request_response::Event::Message { peer, message }
                        )) => {
                            match message {
                                request_response::Message::Request { request, channel, .. } => {
                                    info!("Received remote generation request from {}: {}", peer, request.prompt);

                                    // 0. RAG Retrieval
                                    let mut context = String::new();
                                    if let Ok(query_vec) = self.embedder.embed(&request.prompt).await {
                                        if let Ok(results) = self.vector_store.search(query_vec, 3).await {
                                            for (text, score) in results {
                                                if score > 0.4 {
                                                    info!("RAG Match found (score {:.2}): {}", score, text);
                                                    context.push_str(&format!("Context information: {}\n", text));
                                                }
                                            }
                                        }
                                    }

                                    // 1. Format with Context & Time
                                    let current_time = chrono::Utc::now().to_rfc3339();
                                    let time_context = format!("Current System Time: {}\n", current_time);

                                    let final_prompt = format!("{}{}{}", time_context, context, request.prompt);

                                    let response = match self.ai_engine.generate(&final_prompt).await {
                                        Ok(res) => res,
                                        Err(e) => format!("Error: {}", e),
                                    };
                                    let _ = self.swarm.behaviour_mut().request_response.send_response(channel, GenerateResponse { response });
                                }
                                request_response::Message::Response { request_id, response } => {
                                    info!("Received remote response (len={})", response.response.len());
                                    if let Some(tx) = self.pending_requests.remove(&request_id) {
                                        let _ = tx.send(response.response).await;
                                    } else if let Some(tx) = self.pending_offloads.remove(&request_id) {
                                        let _ = tx.send(Ok(response.response)).await;
                                    }
                                }
                            }
                        }
                        SwarmEvent::Behaviour(_) => {}
                        _ => {}
                    }
                }
                cmd = self.command_rx.recv() => {
                    match cmd {
                        Some(NodeCommand::Shutdown) => {
                            info!("Shutting down Node Service...");
                            break;
                        }
                        Some(NodeCommand::Generate { prompt, respond_to }) => {
                            if prompt.starts_with("/remote ") {
                                let remote_prompt = prompt.trim_start_matches("/remote ").to_string();
                                info!("Dispatching remote request: {}", remote_prompt);

                                // Select a peer
                                let peers: Vec<_> = self.swarm.connected_peers().cloned().collect();
                                if let Some(peer) = peers.first() {
                                    let request_id = self.swarm.behaviour_mut().request_response.send_request(
                                        peer,
                                        GenerateRequest { prompt: remote_prompt }
                                    );
                                    info!("Sent request {} to peer {}", request_id, peer);
                                    // Store the channel to respond later
                                    self.pending_requests.insert(request_id, respond_to);
                                } else {
                                    let _ = respond_to.send("No peers connected for remote inference.".to_string()).await;
                                }
                            } else if prompt.starts_with("/clear_history") {
                                info!("Purging chat history from disk...");
                                self.chat_history.clear();
                                self.save_history();
                                let _ = respond_to.send("[Started processing new context...]".to_string()).await;
                            } else if prompt.starts_with("/purge_memory") {
                                info!("Purging ALL vector memories from disk...");

                                // Best effort purge for SimpleVectorStore
                                if let Some(proj_dirs) = directories_next::ProjectDirs::from("com", "plexus", "mesh") {
                                    let vectors_path = proj_dirs.data_dir().join("vectors.json");
                                    let _ = std::fs::remove_file(&vectors_path);
                                }

                                self.chat_history.clear();
                                self.save_history();
                                let _ = respond_to.send("[Memory successfully wiped!]".to_string()).await;
                            } else if prompt.starts_with("/save ") {
                                let content = prompt.trim_start_matches("/save ").to_string();
                                info!("Saving to memory: {}", content);

                                match self.embedder.embed(&content).await {
                                    Ok(embedding) => {
                                        let id = format!("{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos());
                                        if let Err(e) = self.vector_store.add_document(&id, &content, embedding).await {
                                            let _ = respond_to.send(format!("Error saving: {}", e)).await;
                                        } else {
                                            let _ = respond_to.send(format!("Saved to memory: \"{}\"", content)).await;
                                        }
                                    }
                                    Err(e) => {
                                        let _ = respond_to.send(format!("Error embedding: {}", e)).await;
                                    }
                                }
                            } else {
                                info!("Processing local generation request: {}", prompt);

                                // Check if Dart already injected context (marked with [Your Data...])
                                // When present, skip Rust-side RAG to avoid double context injection
                                // which wastes tokens and degrades answer quality on small models.
                                let has_dart_context = prompt.contains("[Your Data");

                                // Extract the raw user question for chat history.
                                // If Dart injected context, the format is:
                                //   [Your Data — ...]\n...\n[Question]\n<actual question>
                                // We store only the actual question in history.
                                let user_question = if has_dart_context {
                                    if let Some(q_idx) = prompt.find("[Question]\n") {
                                        prompt[q_idx + "[Question]\n".len()..].to_string()
                                    } else {
                                        prompt.clone()
                                    }
                                } else {
                                    prompt.clone()
                                };

                                // 0. RAG Retrieval & Context Construction
                                let mut augmented_prompt = String::new();

                                // Add Time Context
                                let current_time = chrono::Utc::now().to_rfc3339();
                                augmented_prompt.push_str(&format!("System Time: {}\n", current_time));

                                if has_dart_context {
                                    // Dart already provided structured context — use it directly.
                                    // The entire prompt (with [Your Data] prefix) becomes the context.
                                    augmented_prompt.push_str(&prompt);
                                } else {
                                    // No Dart context — run Rust-side RAG for thoughts/journal entries.
                                    if let Ok(query_vec) = self.embedder.embed(&prompt).await {
                                        if let Ok(results) = self.vector_store.search(query_vec, 3).await {
                                            for (text, score) in results {
                                                if score > 0.4 {
                                                    info!("RAG Match found (score {:.2}): {}", score, text);
                                                    augmented_prompt.push_str(&format!("Relevant Memory: {}\n", text));
                                                }
                                            }
                                        }
                                    }
                                }

                                let system_context_opt = if !augmented_prompt.is_empty() {
                                    Some(augmented_prompt)
                                } else {
                                    None
                                };

                                // 1. Add Pure User Message to History (without context prefix)
                                self.chat_history.add_user(user_question);
                                self.save_history();

                                // 2. Format for LLM (Standard Template)
                                let context_prompt = self.chat_history.format_for_llama(system_context_opt);

                                // 3. Generate with Streaming
                                // We clone respond_to to keep using it in the stream if needed (async rules)
                                let stream_tx = respond_to.clone();

                                // We need to accumulate the full response for ChatHistory
                                // Let's use a proxy channel to capture text for History.
                                let (proxy_tx, mut proxy_rx): (mpsc::Sender<String>, mpsc::Receiver<String>) = mpsc::channel(32);

                                // Spawn a task to forward tokens to Tauri AND accumulate them
                                let forward_task = tokio::spawn(async move {
                                    let mut accumulator = String::new();
                                    while let Some(token) = proxy_rx.recv().await {
                                        accumulator.push_str(&token);
                                        if respond_to.send(token).await.is_err() {
                                            tracing::info!("Client disconnected. Aborting generation forward.");
                                            break;
                                        }
                                    }
                                    accumulator
                                });

                                match self.ai_engine.generate_stream(&context_prompt, proxy_tx).await {
                                    Ok(_) => {
                                        // Wait for forwarding to finish (sender dropped)
                                        if let Ok(final_text) = forward_task.await {
                                            // 4. Add Assistant Message to History
                                            self.chat_history.add_assistant(final_text);
                                            self.save_history(); // Save after full response
                                        }
                                    }
                                    Err(e) => {
                                        let _ = stream_tx.send(format!("Error: {:#}", e)).await;
                                    }
                                }
                            }
                        }
                        Some(NodeCommand::GetStatus { respond_to }) => {
                            let status = NodeStatus {
                                peer_id: self.swarm.local_peer_id().to_string(),
                                connected_peers: self.swarm.network_info().num_peers(),
                            };
                            let _ = respond_to.send(status).await;
                        }
                        Some(NodeCommand::GetMeshState { respond_to }) => {
                            // Extract values from HashMap
                            // Extract values from DB
                            let state: Vec<Heartbeat> = self.mesh_state.get_all();
                            let _ = respond_to.send(state).await;
                        }
                        Some(NodeCommand::SetSystemPrompt { prompt, respond_to }) => {
                            info!("Setting System Prompt to: {}", prompt);
                            self.chat_history.clear();
                            self.chat_history.add_system(prompt);
                            self.save_history();
                            let _ = respond_to.send(()).await;
                        }
                        Some(NodeCommand::Transcribe { audio_data, respond_to }) => {
                            info!("Received audio transcription request: {} samples", audio_data.len());
                            let engine = self.whisper_engine.lock().await;
                            match engine.transcribe(audio_data).await {
                                Ok(text) => {
                                    let _ = respond_to.send(text).await;
                                }
                                Err(e) => {
                                    error!("Transcribe failed: {}", e);
                                    let _ = respond_to.send(format!("Error: {}", e)).await;
                                }
                            }
                        }
                        Some(NodeCommand::GetSystemInfo { respond_to }) => {
                            self.system.refresh_all();

                            let cpu_model = self.system.cpus().first()
                                .map(|cpu| cpu.brand().to_string())
                                .unwrap_or_else(|| "Unknown CPU".to_string());

                            let total_memory_gb = self.system.total_memory() / 1024 / 1024 / 1024;
                            let used_memory_gb = self.system.used_memory() / 1024 / 1024 / 1024;
                            let cpu_cores = self.system.cpus().len();

                            let info = SystemCapabilities {
                                cpu_model,
                                total_memory_gb,
                                used_memory_gb,
                                cpu_cores,
                            };
                            let _ = respond_to.send(info).await;
                        }
                        Some(NodeCommand::StartPairing { respond_to }) => {
                            info!("Generating pairing code...");
                            let code = uuid::Uuid::new_v4().to_string().chars().take(8).collect::<String>().to_uppercase();

                            let addresses: Vec<String> = self.swarm.listeners().map(|a| a.to_string()).collect();
                            info!("Generated Pairing Code: {} for addresses: {:?}", code, addresses);

                            let response = PairingResponse {
                                code,
                                addresses,
                            };

                            let json_response = serde_json::to_string(&response).unwrap_or_default();
                            let _ = respond_to.send(json_response).await;
                        }
                        Some(NodeCommand::Search { query, limit, respond_to }) => {
                            info!("Processing Search Request: {}", query);
                            match self.embedder.embed(&query).await {
                                Ok(embedding) => {
                                    match self.vector_store.search(embedding, limit).await {
                                        Ok(results) => {
                                            let _ = respond_to.send(Ok(results)).await;
                                        }
                                        Err(e) => {
                                            error!("Search query failed: {}", e);
                                            let _ = respond_to.send(Err(e.to_string())).await;
                                        }
                                    }
                                }
                                Err(e) => {
                                    error!("Embedding failed for search: {}", e);
                                    let _ = respond_to.send(Err(e.to_string())).await;
                                }
                            }
                        }
                        Some(NodeCommand::ConnectPeer { address, respond_to }) => {
                            info!("ConnectPeer: Dialing {}", address);
                            match address.parse::<Multiaddr>() {
                                Ok(addr) => {
                                    // Extract peer_id from the address if present
                                    let peer_id = addr.iter().find_map(|p| {
                                        if let Protocol::P2p(id) = p { Some(id) } else { None }
                                    });

                                    // Add to Kademlia for discovery
                                    if let Some(pid) = peer_id {
                                        self.swarm.behaviour_mut().kademlia.add_address(&pid, addr.clone());
                                        self.paired_peers.push((pid, addr.clone()));
                                    }

                                    match self.swarm.dial(addr) {
                                        Ok(_) => {
                                            let _ = respond_to.send(Ok(())).await;
                                        }
                                        Err(e) => {
                                            let _ = respond_to.send(Err(format!("Dial failed: {}", e))).await;
                                        }
                                    }
                                }
                                Err(e) => {
                                    let _ = respond_to.send(Err(format!("Invalid address: {}", e))).await;
                                }
                            }
                        }
                        Some(NodeCommand::SetRelay { relay_address, respond_to }) => {
                            info!("SetRelay: Configuring relay at {}", relay_address);
                            match relay_address.parse::<Multiaddr>() {
                                Ok(addr) => {
                                    // Listen on the relay for inbound connections
                                    let mut listen_addr = addr.clone();
                                    listen_addr.push(Protocol::P2pCircuit);

                                    match self.swarm.listen_on(listen_addr) {
                                        Ok(_) => {
                                            self.relay_addr = Some(addr);
                                            info!("SetRelay: Now listening via relay");
                                            let _ = respond_to.send(Ok(())).await;
                                        }
                                        Err(e) => {
                                            let _ = respond_to.send(Err(format!("Relay listen failed: {}", e))).await;
                                        }
                                    }
                                }
                                Err(e) => {
                                    let _ = respond_to.send(Err(format!("Invalid relay address: {}", e))).await;
                                }
                            }
                        }
                        Some(NodeCommand::OffloadInference { query, respond_to }) => {
                            info!("OffloadInference: Looking for capable peer...");
                            if let Some(peer) = self.select_best_peer_for_offload() {
                                info!("OffloadInference: Sending to peer {}", peer);
                                let request_id = self.swarm.behaviour_mut().request_response.send_request(
                                    &peer,
                                    GenerateRequest { prompt: query },
                                );
                                self.pending_offloads.insert(request_id, respond_to);
                            } else {
                                let _ = respond_to.send(Err(
                                    "No connected peers available for inference offloading".to_string()
                                )).await;
                            }
                        }
                        None => {
                            // Channel closed
                            break;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn update_mesh_state(&mut self, heartbeat: Heartbeat) {
        if let Err(e) = self.mesh_state.update(heartbeat) {
            error!("Failed to update mesh state: {}", e);
        }
    }
}

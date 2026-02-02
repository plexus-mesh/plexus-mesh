use crate::{
    build_swarm,
    protocol::{Heartbeat, NodeCapabilities},
    swarm::PlexusBehaviourEvent,
    GenerateRequest, GenerateResponse, IdentityStore, PlexusBehaviour,
};
use anyhow::{Context, Result};
use futures::StreamExt;
use libp2p::{
    gossipsub::{self, IdentTopic},
    mdns,
    multiaddr::Protocol,
    request_response::{self, OutboundRequestId},
    swarm::SwarmEvent,
    Swarm,
};
use plexus_ai::{
    voice::WhisperEngine, BertEmbedder, ChatHistory, LLMEngine, LanceDbStore, QdrantStore,
    SimpleVectorStore, TinyLlamaEngine, VectorStore,
};
use std::collections::HashMap; // Use HashMap instead of CRDTs
use std::path::PathBuf;
use std::sync::Arc;
use sysinfo::{Networks, System};
use tokio::sync::mpsc;
use tokio::time::{interval, Duration};
use tracing::{error, info};

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
    }, // Return list of heartbeats (state)
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
}

use tokio::sync::Mutex;

// ...

pub struct NodeService {
    swarm: Swarm<PlexusBehaviour>,
    command_rx: mpsc::Receiver<NodeCommand>,
    ai_engine: Box<dyn LLMEngine>,             // Dynamic dispatch
    whisper_engine: Arc<Mutex<WhisperEngine>>, // Wrapped in Arc<Mutex>
    pending_requests: HashMap<OutboundRequestId, mpsc::Sender<String>>,
    chat_history: ChatHistory,
    history_path: PathBuf,
    embedder: BertEmbedder,
    vector_store: Arc<dyn VectorStore>,
    system: System,
    // REFACTOR: Use HashMap instead of crdts::Map for simplicity and build stability
    // REFACTOR: Use HashMap instead of crdts::Map for simplicity and build stability
    mesh_state: crate::crdt::MeshState,
    heartbeat_topic: IdentTopic,
    active_model: String,
}

impl NodeService {
    pub async fn new(
        identity_path: PathBuf,
        command_rx: mpsc::Receiver<NodeCommand>,
        model_id: String,
        bootstrap_peers: Vec<libp2p::Multiaddr>,
        data_dir: Option<PathBuf>,
    ) -> Result<Self> {
        info!("NodeService: Initializing...");
        info!("NodeService: Selected Model: {}", model_id);

        let identity_store = IdentityStore::new(identity_path.clone());
        info!("NodeService: Loading/Generating identity...");
        let keypair = identity_store
            .load_or_generate()
            .context("Failed to load identity")?;
        info!("NodeService: Identity loaded.");

        info!("NodeService: Building Swarm...");
        let mut swarm = build_swarm(keypair)
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

        info!("NodeService: Initializing AI Engine...");
        let ai_engine: Box<dyn plexus_ai::LLMEngine> = match model_id.as_str() {
            "tinyllama" => Box::new(TinyLlamaEngine::new()),
            "phi" => {
                // Placeholder for Phi engine when implemented
                // For now, fall back or error, but let's just warn and use TinyLlama for demo
                // warn!("Phi engine not yet implemented. Using TinyLlama.");
                // We use fully qualified path or ensure warn is imported.
                // Assuming tracing::* is prelude or imported.
                tracing::warn!("Phi engine not yet implemented. Using TinyLlama.");
                Box::new(TinyLlamaEngine::new())
            }
            _ => {
                tracing::warn!("Unknown model '{}'. Defaulting to TinyLlama.", model_id);
                Box::new(TinyLlamaEngine::new())
            }
        };

        info!("NodeService: Initializing Whisper Engine...");
        let whisper_engine = Arc::new(Mutex::new(WhisperEngine::new())); // Wrapped

        info!("NodeService: Initializing Embedder...");
        let embedder = BertEmbedder::new();

        // Determine Data Directory early for LanceDB
        let app_data_dir = if let Some(path) = data_dir.clone() {
            path
        } else {
            let project_dirs = directories_next::ProjectDirs::from("com", "plexus", "mesh")
                .context("Could not determine data directory")?;
            project_dirs.data_dir().to_path_buf()
        };
        std::fs::create_dir_all(&app_data_dir).context("Failed to create data directory")?;

        // Connect to Qdrant or Fallback with Timeout
        info!("NodeService: Connecting to Vector Store...");
        let qdrant_future = async { QdrantStore::new("http://localhost:6334").await };

        // 2-second timeout for Qdrant connection
        let vector_store: Arc<dyn VectorStore> =
            match tokio::time::timeout(Duration::from_secs(2), qdrant_future).await {
                Ok(Ok(store)) => {
                    info!("Connected to Qdrant Vector Database.");
                    Arc::new(store)
                }
                Ok(Err(e)) => {
                    error!(
                        "Failed to connect to Qdrant ({}). Falling back to LanceDB.",
                        e
                    );
                    let lance_path = app_data_dir.join("vectors.lance");
                    match LanceDbStore::new(&lance_path).await {
                        Ok(store) => {
                            info!("Connected to Embedded LanceDB at {:?}", lance_path);
                            Arc::new(store)
                        }
                        Err(e) => {
                            error!("Failed to init LanceDB: {}. Falling back to In-Memory.", e);
                            Arc::new(SimpleVectorStore::new())
                        }
                    }
                }
                Err(_) => {
                    error!("Qdrant connection timed out. Falling back to LanceDB.");
                    let lance_path = app_data_dir.join("vectors.lance");
                    match LanceDbStore::new(&lance_path).await {
                        Ok(store) => {
                            info!("Connected to Embedded LanceDB at {:?}", lance_path);
                            Arc::new(store)
                        }
                        Err(e) => {
                            error!("Failed to init LanceDB: {}. Falling back to In-Memory.", e);
                            Arc::new(SimpleVectorStore::new())
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
        let mut system = System::new_all();
        system.refresh_all();

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
            chat_history,
            history_path,
            embedder,
            vector_store,
            system,
            mesh_state,
            heartbeat_topic,
            active_model: model_id,
        })
    }

    fn save_history(&self) {
        let _ = self.chat_history.save_to_file(&self.history_path);
    }

    pub async fn run(mut self) -> Result<()> {
        // Listen on all interfaces
        self.swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

        let mut heartbeat_interval = interval(Duration::from_secs(10));

        loop {
            tokio::select! {
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
                        SwarmEvent::Behaviour(PlexusBehaviourEvent::Mdns(mdns::Event::Discovered(peers))) => {
                            for (peer, addr) in peers {
                                info!("MDNS Discovered: {} at {}", peer, addr);
                                self.swarm.behaviour_mut().kademlia.add_address(&peer, addr.clone());
                                if let Err(e) = self.swarm.dial(addr) {
                                     info!("Failed to dial {}: {}", peer, e);
                                }
                            }
                        }
                        SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                            info!("Connection established with {}", peer_id);
                        }
                        SwarmEvent::ConnectionClosed { peer_id, .. } => {
                            info!("Connection closed with {}", peer_id);
                        }
                        SwarmEvent::Behaviour(PlexusBehaviourEvent::RequestResponse(
                            request_response::Event::Message { peer, message }
                        )) => {
                            match message {
                                request_response::Message::Request { request, channel, .. } => {
                                    info!("Received remote generation request from {}: {}", peer, request.prompt);
                                    let response = match self.ai_engine.generate(&request.prompt).await {
                                        Ok(res) => res,
                                        Err(e) => format!("Error: {}", e),
                                    };
                                    let _ = self.swarm.behaviour_mut().request_response.send_response(channel, GenerateResponse { response });
                                }
                                request_response::Message::Response { request_id, response } => {
                                    info!("Received remote response: {}", response.response);
                                    if let Some(tx) = self.pending_requests.remove(&request_id) {
                                        let _ = tx.send(response.response).await;
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

                                // 0. RAG Retrieval
                                // We embed the prompt to find relevant context in our vector store.
                                if let Ok(query_vec) = self.embedder.embed(&prompt).await {
                                    if let Ok(results) = self.vector_store.search(query_vec, 1).await {
                                        if let Some((text, score)) = results.first() {
                                            // 0.4 is an arbitrary similarity threshold.
                                            if *score > 0.4 {
                                                info!("RAG Match found (score {:.2}): {}", score, text);
                                                // Inject context as a "System" message effectively
                                                let context_msg = format!("Context information: {}", text);
                                                self.chat_history.add_system(context_msg);
                                                self.save_history();
                                            }
                                        }
                                    }
                                }

                                // 1. Add User Message to History
                                self.chat_history.add_user(prompt.clone());
                                self.save_history();

                                // 2. Format with Context
                                let context_prompt = self.chat_history.format_for_llama();

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
                                        let _ = respond_to.send(token).await;
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
                                        let _ = stream_tx.send(format!("Error: {}", e)).await;
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

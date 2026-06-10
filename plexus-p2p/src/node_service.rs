use crate::{
    build_swarm_safe,
    protocol::{Heartbeat, NodeCapabilities},
    swarm::PlexusBehaviourEvent,
    GenerateRequest, GenerateResponse, IdentityStore, NegotiationRequest, NegotiationResponse,
    PlexusBehaviour, SyncRequest, SyncResponse,
};
use anyhow::{Context, Result};
use futures::StreamExt;
use libp2p::{
    core::ConnectedPoint,
    dcutr,
    gossipsub::{self, IdentTopic},
    identify, mdns,
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
use tokio::sync::{mpsc, oneshot};
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
    /// Like [`NodeCommand::Search`] but restricts results by metadata
    /// (`source_type` and/or a `date_key` range, all `YYYY-MM-DD`). Filtering
    /// happens inside the vector store, alongside the embedding and any
    /// over-fetch/fallback logic — see `VectorStore::search_filtered`.
    SearchFiltered {
        query: String,
        limit: usize,
        source_type: Option<String>,
        date_from: Option<String>,
        date_to: Option<String>,
        respond_to: mpsc::Sender<Result<Vec<(String, f32)>, String>>,
    },
    /// Embed `content` and store it under a caller-supplied `id`, so the app's
    /// memory UUID and the Rust CRDT row share a primary key. This is what lets
    /// the UI address a record's attachments by the same id it already holds.
    /// Contrast the `/save ` shorthand in [`NodeCommand::Generate`], which mints
    /// a throwaway timestamp id the caller never sees — fine for daily summaries
    /// (deduped by metadata), but useless for per-record attachment linking.
    SaveWithId {
        id: String,
        content: String,
        respond_to: mpsc::Sender<Result<(), String>>,
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
    /// Publish an already-encrypted CRDT sync payload to the sync gossipsub
    /// topic. Fire-and-forget: gossipsub is best-effort, so a publish with
    /// no current subscribers simply no-ops (the record will be picked up
    /// later by Merkle reconciliation — step 5). The bytes are opaque to
    /// the mesh layer; they're a `core::crypto` envelope wrapping a
    /// `core::sync_wire::WireRecord`, sealed by the app before it gets here.
    PublishSync {
        data: Vec<u8>,
    },
    /// Install (or replace) the sink that received sync-topic payloads are
    /// forwarded to. The app side drains this channel, decrypts, and feeds
    /// each payload to its LWW merge (`apply_remote`). Sent once during
    /// node startup — kept as a runtime command rather than a constructor
    /// arg so `NodeService::new`'s signature stays stable for the other
    /// (non-Mindora) binaries that call it.
    SetSyncSink {
        sender: mpsc::Sender<Vec<u8>>,
    },
    /// Install the channel the swarm uses to ask the app for Merkle
    /// reconciliation data (step 5). Like [`NodeCommand::SetSyncSink`], it's a
    /// runtime command so `NodeService::new`'s signature stays stable. The app
    /// drains the channel and answers each [`ReconcileQuery`]; while no
    /// responder is installed the swarm simply doesn't initiate or answer
    /// anti-entropy (gossipsub live-push still works).
    SetReconcileResponder {
        sender: mpsc::Sender<ReconcileQuery>,
    },
    /// Send one agent-negotiation turn to `peer_b58` and await its reply
    /// (Mindora 2.0). `payload_cbor` is the opaque, app-serialized
    /// `NegotiationPayload`; the swarm wraps it in a [`NegotiationRequest`],
    /// routes it over `/mindora/negotiate/1.0.0`, and resolves `reply` with the
    /// peer's response bytes (or a transport error string). The app's
    /// `NegotiationTransport` impl owns the byte ↔ payload conversion — the
    /// network layer never inspects the bytes.
    SendNegotiation {
        peer_b58: String,
        payload_cbor: Vec<u8>,
        reply: oneshot::Sender<Result<Vec<u8>, String>>,
    },
    /// Install the channel the swarm uses to ask the app to answer an inbound
    /// negotiation turn. Mirrors [`NodeCommand::SetReconcileResponder`]: a
    /// runtime command (keeps `NodeService::new` stable) carrying each inbound
    /// request as a [`NegotiationQuery`]. While no responder is installed,
    /// inbound negotiation requests are answered with an empty envelope (the
    /// initiator decodes that as a dropped turn and stops).
    SetNegotiationResponder {
        sender: mpsc::Sender<NegotiationQuery>,
    },
    /// Direct, single-shot local model generation on a fully-formed prompt —
    /// no chat history, no RAG augmentation, no remote offload. Used by the
    /// agent-negotiation evaluator so its prompts never pollute the user's
    /// conversation or persist to disk. Resolves with the model's full text or
    /// an error string.
    GenerateRaw {
        prompt: String,
        respond_to: oneshot::Sender<Result<String, String>>,
    },
    /// Install a sink that receives a peer's base-58 `PeerId` each time a
    /// connection to it is established. The app uses it to resume negotiations
    /// that dropped with that peer (Mindora 2.0, Step B). Best-effort
    /// (`try_send`); a full channel just skips a notification.
    SetPeerReconnectSink {
        sender: mpsc::Sender<String>,
    },
}

/// What the swarm asks the app when a negotiation request arrives from a peer.
///
/// Mirrors [`ReconcileQuery`]: `plexus-p2p` can't depend on the app's
/// negotiation engine (dependency cycle), so the app drains this channel,
/// deserializes `payload_cbor` into its `NegotiationPayload`, runs
/// `AgentReconciliationEngine::handle_inbound`, and replies with the response
/// bytes. The engine — the local AI gatekeeper — therefore stays the exclusive
/// arbiter of what the remote peer ever sees (SovereigntyShield).
pub enum NegotiationQuery {
    /// A peer sent us a negotiation turn. `peer_b58` is its base-58 `PeerId`;
    /// `payload_cbor` the opaque inbound bytes. The app replies with the
    /// response bytes (an empty `Vec` drops the turn).
    Inbound {
        peer_b58: String,
        payload_cbor: Vec<u8>,
        reply: oneshot::Sender<Vec<u8>>,
    },
}

/// A reconciliation question the swarm asks the app to answer. Every byte
/// blob is an opaque `core::crypto` envelope — the mesh layer never sees
/// plaintext, it only shuttles sealed bodies between the wire and the app.
#[derive(Debug)]
pub enum ReconcileQuery {
    /// "Give me my own sealed Merkle summary to advertise to a peer." The app
    /// replies with the sealed summary, or an empty `Vec` if this device is
    /// unpaired (nothing to advertise → the swarm won't initiate).
    Summary { reply: oneshot::Sender<Vec<u8>> },
    /// "A peer advertised this sealed summary — which of my records should I
    /// send back?" The app opens it, diffs it against its own tree, and
    /// replies with the sealed records for the differing day buckets (empty
    /// if unpaired or the summary won't open under any held key).
    Diff {
        peer_summary: Vec<u8>,
        reply: oneshot::Sender<Vec<Vec<u8>>>,
    },
    /// "Give me my own sealed blob have-set to advertise to a peer" (step 7).
    /// The app replies with its sealed set of attachment-blob hashes, or an
    /// empty `Vec` if unpaired (→ the swarm won't advertise blobs).
    BlobHaveSet { reply: oneshot::Sender<Vec<u8>> },
    /// "A peer advertised this sealed blob have-set — which of my blobs is it
    /// missing?" The app opens it, computes the set difference, and replies
    /// with the sealed blob payloads to send back (empty if unpaired or the
    /// have-set won't open under any held key).
    BlobDiff {
        peer_haveset: Vec<u8>,
        reply: oneshot::Sender<Vec<Vec<u8>>>,
    },
    /// "Store these sealed blobs a peer sent me" (step 7). Fire-and-forget:
    /// the app opens each envelope and puts it into the content-addressed
    /// store (verifying the hash). No reply — mirrors the record inbound sink.
    IngestBlobs { sealed_blobs: Vec<Vec<u8>> },
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
    /// Gossipsub topic carrying encrypted CRDT record pushes between paired
    /// devices. Separate from the heartbeat topic so the message handler can
    /// dispatch by topic hash without sniffing payloads.
    sync_topic: IdentTopic,
    /// Sink for inbound sync-topic payloads, installed via
    /// [`NodeCommand::SetSyncSink`]. `None` until the app wires it up; while
    /// `None`, received sync messages are dropped (the app isn't ready to
    /// merge them yet, and Merkle reconciliation will recover anything missed).
    sync_inbound: Option<mpsc::Sender<Vec<u8>>>,
    /// Channel the swarm uses to ask the app for reconciliation data,
    /// installed via [`NodeCommand::SetReconcileResponder`]. `None` until the
    /// app wires it up; while `None`, anti-entropy is inert (the device never
    /// initiates a summary exchange and answers inbound ones with nothing).
    sync_responder: Option<mpsc::Sender<ReconcileQuery>>,
    /// Pending outbound negotiation turns (mapped from request_id to the
    /// one-shot the app's `NegotiationTransport` is awaiting). Fulfilled when
    /// the peer's response — or an outbound failure — arrives.
    pending_negotiations: HashMap<OutboundRequestId, oneshot::Sender<Result<Vec<u8>, String>>>,
    /// Channel the swarm uses to ask the app to answer an inbound negotiation
    /// turn, installed via [`NodeCommand::SetNegotiationResponder`]. `None`
    /// until the app wires up its reconciliation engine; while `None`, inbound
    /// negotiation requests are answered with an empty envelope.
    negotiation_responder: Option<mpsc::Sender<NegotiationQuery>>,
    /// Sink notified with a peer's base-58 id on each connection establishment,
    /// installed via [`NodeCommand::SetPeerReconnectSink`]. Drives negotiation
    /// resume-after-reconnect; `None` until the app wires it up.
    peer_reconnect_sink: Option<mpsc::Sender<String>>,
    active_model: String,
    /// Known paired peers for auto-reconnection
    paired_peers: Vec<(PeerId, Multiaddr)>,
    /// Active relay address (for NAT traversal over mobile data)
    relay_addr: Option<Multiaddr>,
    /// Last successful connection time per peer (for exponential backoff)
    last_connect_attempt: HashMap<PeerId, Instant>,
    /// Reconnection backoff counter per peer
    reconnect_backoff: HashMap<PeerId, u32>,
    /// In-flight explicit dials (from `ConnectPeer`) awaiting their real
    /// outcome. `swarm.dial()` only reports that a dial was *initiated*; the
    /// actual success/failure arrives later as `ConnectionEstablished` /
    /// `OutgoingConnectionError`. We park the reply here keyed by the dialed
    /// peer and resolve it on that event, so the app sees the true result
    /// (e.g. a timeout) instead of a misleading immediate "ok".
    pending_dials: HashMap<PeerId, mpsc::Sender<Result<(), String>>>,
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

        // CRDT live-push topic. Subscribing unconditionally is safe: without
        // a pair key the app can't decrypt anything it receives here, so the
        // worst case is wasted gossip relay work, not a data leak.
        let sync_topic = IdentTopic::new("plexus-mesh/sync/1.0.0");
        swarm.behaviour_mut().gossipsub.subscribe(&sync_topic)?;

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
            sync_topic,
            sync_inbound: None,
            sync_responder: None,
            pending_negotiations: HashMap::new(),
            negotiation_responder: None,
            peer_reconnect_sink: None,
            active_model: model_id,
            paired_peers: Vec::new(),
            relay_addr: None,
            last_connect_attempt: HashMap::new(),
            reconnect_backoff: HashMap::new(),
            pending_dials: HashMap::new(),
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

    /// Stable TCP listen port: a restarted peer comes back at the SAME address,
    /// so the paired QR address and auto-reconnect survive an app restart
    /// (mDNS can't rediscover on iOS, so a fresh ephemeral port would strand
    /// the peer until a re-scan). An ephemeral TCP listener is kept as well so
    /// a rare port conflict can never leave us with no TCP transport at all.
    pub const MINDORA_TCP_PORT: u16 = 47474;

    pub async fn run(mut self) -> Result<()> {
        if let Err(e) = self
            .swarm
            .listen_on(format!("/ip4/0.0.0.0/tcp/{}", Self::MINDORA_TCP_PORT).parse()?)
        {
            warn!("Failed to bind stable TCP port {}: {}", Self::MINDORA_TCP_PORT, e);
        }
        // Ephemeral fallback (TCP) + QUIC.
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
                            // Dispatch by topic hash so the heartbeat and the
                            // sync streams never get parsed against each other.
                            if message.topic == self.sync_topic.hash() {
                                // Opaque encrypted CRDT payload. Hand it to the
                                // app's sink if installed; otherwise drop it
                                // (Merkle reconciliation will recover later).
                                if let Some(sink) = &self.sync_inbound {
                                    if let Err(e) = sink.try_send(message.data) {
                                        warn!("Sync inbound sink full/closed, dropping record: {}", e);
                                    }
                                } else {
                                    tracing::debug!("Received sync record but no sink installed yet; dropping");
                                }
                            } else if let Ok(heartbeat) = serde_json::from_slice::<Heartbeat>(&message.data) {
                                info!("Received Heartbeat from {}: {} Cores, {} MB RAM",
                                    heartbeat.peer_id,
                                    heartbeat.capabilities.cpu_cores,
                                    heartbeat.capabilities.total_memory / 1024 / 1024
                                );
                                self.update_mesh_state(heartbeat);
                            }
                        }
                        // Local-network discovery: add each mDNS-found peer to
                        // Kademlia and dial it, so two devices on the same Wi-Fi
                        // connect without any internet bootstrap.
                        SwarmEvent::Behaviour(PlexusBehaviourEvent::Mdns(mdns::Event::Discovered(peers))) => {
                            for (peer, addr) in peers {
                                info!("MDNS Discovered: {} at {}", peer, addr);
                                self.swarm.behaviour_mut().kademlia.add_address(&peer, addr.clone());
                                if let Err(e) = self.swarm.dial(addr) {
                                     info!("Failed to dial {}: {}", peer, e);
                                }
                            }
                        }
                        SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } => {
                            info!("Connection established with {}", peer_id);
                            // Reset backoff on successful connection
                            self.reconnect_backoff.remove(&peer_id);
                            // A restart-surviving dial address for this peer:
                            //  * outbound: the address we just successfully
                            //    dialed — provably reachable.
                            //  * inbound: the remote's source port is an
                            //    ephemeral outbound port (not dialable later),
                            //    so rewrite to the stable listen port every
                            //    node binds (MINDORA_TCP_PORT) at the observed
                            //    IP — heals a peer's DHCP/IP change without a
                            //    re-scan.
                            let reachable: Option<Multiaddr> = match &endpoint {
                                ConnectedPoint::Dialer { address, .. } => Some(address.clone()),
                                ConnectedPoint::Listener { send_back_addr, .. } => send_back_addr
                                    .iter()
                                    .find_map(|p| match p {
                                        Protocol::Ip4(ip) => Some(ip),
                                        _ => None,
                                    })
                                    .map(|ip| {
                                        let mut a = Multiaddr::empty();
                                        a.push(Protocol::Ip4(ip));
                                        a.push(Protocol::Tcp(Self::MINDORA_TCP_PORT));
                                        a.push(Protocol::P2p(peer_id));
                                        a
                                    }),
                            };
                            // Self-heal the reconnect target — but only for
                            // peers already registered as paired; an arbitrary
                            // inbound connection must never enrol itself into
                            // the auto-reconnect loop.
                            if let Some(addr) = &reachable {
                                if let Some(entry) =
                                    self.paired_peers.iter_mut().find(|(p, _)| p == &peer_id)
                                {
                                    entry.1 = addr.clone();
                                }
                            }
                            // Anti-entropy: catch up anything missed while one
                            // of us was offline. No-op unless paired.
                            self.initiate_reconcile(peer_id).await;
                            // Notify the app so it can resume any negotiation
                            // that dropped with this peer (Mindora 2.0, Step B).
                            // Payload is "b58" or "b58|<reachable multiaddr>";
                            // the address half lets the app persist a fresh
                            // dial address (see Mindora's peer-address bus).
                            if let Some(tx) = &self.peer_reconnect_sink {
                                let payload = match &reachable {
                                    Some(a) => format!("{}|{}", peer_id.to_base58(), a),
                                    None => peer_id.to_base58(),
                                };
                                let _ = tx.try_send(payload);
                            }
                            // Resolve an explicit ConnectPeer dial that was waiting
                            // on this connection's real outcome.
                            if let Some(reply) = self.pending_dials.remove(&peer_id) {
                                let _ = reply.send(Ok(())).await;
                            }
                        }
                        SwarmEvent::ConnectionClosed { peer_id, .. } => {
                            info!("Connection closed with {}", peer_id);
                        }
                        SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                            warn!("Outgoing connection error to {:?}: {}", peer_id, error);
                            // Surface the real dial failure (timeout, refused,
                            // handshake, …) back to a waiting ConnectPeer caller.
                            if let Some(pid) = peer_id {
                                if let Some(reply) = self.pending_dials.remove(&pid) {
                                    let _ = reply.send(Err(format!("{}", error))).await;
                                }
                            }
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
                        SwarmEvent::Behaviour(PlexusBehaviourEvent::SyncRr(
                            request_response::Event::Message { peer, message }
                        )) => {
                            match message {
                                // A peer advertised its sealed Merkle summary.
                                // Ask the app which of our records differ and
                                // ship them straight back as the response.
                                request_response::Message::Request { request, channel, .. } => {
                                    // Two request kinds share this protocol; each
                                    // maps to its own ReconcileQuery and reply.
                                    let response = match request {
                                        SyncRequest::Summary { sealed_summary } => {
                                            info!("Sync summary from {} ({} bytes)", peer, sealed_summary.len());
                                            let sealed_records = match &self.sync_responder {
                                                Some(responder) => {
                                                    let (tx, rx) = oneshot::channel();
                                                    if responder
                                                        .send(ReconcileQuery::Diff { peer_summary: sealed_summary, reply: tx })
                                                        .await
                                                        .is_ok()
                                                    {
                                                        rx.await.unwrap_or_default()
                                                    } else {
                                                        Vec::new()
                                                    }
                                                }
                                                None => Vec::new(),
                                            };
                                            SyncResponse::Records { sealed_records }
                                        }
                                        SyncRequest::BlobHaveSet { sealed_haveset } => {
                                            info!("Blob have-set from {} ({} bytes)", peer, sealed_haveset.len());
                                            let sealed_blobs = match &self.sync_responder {
                                                Some(responder) => {
                                                    let (tx, rx) = oneshot::channel();
                                                    if responder
                                                        .send(ReconcileQuery::BlobDiff { peer_haveset: sealed_haveset, reply: tx })
                                                        .await
                                                        .is_ok()
                                                    {
                                                        rx.await.unwrap_or_default()
                                                    } else {
                                                        Vec::new()
                                                    }
                                                }
                                                None => Vec::new(),
                                            };
                                            SyncResponse::Blobs { sealed_blobs }
                                        }
                                    };
                                    let _ = self
                                        .swarm
                                        .behaviour_mut()
                                        .sync_rr
                                        .send_response(channel, response);
                                }
                                request_response::Message::Response { response, .. } => match response {
                                    // The records we're missing arrived. Feed each
                                    // through the same inbound sink as live-push so
                                    // they hit `ingest_sealed` (decrypt → LWW merge).
                                    SyncResponse::Records { sealed_records } => {
                                        info!("Sync delivered {} records", sealed_records.len());
                                        if let Some(sink) = &self.sync_inbound {
                                            for record in sealed_records {
                                                if let Err(e) = sink.try_send(record) {
                                                    warn!("Sync inbound sink full/closed during reconcile: {}", e);
                                                }
                                            }
                                        } else {
                                            tracing::debug!("Reconcile records arrived but no sink installed; dropping");
                                        }
                                    }
                                    // The blobs we're missing arrived. Hand them to
                                    // the app to open + store (content-addressed,
                                    // hash-verified) via the responder channel.
                                    SyncResponse::Blobs { sealed_blobs } => {
                                        info!("Sync delivered {} blobs", sealed_blobs.len());
                                        if sealed_blobs.is_empty() {
                                            // nothing to ingest
                                        } else if let Some(responder) = &self.sync_responder {
                                            if responder
                                                .send(ReconcileQuery::IngestBlobs { sealed_blobs })
                                                .await
                                                .is_err()
                                            {
                                                warn!("Reconcile responder dropped; blobs not ingested");
                                            }
                                        } else {
                                            tracing::debug!("Reconcile blobs arrived but no responder installed; dropping");
                                        }
                                    }
                                },
                            }
                        }
                        SwarmEvent::Behaviour(PlexusBehaviourEvent::Negotiation(
                            request_response::Event::Message { peer, message }
                        )) => {
                            match message {
                                // A peer sent us a negotiation turn. Hand the
                                // opaque bytes to the app's engine (the local AI
                                // gatekeeper) and ship its reply straight back.
                                request_response::Message::Request { request, channel, .. } => {
                                    let payload_cbor = match &self.negotiation_responder {
                                        Some(responder) => {
                                            let (tx, rx) = oneshot::channel();
                                            if responder
                                                .send(NegotiationQuery::Inbound {
                                                    peer_b58: peer.to_base58(),
                                                    payload_cbor: request.payload_cbor,
                                                    reply: tx,
                                                })
                                                .await
                                                .is_ok()
                                            {
                                                rx.await.unwrap_or_default()
                                            } else {
                                                Vec::new()
                                            }
                                        }
                                        None => {
                                            tracing::debug!(
                                                "Negotiation request from {} dropped; no responder installed",
                                                peer
                                            );
                                            Vec::new()
                                        }
                                    };
                                    let _ = self
                                        .swarm
                                        .behaviour_mut()
                                        .negotiation
                                        .send_response(channel, NegotiationResponse { payload_cbor });
                                }
                                // The peer's reply to a turn we initiated.
                                request_response::Message::Response { request_id, response } => {
                                    if let Some(tx) = self.pending_negotiations.remove(&request_id) {
                                        let _ = tx.send(Ok(response.payload_cbor));
                                    }
                                }
                            }
                        }
                        SwarmEvent::Behaviour(PlexusBehaviourEvent::Negotiation(
                            request_response::Event::OutboundFailure { request_id, error, .. }
                        )) => {
                            // Free the waiting transport instead of letting it
                            // hang until its own timeout.
                            if let Some(tx) = self.pending_negotiations.remove(&request_id) {
                                let _ = tx.send(Err(format!("negotiation outbound failure: {error}")));
                            }
                        }
                        SwarmEvent::Behaviour(PlexusBehaviourEvent::Identify(
                            identify::Event::Received { peer_id, info, .. }
                        )) => {
                            // Feed the peer's advertised listen addresses into
                            // Kademlia so dcutr / negotiation can reach it
                            // directly as connectivity changes.
                            for addr in info.listen_addrs {
                                self.swarm.behaviour_mut().kademlia.add_address(&peer_id, addr);
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
                        Some(NodeCommand::SaveWithId { id, content, respond_to }) => {
                            info!("Saving to memory (client id {}): {}", id, content);
                            match self.embedder.embed(&content).await {
                                Ok(embedding) => {
                                    let result = self
                                        .vector_store
                                        .add_document(&id, &content, embedding)
                                        .await
                                        .map_err(|e| e.to_string());
                                    let _ = respond_to.send(result).await;
                                }
                                Err(e) => {
                                    let _ = respond_to
                                        .send(Err(format!("Error embedding: {}", e)))
                                        .await;
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
                        Some(NodeCommand::SearchFiltered { query, limit, source_type, date_from, date_to, respond_to }) => {
                            info!("Processing Filtered Search Request: {} (type={:?})", query, source_type);
                            match self.embedder.embed(&query).await {
                                Ok(embedding) => {
                                    match self
                                        .vector_store
                                        .search_filtered(embedding, limit, source_type, date_from, date_to)
                                        .await
                                    {
                                        Ok(results) => {
                                            let _ = respond_to.send(Ok(results)).await;
                                        }
                                        Err(e) => {
                                            error!("Filtered search query failed: {}", e);
                                            let _ = respond_to.send(Err(e.to_string())).await;
                                        }
                                    }
                                }
                                Err(e) => {
                                    error!("Embedding failed for filtered search: {}", e);
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
                                        // Replace any existing entry for this peer so a
                                        // re-scan's fresh address supersedes a stale one
                                        // (ports change across restarts) — otherwise the
                                        // reconnect loop keeps dialing dead addresses.
                                        self.paired_peers.retain(|(existing, _)| existing != &pid);
                                        self.paired_peers.push((pid, addr.clone()));
                                    }

                                    match self.swarm.dial(addr) {
                                        Ok(_) => match peer_id {
                                            // Resolve later, on the real connection
                                            // outcome (ConnectionEstablished /
                                            // OutgoingConnectionError).
                                            Some(pid) => {
                                                self.pending_dials.insert(pid, respond_to);
                                            }
                                            // No peer id in the address — can't
                                            // correlate the outcome; report that the
                                            // dial at least started.
                                            None => {
                                                let _ = respond_to.send(Ok(())).await;
                                            }
                                        },
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
                        Some(NodeCommand::PublishSync { data }) => {
                            // Best-effort live push. `InsufficientPeers` is the
                            // expected, benign case when this device is alone on
                            // the mesh — log at debug, not error, so it doesn't
                            // spam the console for single-device users.
                            match self.swarm.behaviour_mut().gossipsub.publish(self.sync_topic.clone(), data) {
                                Ok(_) => tracing::debug!("Published sync record"),
                                Err(gossipsub::PublishError::InsufficientPeers) => {
                                    tracing::debug!("No sync peers connected; record will reconcile later");
                                }
                                Err(e) => warn!("Failed to publish sync record: {}", e),
                            }
                        }
                        Some(NodeCommand::SetSyncSink { sender }) => {
                            info!("Sync inbound sink installed");
                            self.sync_inbound = Some(sender);
                        }
                        Some(NodeCommand::SetReconcileResponder { sender }) => {
                            info!("Reconcile responder installed");
                            self.sync_responder = Some(sender);
                        }
                        Some(NodeCommand::SetNegotiationResponder { sender }) => {
                            info!("Negotiation responder installed");
                            self.negotiation_responder = Some(sender);
                        }
                        Some(NodeCommand::SetPeerReconnectSink { sender }) => {
                            info!("Peer-reconnect sink installed");
                            self.peer_reconnect_sink = Some(sender);
                        }
                        Some(NodeCommand::GenerateRaw { prompt, respond_to }) => {
                            // Direct local generation — no history, no RAG, no
                            // offload. Errors are returned to the caller rather
                            // than surfaced as chat output.
                            let result = self
                                .ai_engine
                                .generate(&prompt)
                                .await
                                .map_err(|e| e.to_string());
                            let _ = respond_to.send(result);
                        }
                        Some(NodeCommand::SendNegotiation { peer_b58, payload_cbor, reply }) => {
                            match peer_b58.parse::<PeerId>() {
                                Ok(peer) => {
                                    let request_id = self.swarm.behaviour_mut().negotiation.send_request(
                                        &peer,
                                        NegotiationRequest { payload_cbor },
                                    );
                                    // Resolved when the matching Response /
                                    // OutboundFailure event arrives.
                                    self.pending_negotiations.insert(request_id, reply);
                                }
                                Err(e) => {
                                    let _ = reply.send(Err(format!("invalid peer id '{peer_b58}': {e}")));
                                }
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

    /// Kick off a Merkle reconciliation with a freshly-connected peer: ask the
    /// app for our own sealed summary and, if we have one (i.e. we're paired),
    /// advertise it on `/plexus/sync/1.0.0`. The peer answers with the sealed
    /// records we're missing. Inert when no responder is installed or the
    /// device is unpaired (empty summary → nothing to advertise).
    async fn initiate_reconcile(&mut self, peer_id: PeerId) {
        let responder = match &self.sync_responder {
            Some(r) => r.clone(),
            None => return,
        };
        let (tx, rx) = oneshot::channel();
        if responder
            .send(ReconcileQuery::Summary { reply: tx })
            .await
            .is_err()
        {
            warn!("Reconcile responder dropped; cannot initiate sync");
            return;
        }
        let sealed_summary = match rx.await {
            Ok(s) => s,
            Err(_) => return,
        };
        if sealed_summary.is_empty() {
            // Unpaired (no key) — advertise nothing.
            return;
        }
        self.swarm
            .behaviour_mut()
            .sync_rr
            .send_request(&peer_id, SyncRequest::Summary { sealed_summary });
        info!("Initiated Merkle reconciliation with {}", peer_id);

        // Step 7: also advertise our attachment-blob have-set so the peer can
        // push back any blob we're missing. Independent of the record round
        // above — request_response multiplexes both in flight. Unpaired
        // devices return an empty have-set and advertise nothing.
        let (btx, brx) = oneshot::channel();
        if responder
            .send(ReconcileQuery::BlobHaveSet { reply: btx })
            .await
            .is_err()
        {
            return;
        }
        let sealed_haveset = match brx.await {
            Ok(s) => s,
            Err(_) => return,
        };
        if sealed_haveset.is_empty() {
            return;
        }
        self.swarm
            .behaviour_mut()
            .sync_rr
            .send_request(&peer_id, SyncRequest::BlobHaveSet { sealed_haveset });
        info!("Advertised blob have-set to {}", peer_id);
    }
}

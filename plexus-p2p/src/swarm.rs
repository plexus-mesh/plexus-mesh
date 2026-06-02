use crate::{GenerateRequest, GenerateResponse, SyncRequest, SyncResponse};
use anyhow::{Context, Result};
use libp2p::{
    dcutr, gossipsub,
    identity::Keypair,
    kad, noise, relay,
    request_response::{self, cbor, ProtocolSupport},
    swarm::NetworkBehaviour,
    tcp, yamux, StreamProtocol, Swarm,
};
use std::time::Duration;
use tracing::info;

/// Production-ready network behaviour combining:
///
/// - **Gossipsub**: Pub/sub for heartbeats and capability announcements
/// - **Kademlia**: Peer discovery and DHT-based routing
/// - **Request-Response**: Point-to-point compute offloading (AI inference)
/// - **Relay Client**: NAT traversal via relay nodes (for mobile data)
/// - **DCUtR**: Direct Connection Upgrade through Relay (hole-punching)
///
/// This enables mobile devices on cellular data to reach desktop nodes
/// behind home NATs by first connecting through a relay, then upgrading
/// to a direct connection when possible.
#[derive(NetworkBehaviour)]
pub struct PlexusBehaviour {
    pub gossipsub: gossipsub::Behaviour,
    pub kademlia: kad::Behaviour<kad::store::MemoryStore>,
    pub request_response: cbor::Behaviour<GenerateRequest, GenerateResponse>,
    /// Point-to-point anti-entropy (Merkle reconciliation, step 5). Separate
    /// from `request_response` (compute offload) so the two never share a
    /// protocol or get parsed against each other. Payloads are opaque sealed
    /// envelopes — see [`SyncRequest`].
    pub sync_rr: cbor::Behaviour<SyncRequest, SyncResponse>,
    pub relay_client: relay::client::Behaviour,
    pub dcutr: dcutr::Behaviour,
}

/// Build a production-ready swarm with full NAT traversal support.
///
/// Transport stack (layered, in priority order):
/// 1. **QUIC** — UDP-based, faster handshakes, works through more NATs
/// 2. **TCP + Noise + Yamux** — fallback when QUIC is blocked
/// 3. **Relay** — circuit relay for when direct connections are impossible
/// 4. **DCUtR** — upgrades relay connections to direct via hole-punching
///
/// All connections are encrypted end-to-end via Noise XX handshake
/// (ed25519 identity keys). No plaintext is ever transmitted.
///
/// # Security Model
///
/// - Every connection is mutually authenticated via the node's ed25519 keypair
/// - Relay nodes cannot read message content (Noise encryption is end-to-end)
/// - Kademlia is used for peer discovery only — no sensitive data in DHT
/// - Request-Response payloads (AI prompts/responses) are inside the encrypted channel
pub async fn build_swarm_safe(keypair: Keypair) -> Result<Swarm<PlexusBehaviour>> {
    info!("build_swarm: Starting production swarm build...");
    let peer_id = keypair.public().to_peer_id();

    // ── Gossipsub ────────────────────────────────────────────────────────
    let gossipsub_config = gossipsub::ConfigBuilder::default()
        .heartbeat_interval(Duration::from_secs(15))
        .validation_mode(gossipsub::ValidationMode::Strict)
        // Mesh parameters tuned for small personal networks (2-10 nodes).
        //
        // libp2p gossipsub enforces `mesh_outbound_min <= mesh_n / 2` at
        // build time. The crate's default `mesh_outbound_min` is 2, but
        // we picked `mesh_n = 3` for tiny meshes — integer division gives
        // `3 / 2 = 1`, so the default 2 fails validation and `build()`
        // returns "The inequality doesn't hold mesh_outbound_min <=
        // self.config.mesh_n / 2", which used to crash the node on
        // every boot. Set it to 1 explicitly so the small-mesh design
        // intent is encoded in source.
        .mesh_n(3)
        .mesh_n_low(2)
        .mesh_n_high(6)
        .mesh_outbound_min(1)
        .gossip_lazy(3)
        .history_length(5)
        .history_gossip(3)
        // Message size limit: 256KB (enough for AI prompts + context)
        .max_transmit_size(256 * 1024)
        .build()
        .map_err(|e| anyhow::anyhow!(e))
        .context("Failed to build gossipsub config")?;

    let gossipsub = gossipsub::Behaviour::new(
        gossipsub::MessageAuthenticity::Signed(keypair.clone()),
        gossipsub_config,
    )
    .map_err(|e| anyhow::anyhow!(e))
    .context("Failed to init gossipsub behaviour")?;

    // ── Kademlia ─────────────────────────────────────────────────────────
    let kademlia_store = kad::store::MemoryStore::new(peer_id);
    let mut kademlia_config = kad::Config::default();
    // Personal mesh: fewer replication targets, longer record TTL
    kademlia_config.set_replication_factor(std::num::NonZeroUsize::new(3).unwrap());
    kademlia_config.set_query_timeout(Duration::from_secs(30));
    let kademlia = kad::Behaviour::with_config(peer_id, kademlia_store, kademlia_config);

    // ── Request-Response ─────────────────────────────────────────────────
    let mut rr_config = request_response::Config::default();
    // 5-minute timeout for AI generation (large models on desktop can be slow)
    rr_config.set_request_timeout(Duration::from_secs(300));

    let request_response = cbor::Behaviour::new(
        [(
            StreamProtocol::new("/plexus/compute/1.0.0"),
            ProtocolSupport::Full,
        )],
        rr_config,
    );

    // ── Sync Request-Response (Merkle reconciliation) ────────────────────
    // Its own protocol and a tighter timeout: a summary exchange is a couple
    // of small encrypted blobs, not a minutes-long model generation.
    let mut sync_rr_config = request_response::Config::default();
    sync_rr_config.set_request_timeout(Duration::from_secs(60));
    let sync_rr = cbor::Behaviour::new(
        [(
            StreamProtocol::new("/plexus/sync/1.0.0"),
            ProtocolSupport::Full,
        )],
        sync_rr_config,
    );

    info!("build_swarm: Configuring transport stack...");

    // ── Transport: TCP + QUIC + Relay, all with Noise encryption ─────────
    // The relay client behaviour is injected by the builder via with_relay_client(),
    // so we construct it inside the with_behaviour closure.
    let swarm = libp2p::SwarmBuilder::with_existing_identity(keypair)
        .with_tokio()
        .with_tcp(
            tcp::Config::default().nodelay(true),
            noise::Config::new,
            yamux::Config::default,
        )
        .context("Failed to init TCP transport")?
        .with_quic()
        .with_relay_client(noise::Config::new, yamux::Config::default)
        .context("Failed to init relay client transport")?
        .with_behaviour(|key, relay_client| {
            let peer_id = key.public().to_peer_id();

            // DCUtR: upgrades relay connections to direct via hole-punching
            let dcutr = dcutr::Behaviour::new(peer_id);

            Ok(PlexusBehaviour {
                gossipsub,
                kademlia,
                request_response,
                sync_rr,
                relay_client,
                dcutr,
            })
        })
        .context("Failed to build Swarm behaviour")?
        .with_swarm_config(|c| {
            c.with_idle_connection_timeout(Duration::from_secs(120))
                // Notify handler buffer size — large enough for concurrent requests
                .with_notify_handler_buffer_size(std::num::NonZeroUsize::new(32).unwrap())
                // Per-connection event buffer
                .with_per_connection_event_buffer_size(16)
        })
        .build();

    info!(
        "build_swarm: Production swarm ready. PeerID: {}",
        swarm.local_peer_id()
    );
    Ok(swarm)
}

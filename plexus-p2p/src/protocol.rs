use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct NodeCapabilities {
    pub cpu_cores: usize,
    pub total_memory: u64, // Bytes
    pub gpu_info: Option<String>,
    pub model_loaded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Heartbeat {
    pub peer_id: String,
    pub model: String, // Active model (e.g. tinyllama)
    pub capabilities: NodeCapabilities,
    pub timestamp: u64, // Unix timestamp for LWW
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub response: String,
}

/// Anti-entropy (Merkle reconciliation, step 5) request between paired
/// devices, carried point-to-point inside the Noise-encrypted channel on
/// `/plexus/sync/1.0.0`. The byte blobs are **opaque** to the mesh layer:
/// they are `core::crypto` envelopes sealed by the app before they get here,
/// so no plaintext (not even the day buckets) ever crosses the wire.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncRequest {
    /// "Here is my sealed Merkle summary — send back the records I'm missing
    /// or that differ." The responder opens this against its own pair keys,
    /// diffs it against its own tree, and replies with [`SyncResponse::Records`].
    /// An unpaired device advertises nothing, so it never sends this.
    Summary { sealed_summary: Vec<u8> },
    /// "Here is the sealed set of attachment-blob hashes I already hold — send
    /// back any blob you have that isn't in it." (step 7). The responder opens
    /// the have-set, computes which of *its* blobs the requester lacks, and
    /// replies with [`SyncResponse::Blobs`]. Like [`Summary`](Self::Summary)
    /// the body is an opaque `core::crypto` envelope: the mesh never sees a
    /// hash in the clear, and an unpaired device advertises nothing.
    BlobHaveSet { sealed_haveset: Vec<u8> },
}

/// Reply to a [`SyncRequest`]. See that type for the privacy model — these
/// records are individually sealed envelopes the requester feeds straight
/// into its LWW merge (`ingest_sealed`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncResponse {
    /// The responder's sealed records for the day buckets where the two
    /// devices' Merkle trees disagree. Empty when the devices already agree,
    /// when the responder is unpaired, or when the advertised summary won't
    /// open under any held key. Replies to [`SyncRequest::Summary`].
    Records { sealed_records: Vec<Vec<u8>> },
    /// The responder's sealed blob payloads the requester was missing — each
    /// an opaque `core::crypto` envelope the requester feeds into its
    /// content-addressed store (verifying the hash on open). Empty when the
    /// requester already has everything, or the responder is unpaired. Replies
    /// to [`SyncRequest::BlobHaveSet`].
    Blobs { sealed_blobs: Vec<Vec<u8>> },
}

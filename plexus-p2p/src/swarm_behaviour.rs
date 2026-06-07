//! Composite `NetworkBehaviour` for the generic agent-negotiation protocol.
//!
//! This is the network binding for Mindora 2.0's negotiation engine (which lives
//! in the app crate, `rust_lib_mindora::core::negotiation`). It bundles exactly
//! the three behaviours the protocol needs:
//!
//! - [`request_response::cbor::Behaviour`] for low-latency, type-safe agent
//!   chatter (one request → one response per turn),
//! - [`dcutr::Behaviour`] to upgrade relayed connections to direct ones via
//!   hole-punching (so two phones behind NATs talk peer-to-peer),
//! - [`identify::Behaviour`] so peers learn each other's listen addresses and
//!   protocols as connectivity changes across Wi-Fi / cellular / relay.
//!
//! # The byte-envelope boundary (why this crate stays type-agnostic)
//!
//! `plexus-p2p` is a *dependency of* the app crate, so it cannot import the
//! app's `NegotiationPayload` without a dependency cycle. Instead the wire types
//! here carry an **opaque CBOR byte envelope** ([`NegotiationRequest::payload_cbor`]).
//! The strongly-typed `NegotiationPayload` is serialized to those bytes by the
//! app's `NegotiationTransport` implementation and deserialized again on the
//! far side. The network layer therefore routes bytes and knows nothing about
//! intents or constraints — which is precisely the "transport-agnostic protocol"
//! property the engine is designed around. Adding a new `ProtocolIntent` never
//! touches this file.

use std::time::Duration;

use libp2p::{
    dcutr, identify,
    identity::PublicKey,
    request_response::{self, cbor, ProtocolSupport},
    swarm::NetworkBehaviour,
    PeerId, StreamProtocol,
};
use serde::{Deserialize, Serialize};

/// request_response wire protocol id for agent negotiation. Bump the version
/// suffix only on a *breaking* change to the envelope framing (not the inner
/// payload — that's the app's concern and is versioned independently).
pub const NEGOTIATION_PROTOCOL: &str = "/mindora/negotiate/1.0.0";

/// `agentVersion`-style identify protocol string advertised to peers.
pub const IDENTIFY_PROTOCOL: &str = "/mindora/id/1.0.0";

/// Request timeout for a single negotiation turn. Must comfortably cover a full
/// on-device LLM evaluation on the *remote* peer — a small model (e.g. Gemma 2B)
/// generating a verdict can take tens of seconds, more on a cold model load —
/// while still freeing resources if a peer goes dark mid-turn. 30s proved too
/// tight on real devices (the responder was still evaluating when the initiator
/// timed out with "Timeout while waiting for a response").
const NEGOTIATION_TIMEOUT: Duration = Duration::from_secs(120);

/// Opaque request envelope: the CBOR-serialized bytes of the app's
/// `NegotiationPayload`. This crate never inspects them.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NegotiationRequest {
    pub payload_cbor: Vec<u8>,
}

/// Opaque response envelope (the peer's reply turn, CBOR bytes).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NegotiationResponse {
    pub payload_cbor: Vec<u8>,
}

/// Composite behaviour orchestrating agent negotiation over the mesh.
///
/// The `#[derive(NetworkBehaviour)]` macro generates `AgentNetworkBehaviourEvent`
/// — match on it in the swarm event loop to drive inbound requests into the
/// app's reconciliation engine (see [`NegotiationHandler`]).
#[derive(NetworkBehaviour)]
pub struct AgentNetworkBehaviour {
    /// One-shot request/response agent chatter, CBOR-framed.
    pub negotiation: cbor::Behaviour<NegotiationRequest, NegotiationResponse>,
    /// Hole-punching to promote relayed connections to direct.
    pub dcutr: dcutr::Behaviour,
    /// Peer address/protocol discovery as connectivity changes.
    pub identify: identify::Behaviour,
}

impl AgentNetworkBehaviour {
    /// Construct the composite behaviour for `peer_id` advertising
    /// `local_public_key` over identify.
    pub fn new(peer_id: PeerId, local_public_key: PublicKey) -> Self {
        let negotiation = cbor::Behaviour::new(
            [(StreamProtocol::new(NEGOTIATION_PROTOCOL), ProtocolSupport::Full)],
            request_response::Config::default().with_request_timeout(NEGOTIATION_TIMEOUT),
        );

        AgentNetworkBehaviour {
            negotiation,
            dcutr: dcutr::Behaviour::new(peer_id),
            identify: identify::Behaviour::new(identify::Config::new(
                IDENTIFY_PROTOCOL.to_string(),
                local_public_key,
            )),
        }
    }
}

/// Inversion-of-control seam for inbound dispatch.
///
/// `plexus-p2p` can't depend on the app's negotiation engine (cycle), so the
/// application implements this and registers it with the node. The swarm event
/// loop, on receiving a [`NegotiationRequest`], calls [`on_request`] and writes
/// the returned [`NegotiationResponse`] back through the request_response
/// channel. The implementation is where the app deserializes the bytes into a
/// `NegotiationPayload`, runs `AgentReconciliationEngine::handle_inbound`, and
/// re-serializes the reply — keeping all negotiation *meaning* in the app crate.
///
/// [`on_request`]: NegotiationHandler::on_request
pub trait NegotiationHandler: Send + Sync + 'static {
    fn on_request(
        &self,
        peer: PeerId,
        request: NegotiationRequest,
    ) -> impl std::future::Future<Output = NegotiationResponse> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::identity::Keypair;

    #[test]
    fn builds_with_all_three_behaviours() {
        // Smoke test: the composite behaviour constructs and the generated
        // event type exists (compile-time proof the derive succeeded).
        let kp = Keypair::generate_ed25519();
        let peer_id = kp.public().to_peer_id();
        let _behaviour = AgentNetworkBehaviour::new(peer_id, kp.public());
        // Touch the generated event enum so it can't be optimized/renamed away.
        fn _accepts_event(_e: AgentNetworkBehaviourEvent) {}
    }

    #[test]
    fn negotiation_envelope_serde_round_trips() {
        // The wire codec is CBOR (request_response::cbor); serde_json exercises
        // the same derive without a transitive-only dependency.
        let req = NegotiationRequest {
            payload_cbor: vec![1, 2, 3, 4],
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: NegotiationRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.payload_cbor, req.payload_cbor);
    }
}

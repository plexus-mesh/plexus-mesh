use plexus_p2p::crdt::MeshState;
use plexus_p2p::protocol::{Heartbeat, NodeCapabilities};
use proptest::prelude::*;
use tempfile::Builder;

// Strategy to generate random Heartbeats
fn heartbeat_strategy() -> impl Strategy<Value = Heartbeat> {
    (
        "[a-z0-9]{10}", // peer_id
        "[a-z0-9]{5}",  // model
        any::<u64>(),   // timestamp
        any::<usize>(), // cpu_cores
        any::<u64>(),   // total_memory
    )
        .prop_map(
            |(peer_id, model, timestamp, cpu_cores, total_memory)| Heartbeat {
                peer_id,
                model,
                timestamp,
                capabilities: NodeCapabilities {
                    cpu_cores,
                    total_memory,
                    gpu_info: None,
                    model_loaded: true,
                },
            },
        )
}

// Helper to create a temporary DB for tests
fn create_temp_mesh_state() -> MeshState {
    let dir = Builder::new().prefix("plexus_crdt_test").tempdir().unwrap();
    MeshState::new(dir.path().to_path_buf()).expect("Failed to create temp mesh state")
}

// Property testing for CRDT properties
proptest! {
    #[test]
    fn test_merge_associativity(
        h1 in proptest::collection::vec(heartbeat_strategy(), 0..5),
        h2 in proptest::collection::vec(heartbeat_strategy(), 0..5),
        h3 in proptest::collection::vec(heartbeat_strategy(), 0..5)
    ) {
        // (A + B) + C == A + (B + C)

        // Setup states
        let state_abc = create_temp_mesh_state();
        state_abc.merge(h1.clone()).unwrap();
        state_abc.merge(h2.clone()).unwrap();
        state_abc.merge(h3.clone()).unwrap();

        let state_a_bc = create_temp_mesh_state();
        state_a_bc.merge(h1.clone()).unwrap();

        let state_bc_temp = create_temp_mesh_state(); // helper to simulate B merged then C
        state_bc_temp.merge(h2.clone()).unwrap();
        state_bc_temp.merge(h3.clone()).unwrap();
        let all_bc = state_bc_temp.get_all(); // extract merged result to pass to A

        state_a_bc.merge(all_bc).unwrap();

        // Compare results
        let final_abc = state_abc.get_all();
        let final_a_bc = state_a_bc.get_all();

        // Sort by peer_id for deterministic comparison
        let mut sorted_abc = final_abc.clone();
        sorted_abc.sort_by_key(|h| h.peer_id.clone());

        let mut sorted_a_bc = final_a_bc.clone();
        sorted_a_bc.sort_by_key(|h| h.peer_id.clone());

        assert_eq!(sorted_abc.len(), sorted_a_bc.len());
        for (a, b) in sorted_abc.iter().zip(sorted_a_bc.iter()) {
            assert_eq!(a.peer_id, b.peer_id);
            assert_eq!(a.timestamp, b.timestamp);
        }
    }

    #[test]
    fn test_merge_commutativity(
        h1 in proptest::collection::vec(heartbeat_strategy(), 0..10),
        h2 in proptest::collection::vec(heartbeat_strategy(), 0..10)
    ) {
        // A + B == B + A

        let state_ab = create_temp_mesh_state();
        state_ab.merge(h1.clone()).unwrap();
        state_ab.merge(h2.clone()).unwrap();

        let state_ba = create_temp_mesh_state();
        state_ba.merge(h2.clone()).unwrap();
        state_ba.merge(h1.clone()).unwrap();

        let mut final_ab = state_ab.get_all();
        final_ab.sort_by_key(|h| h.peer_id.clone());

        let mut final_ba = state_ba.get_all();
        final_ba.sort_by_key(|h| h.peer_id.clone());

        assert_eq!(final_ab.len(), final_ba.len());
        for (a, b) in final_ab.iter().zip(final_ba.iter()) {
            assert_eq!(a.peer_id, b.peer_id);
            assert_eq!(a.timestamp, b.timestamp);
        }
    }

    #[test]
    fn test_merge_idempotency(
        h1 in proptest::collection::vec(heartbeat_strategy(), 0..10)
    ) {
        // A + A == A
        let state_aa = create_temp_mesh_state();
        state_aa.merge(h1.clone()).unwrap();
        state_aa.merge(h1.clone()).unwrap();

        let state_a = create_temp_mesh_state();
        state_a.merge(h1.clone()).unwrap();

        let mut final_aa = state_aa.get_all();
        final_aa.sort_by_key(|h| h.peer_id.clone());

        let mut final_a = state_a.get_all();
        final_a.sort_by_key(|h| h.peer_id.clone());

        assert_eq!(final_aa.len(), final_a.len());
        for (a, b) in final_aa.iter().zip(final_a.iter()) {
            assert_eq!(a.peer_id, b.peer_id);
            assert_eq!(a.timestamp, b.timestamp);
        }
    }
}

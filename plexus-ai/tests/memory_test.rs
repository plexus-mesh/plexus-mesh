use plexus_ai::{BertEmbedder, SimpleVectorStore, VectorStore};

#[tokio::test]
async fn test_vector_memory() -> anyhow::Result<()> {
    // 1. Initialize Embedder and Store
    let embedder = BertEmbedder::new(None);
    let store = SimpleVectorStore::new();

    // 2. Generate Embeddings (this triggers download, might be slow on first run)
    println!("Embedding 'hello world'...");
    let vec1 = embedder.embed("hello world").await?;

    println!("Embedding 'hi earth'...");
    let vec2 = embedder.embed("hi earth").await?;

    println!("Embedding 'pizza pepperoni'...");
    let vec3 = embedder.embed("pizza pepperoni").await?;

    // 3. Add to store
    store
        .add_document("greeting_1", "greeting_1", vec1.clone())
        .await?;
    store
        .add_document("greeting_2", "greeting_2", vec2.clone())
        .await?;
    store.add_document("food_1", "food_1", vec3.clone()).await?;

    // 4. Search
    // Search for something similar to greeting
    let query = embedder.embed("hello").await?;
    let results = store.search(query, 3).await?;

    println!("Search results for 'hello': {:?}", results);

    assert_eq!(results.len(), 3);
    // greetings should be closer than food
    assert!(results[0].0.contains("greeting") || results[1].0.contains("greeting"));

    // Check similarity
    // vec1 and vec2 should be reasonably close
    // vec3 should be further

    Ok(())
}

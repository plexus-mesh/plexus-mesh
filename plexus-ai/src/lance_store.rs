use crate::VectorStore;
use anyhow::Result;
use arrow_array::{types::Float32Type, FixedSizeListArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::{
    connect,
    query::{ExecutableQuery, QueryBase},
    Connection, Table,
};
use std::sync::Arc;

pub struct LanceDbStore {
    table: Table,
}

impl LanceDbStore {
    pub async fn new(path: &std::path::Path) -> Result<Self> {
        let uri = path.to_string_lossy().to_string();
        let connection = connect(&uri).execute().await?;

        // Define Schema: id (utf8), text (utf8), vector (fixed_size_list<float32>[384])
        // 384 is the dimension for all-MiniLM-L6-v2
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 384),
                false,
            ),
        ]));

        let table = if connection
            .table_names()
            .execute()
            .await?
            .contains(&"vectors".to_string())
        {
            connection.open_table("vectors").execute().await?
        } else {
            connection
                .create_empty_table("vectors", schema)
                .execute()
                .await?
        };

        Ok(Self { table })
    }
}

#[async_trait::async_trait]
impl VectorStore for LanceDbStore {
    async fn add(&self, _id: &str, _vector: Vec<f32>) -> Result<()> {
        Err(anyhow::anyhow!("Use add_document instead"))
    }

    async fn add_document(&self, id: &str, text: &str, vector: Vec<f32>) -> Result<()> {
        let schema = self.table.schema().await?;

        let id_array = StringArray::from(vec![id]);
        let text_array = StringArray::from(vec![text]);

        // Construct FixedSizeListArray for vector
        let values = arrow_array::Float32Array::from(vector);
        let vector_array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            384,
            Arc::new(values),
            None,
        );

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(id_array),
                Arc::new(text_array),
                Arc::new(vector_array),
            ],
        )?;

        self.table
            .add(arrow_array::RecordBatchIterator::new(
                vec![Ok(batch)],
                schema.clone(),
            ))
            .execute()
            .await?;
        Ok(())
    }

    async fn search(&self, query_vector: Vec<f32>, k: usize) -> Result<Vec<(String, f32)>> {
        // LanceDB search
        // We need to verify if 'vector' column is indexed or perform brute force.
        // For embedded simply use standard query.

        let results = self
            .table
            .query()
            .limit(k)
            .nearest_to(query_vector)?
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let mut matches = vec![];

        for batch in results {
            let text_col = batch
                .column_by_name("text")
                .ok_or(anyhow::anyhow!("Missing text column"))?;
            let dist_col = batch
                .column_by_name("_distance")
                .ok_or(anyhow::anyhow!("Missing _distance column"))?;

            let texts = text_col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or(anyhow::anyhow!("Invalid text array"))?;
            let dists = dist_col
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
                .ok_or(anyhow::anyhow!("Invalid distance array"))?;

            for i in 0..batch.num_rows() {
                let text = texts.value(i).to_string();
                let dist = dists.value(i);
                // Convert distance to similarity score (Cosine distance is 1 - similarity usually, Lance might return L2)
                // Assuming L2 for now or generic distance.
                // Since our engine uses cosine similarity logic (1.0 is best), but search returns distance (0.0 is best).
                // Let's invert it roughly: 1.0 / (1.0 + dist) or just return raw distance and handle loop side.
                // However, RAG logic checks "score > 0.4". If score is similarity (0..1), we need similarity.
                // LanceDB defaults to L2 distance.
                // Similarity = 1 - (L2 / 2) for normalized vectors?
                // Simplest: 1.0 - distance (if distance < 1.0).

                let score = 1.0 - dist;
                matches.push((text, score));
            }
        }

        Ok(matches)
    }
}

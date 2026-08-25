//! Minimal Node binding over the `qdrant-edge` crate.
//!
//! Exposes exactly the three operations the benchmark needs: build an index
//! from packed vectors + metadata, query, and filtered query. Filters are
//! constructed from Qdrant's own JSON filter syntax, which keeps this binding
//! independent of the crate's internal type layout.

use std::path::Path;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use qdrant_edge::*;
use serde_json::json;

fn err<E: std::fmt::Display>(e: E) -> Error { Error::from_reason(e.to_string()) }

#[napi]
pub struct QeIndex { shard: EdgeShard, dim: usize }

#[napi]
impl QeIndex {
  /// Build a shard from packed f32 vectors (`n * dim` little-endian floats).
  #[napi(factory)]
  pub fn build(
    dir: String, vectors: Buffer, ids: Vec<i64>,
    folders: Vec<String>, mtimes: Vec<i64>, dim: u32,
  ) -> Result<QeIndex> {
    let dim = dim as usize;
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(err)?;

    let params = EdgeVectorParamsBuilder::new(dim, Distance::Cosine).build();
    let config = EdgeConfigBuilder::new().vector(DEFAULT_VECTOR_NAME, params).build();
    let shard = EdgeShard::new(Path::new(&dir), config).map_err(err)?;

    let bytes: &[u8] = &vectors;
    let n = ids.len();
    if bytes.len() < n * dim * 4 {
      return Err(Error::from_reason("vector buffer too small for ids/dim"));
    }

    // Payload indexes must exist for filters to be resolved by index rather
    // than by full payload scan.
    shard.update(UpdateOperation::FieldIndexOperation(FieldIndexOperations::CreateIndex(
      CreateIndex { field_name: "folder".parse().map_err(|_| err("bad path"))?,
                    field_schema: Some(PayloadFieldSchema::FieldType(PayloadSchemaType::Keyword)) }))).map_err(err)?;
    shard.update(UpdateOperation::FieldIndexOperation(FieldIndexOperations::CreateIndex(
      CreateIndex { field_name: "mtime".parse().map_err(|_| err("bad path"))?,
                    field_schema: Some(PayloadFieldSchema::FieldType(PayloadSchemaType::Integer)) }))).map_err(err)?;

    const BATCH: usize = 2048;
    let mut i = 0usize;
    while i < n {
      let hi = (i + BATCH).min(n);
      let mut pts = Vec::with_capacity(hi - i);
      for j in i..hi {
        let off = j * dim * 4;
        let mut v = Vec::with_capacity(dim);
        for d in 0..dim {
          let b = &bytes[off + d * 4..off + d * 4 + 4];
          v.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        }
        pts.push(PointStruct::new(
          ids[j] as u64, v,
          json!({ "folder": folders[j], "mtime": mtimes[j] }),
        ).0);
      }
      shard.update(UpdateOperation::PointOperation(
        PointOperations::UpsertPoints(PointInsertOperations::PointsList(pts)))).map_err(err)?;
      i = hi;
    }
    Ok(QeIndex { shard, dim })
  }

  /// Run the optimizer, which is what actually builds the HNSW index.
  #[napi]
  pub fn optimize(&self) -> Result<bool> {
    let mut any = false;
    for _ in 0..32 {
      match self.shard.optimize().map_err(err)? { true => any = true, false => break }
    }
    Ok(any)
  }

  #[napi]
  pub fn search(&self, q: Buffer, k: u32) -> Result<Vec<i64>> { self.run(q, k, None) }

  #[napi]
  pub fn search_filtered(&self, q: Buffer, k: u32, folders: Vec<String>, mtime_from: i64) -> Result<Vec<i64>> {
    let mut must = vec![];
    if !folders.is_empty() { must.push(json!({"key":"folder","match":{"any": folders}})); }
    if mtime_from > 0 { must.push(json!({"key":"mtime","range":{"gte": mtime_from}})); }
    let f: Filter = serde_json::from_value(json!({"must": must})).map_err(err)?;
    self.run(q, k, Some(f))
  }

  fn run(&self, q: Buffer, k: u32, filter: Option<Filter>) -> Result<Vec<i64>> {
    let b: &[u8] = &q;
    let mut v = Vec::with_capacity(self.dim);
    for d in 0..self.dim {
      let c = &b[d * 4..d * 4 + 4];
      v.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    let req = SearchRequest {
      query: QueryEnum::Nearest(NamedQuery::default_dense(v)),
      filter, params: None, limit: k as usize, offset: 0,
      with_payload: None, with_vector: None, score_threshold: None,
    };
    Ok(self.shard.search(req).map_err(err)?.into_iter().map(|p| match p.id {
      PointId::NumId(n) => n as i64,
      PointId::Uuid(_) => -1,
    }).collect())
  }
}

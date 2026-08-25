# vaultbench

Does the choice of embedded vector store matter at the corpus sizes a personal knowledge base actually reaches?

Obsidian semantic-search plugins generally reach for a JavaScript-native store: Orama, `sql.js`, or a plain `embeddings.json`. This measures whether that choice costs anything at 1k, 10k and 100k chunks, with recall numbers alongside the latency figures so the latter can be interpreted.

The deliverable is [`FINDINGS.md`](FINDINGS.md). This repository is the apparatus.

## Engines

| Engine | Install path | Exact? |
|---|---|---|
| brute force | none, a flat `Float32Array` | exact, and defines ground truth |
| Orama | `npm i @orama/orama` | exact |
| sqlite-vec | `npm i sqlite-vec better-sqlite3` | exact |
| LanceDB | `npm i @lancedb/lancedb` | approximate (IVF_PQ); also run with `refineFactor(10)` and with `IVF_FLAT` |
| Qdrant Edge | `cargo` plus the local napi binding in `engines-native/qe-napi` | approximate (HNSW) above its indexing threshold, exact below it |

## Reproducing

```bash
npm install
python -m pip install numpy onnxruntime transformers tokenizers huggingface_hub

# 0. embedding model (384-dim MiniLM, ONNX export)
python scripts/00_model.py

# 1. corpus: four public doc repos at pinned commits, plus real git mtimes
bash scripts/00_fetch_corpus.sh

# 2. chunk once; every engine consumes the identical chunks
python scripts/01_chunk.py

# 3. embed once; every engine loads the identical vectors
python scripts/02_embed.py chunks
python scripts/03_queries.py && python scripts/02_embed.py queries

# 4. calibrate filters, build the metadata sidecar, compute exact truth
python scripts/04_filters.py
python scripts/04b_meta.py
python scripts/05_groundtruth.py

# 5. build the Qdrant Edge binding (.so on Linux, .dylib on macOS)
cd engines-native/qe-napi
cargo build --release && cp target/release/qe_napi.dll qe_napi.node
cd ../..

# 6. run everything, then aggregate
bash scripts/run_all.sh
python scripts/06_aggregate.py && python scripts/08_tables.py
```

## Layout

```
scripts/                  corpus build, chunking, embedding, ground truth, aggregation
bench/                    harness (run.js) and one adapter per engine under engines/
engines-native/qe-napi/   napi-rs binding over the published qdrant-edge crate
data/                     chunks, vectors, filters, ground truth (generated)
results/                  raw per-run JSON plus summary.json
```

## Method notes

Embeddings are computed once and loaded byte-identically into every engine, so no measurement includes embedding cost.

Chunks are 256 WordPiece tokens at stride 224. This is not the conventional 512, because `all-MiniLM-L6-v2` has a real context of 256 and 512-token chunks would be truncated in half before embedding.

Ground truth is exact brute force over the same vectors, recomputed for each corpus size and each filter.

Filters are `folder IN (...) AND mtime >= T`, which pairs a single-valued category with a date range. That shape is expressible in all five engines. The selectivities were measured against the real corpus distribution and are reported as measured.

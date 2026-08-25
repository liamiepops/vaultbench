# vaultbench

Does the choice of embedded vector store matter at the corpus sizes a personal
knowledge base actually reaches?

Every Obsidian semantic-search plugin reaches for a JavaScript-native store —
Orama, `sql.js`, or a plain `embeddings.json`. This measures whether that
choice costs anything at 1k / 10k / 100k chunks, with recall numbers so the
latency figures mean something.

**The deliverable is [`FINDINGS.md`](FINDINGS.md).** This repo is the apparatus.

## Engines

| Engine | Install path | Exact? |
|---|---|---|
| brute force | none — a flat `Float32Array` | exact (defines ground truth) |
| Orama | `npm i @orama/orama` | exact |
| sqlite-vec | `npm i sqlite-vec better-sqlite3` | exact |
| LanceDB | `npm i @lancedb/lancedb` | approximate (IVF_PQ); also run with `refineFactor(10)` and `IVF_FLAT` |
| Qdrant Edge | `cargo` + local napi binding (`engines-native/qe-napi`) | approximate (HNSW) above its indexing threshold; exact below |

## Reproducing

```bash
npm install
python -m pip install numpy onnxruntime transformers tokenizers huggingface_hub

# 0. embedding model (384-dim MiniLM, ONNX)
python -c "from huggingface_hub import snapshot_download;   snapshot_download('sentence-transformers/all-MiniLM-L6-v2',   allow_patterns=['onnx/model.onnx','*.json','*.txt'],   local_dir='models/all-MiniLM-L6-v2')"

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

# 5. build the Qdrant Edge binding
cd engines-native/qe-napi && cargo build --release   && cp target/release/qe_napi.dll qe_napi.node && cd ../..
#   (.so on Linux, .dylib on macOS)

# 6. run everything, then aggregate
bash scripts/run_all.sh
python scripts/06_aggregate.py && python scripts/08_tables.py
```

## Layout

```
scripts/     corpus build, chunking, embedding, ground truth, aggregation
bench/       harness (run.js) and one adapter per engine under engines/
engines-native/qe-napi/   napi-rs binding over the published qdrant-edge crate
data/        chunks, vectors, filters, ground truth (generated)
results/     raw per-run JSON + summary.json
```

## Method notes

- **Embeddings are computed once** and loaded byte-identically into every
  engine. No measurement includes embedding cost.
- **Chunks are 256 WordPiece tokens, stride 224.** Not the conventional 512:
  `all-MiniLM-L6-v2` has a real context of 256, so 512-token chunks would be
  silently truncated in half before embedding.
- **Ground truth is exact brute force** over the same vectors, recomputed per
  corpus size and per filter.
- **Filters are `folder IN (...) AND mtime >= T`** — a single-valued category
  plus a date range, because that is expressible in all five engines.
  Selectivities are measured against the real corpus, not assumed.

# vaultbench

Obsidian semantic-search plugins store their embeddings in a JavaScript-native vector store: Orama, `sql.js` with FTS5, or a plain `embeddings.json` at the vault root. Whether that choice costs anything at the corpus sizes a personal vault reaches does not appear to have been measured. vaultbench measures it.

Five embedded vector engines are run over an identical corpus of real markdown at 1,000, 10,000 and 100,000 chunks, using identical pre-computed embeddings. Every engine is scored for recall against exact brute force, so its latency can be read against how often it returns the right answer. Filtered search is measured separately at three filter selectivities, because "find notes like this one, tagged `work`, changed this year" is the query a vault plugin actually serves.

The full report is [`FINDINGS.md`](FINDINGS.md).

## Headline result

At 100,000 chunks, on the hardware described below. Latency is p50 over 200 queries, median of 3 runs. "Restrictive" is a filter matching 0.994% of the corpus.

| Engine | Query (ms) | Recall@10 | Filtered (ms) | Filtered recall | RSS (MB) |
|---|---:|---:|---:|---:|---:|
| brute force | 32.66 | exact | 0.71 | exact | 306 |
| Orama | 89.78 | exact | 0.72 | exact | 2200 |
| sqlite-vec | 133.81 | exact | 130.66 | exact | 339 |
| LanceDB (default) | 3.60 | 0.50 | 4.24 | 0.40 | 646 |
| LanceDB + `refineFactor(10)` | 5.97 | exact | 6.49 | exact | 645 |
| Qdrant Edge | 0.52 | exact | 0.20 | exact | 526 |

A plain `Float32Array` is faster than both Orama and sqlite-vec at this size, and returns exact results. At 1,000 and 10,000 chunks every engine answers in under 14 ms and the choice does not matter. LanceDB's default index returns half the correct results until `refineFactor` is set.

## Engines evaluated

| Engine | Version | Install | Index |
|---|---|---|---|
| brute force | n/a | none | flat scan of a `Float32Array`; defines ground truth |
| Orama | 3.1.18 | `npm i @orama/orama` | flat scan |
| sqlite-vec | 0.1.9 | `npm i sqlite-vec better-sqlite3` | flat scan in a `vec0` virtual table |
| LanceDB | 0.37.1 | `npm i @lancedb/lancedb` | IVF_PQ by default; also run with `refineFactor(10)` and with IVF_FLAT |
| Qdrant Edge | 0.7.2 | `cargo`, plus the napi binding in `engines-native/qe-napi` | HNSW above its indexing threshold, flat below it |

Qdrant Edge has no published Node package. The binding in `engines-native/qe-napi` is about 120 lines of napi-rs over the `qdrant-edge` crate, which is on crates.io under Apache-2.0. `FINDINGS.md` covers what that costs to build.

## Corpus

Real markdown from four public documentation repositories, pinned by commit. Chunks are 256 WordPiece tokens at stride 224.

| Source | Commit | Files | Tokens | Chunks |
|---|---|---:|---:|---:|
| mdn/content (`files/en-us`) | `6cee0131` | 13,117 | 9,411,622 | 45,763 |
| dotnet/docs (`docs`) | `414c7826` | 11,974 | 10,787,529 | 51,565 |
| kubernetes/website (`content/en`) | `5836bf49` | 1,814 | 4,421,367 | 20,266 |
| home-assistant.io (`source`) | `ee175d3e` | 3,734 | 10,535,140 | 48,115 |
| **Total** | | **30,639** | **35,155,658** | **165,709** |

The 165,709 chunks are shuffled with seed 42 and the first 100,000 kept, so the 1k and 10k corpora are nested prefixes of the 100k corpus. Each chunk carries a source path, tags, and the real last-modified date of its file taken from that repository's git history.

Embeddings are `all-MiniLM-L6-v2`, 384 dimensions, mean-pooled and L2-normalised, computed once under ONNX Runtime and loaded byte-identically into every engine.

## Install

```bash
npm install
python -m pip install numpy onnxruntime transformers tokenizers huggingface_hub
```

Qdrant Edge additionally needs a Rust toolchain. Everything else ships prebuilt binaries.

## Running

The corpus fetch clones four repositories and reads their full git history, so it takes a while. Embedding 100,000 chunks on CPU takes a few minutes.

```bash
python scripts/00_model.py           # embedding model
bash   scripts/00_fetch_corpus.sh    # corpora at pinned commits, plus git mtimes
python scripts/01_chunk.py           # one chunker, run once
python scripts/02_embed.py chunks    # one embedding pass, reused everywhere
python scripts/03_queries.py && python scripts/02_embed.py queries
python scripts/04_filters.py         # calibrate filter selectivities
python scripts/04b_meta.py
python scripts/05_groundtruth.py     # exact top-10 per size, per filter

cd engines-native/qe-napi            # .so on Linux, .dylib on macOS
cargo build --release && cp target/release/qe_napi.dll qe_napi.node
cd ../..

bash   scripts/run_all.sh            # every engine, every size, 3 repetitions
python scripts/06_aggregate.py && python scripts/08_tables.py
```

`run_all.sh` honours `ENGINES`, `SIZES` and `REPS` if you want a subset.

## Principles

**Embeddings are computed once.** If each engine embedded its own text, the benchmark would measure the embedding model instead of retrieval. Vectors are generated once, written to disk as raw float32, and loaded unchanged into every engine. Query vectors are cached at build time, so query embedding never appears in a measurement.

**One chunker, run once.** Every engine sees byte-identical chunks. Chunks are 256 tokens, not the more common 512, because `all-MiniLM-L6-v2` has a real context of 256, and a 512-token chunk would be cut in half before the model saw it.

**Real text, not generated.** Synthetic vectors have unrealistic distributions and flatter approximate indexes. Real modification dates matter for the same reason: random dates would be uncorrelated with content and would make date filtering easier than it is in a real vault.

**Ground truth is exact.** Brute force defines truth and is recomputed for every corpus size and every filter condition. The brute-force adapter reproduces it at recall 1.000, which acts as the correctness gate. An engine is not called faster without a recall figure beside it.

**Documented defaults.** No engine is tuned beyond its documented options, with one exception: LanceDB's default index scores 0.50 recall, which was treated as a broken default. Two documented remedies are measured and reported under their own names.

**One process per run.** Each engine runs in a fresh Node process so resident memory is attributable. Every configuration runs three times and the median is reported, with the spread in `results/summary.json`.

## Layout

```
scripts/                  corpus build, chunking, embedding, ground truth, aggregation
bench/run.js              harness: one engine, one size, one repetition
bench/engines/            one adapter per engine, each exposing build/query/queryFiltered
engines-native/qe-napi/   napi-rs binding over the qdrant-edge crate
data/                     chunks, vectors, filters, ground truth (generated)
results/                  raw per-run JSON, summary.json, tables.md
```

## Hardware

AMD Ryzen 7 5700G, 8 cores and 16 threads, 32 GB RAM, Windows 11 Pro N 10.0.26200, Node v24.17.0, Python 3.12.3, rustc 1.97.1. All runs were done in one session on an otherwise idle machine.

## Limitations

Results come from one machine, one embedding model at one dimensionality, and one chunking scheme. The workload is read-only: every index is built once and then queried, so nothing here says what incremental insert or delete costs, which is how a real vault is maintained. Recall is measured at k=10 only. Hybrid search combining BM25 with vectors is out of scope, and Orama and Qdrant Edge both ship full-text capability that goes unmeasured here. `FINDINGS.md` sets these out in full.

## Related work

[`photostructure/node-vector-bench`](https://github.com/photostructure/node-vector-bench) benchmarks sqlite-vec, USearch, LanceDB and DuckDB VSS through Node bindings from 1k to 2M vectors, using synthetic vectors from a Gaussian mixture. It is a more thorough performance harness than this one and covers larger scales. It does not cover Orama or Qdrant Edge, and it does not measure filtered search at controlled selectivities.

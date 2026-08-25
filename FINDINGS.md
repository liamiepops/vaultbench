# Embedded vector search at personal-vault scale

**Question.** At the corpus sizes a personal knowledge base actually reaches, does a serious vector engine beat a simple one, and where is the crossover?

**Status.** Every number below was measured on one machine in one session. Each engine consumed byte-identical vectors and identical chunks. Recall is measured against exact brute force, so the latency figures can be interpreted.

---

## 1. Summary

At 1,000 and 10,000 chunks the choice of engine makes no difference a user could perceive. Every engine answered in under 14 ms, and most in under 7 ms.

At 100,000 chunks the spread widens to more than two orders of magnitude, but the fast end is not where the reputations would put it. A plain `Float32Array` scanned in JavaScript returns exact results in 32.7 ms. sqlite-vec takes 133.8 ms for the same query and Orama takes 89.8 ms, both also exact. Only Qdrant Edge at 0.52 ms and LanceDB at 3.6 to 6.0 ms beat the array, and LanceDB's default index returns half the correct results until a documented option is switched on.

Filtered search separates the engines by whether they exploit selectivity. Brute force and Orama scan only the chunks that survive the filter, so a 1% filter costs roughly 1% of a full scan. At 100,000 chunks brute force answers a 1% filtered query in 0.71 ms, beaten only by Qdrant Edge at 0.20 ms. sqlite-vec takes 130.66 ms for that same query, which is what it costs with no filter applied. At looser filters and the largest corpus an index does pull ahead: LanceDB with refinement answers the 80% filter in 8.06 ms against brute force's 25.27 ms.

**For any vault up to about 100,000 chunks, a plain embeddings array holds up.** sqlite-vec was slower than the array it would replace at every size tested.

---

## 2. Method

### Corpus

Real markdown from four public documentation repositories, pinned by commit. Generated text was avoided deliberately, because its embedding distributions are unrealistic and flatter approximate indexes.

| source | commit | files | tokens | chunks |
|---|---|---:|---:|---:|
| [mdn/content](https://github.com/mdn/content) (`files/en-us`) | `6cee0131` | 13,117 | 9,411,622 | 45,763 |
| [dotnet/docs](https://github.com/dotnet/docs) (`docs`) | `414c7826` | 11,974 | 10,787,529 | 51,565 |
| [kubernetes/website](https://github.com/kubernetes/website) (`content/en`) | `5836bf49` | 1,814 | 4,421,367 | 20,266 |
| [home-assistant.io](https://github.com/home-assistant/home-assistant.io) (`source`) | `ee175d3e` | 3,734 | 10,535,140 | 48,115 |
| **total** | | **30,639** | **35,155,658** | **165,709** |

MDN on its own yields about 46k chunks, which is why four sources were combined. The 165,709 chunks were shuffled with seed 42 and the first 100,000 kept. The 1k and 10k corpora are therefore strict nested prefixes of the 100k corpus, and every measured filter selectivity holds at all three sizes.

Front matter is parsed. MDN and Liquid macros, fenced code blocks and raw HTML are stripped. Files with under 400 characters of prose are dropped.

### Chunking

Chunks are 256 WordPiece tokens with a stride of 224, giving 32 tokens of overlap. This departs from the conventional 512 for a specific reason: `all-MiniLM-L6-v2` declares `max_seq_length: 256`. A 512-token chunk would be truncated in half before embedding, and the benchmark would then be measuring retrieval over text the model never saw. One chunker was run once, and all engines consume the identical chunks.

### Embeddings

The model is `sentence-transformers/all-MiniLM-L6-v2`: 384 dimensions, mean pooling, L2-normalised, run under ONNX Runtime 1.29.0 on CPU.

Vectors were computed once, persisted as raw float32, and loaded byte-identically into every engine, so no measurement anywhere includes embedding cost. Because the vectors are normalised, cosine similarity reduces to a dot product.

The pipeline was validated against the published reference similarities for this model. `"That is a happy person"` scored 0.9429 against `"That is a very happy person"`, 0.6946 against `"That is a happy dog"`, and 0.2569 against `"Today is a sunny day"`.

### Queries

200 distinct queries, taken as document titles sampled deterministically with seed 42 from across all four corpora. These are short topical noun phrases, which is what people type into a vault search box. They were embedded once at build time with the same model and cached, so query embedding never enters a measurement.

### Metadata and filters

Each chunk carries a source path, a real modification date taken from the originating repository's git history via `git log`, and tags derived from its path.

Real dates matter here. Randomly assigned dates would be uncorrelated with content, which would make date filtering unrealistically easy. Real dates correlate with topic, because active areas of a documentation set get edited together. Dates span 2018-12-27 to 2026-08-25, and all 100,000 chunks carry one.

Filter predicates are `folder IN (...) AND mtime >= T`. A single-valued category was used in place of multi-valued tags because it is the richest predicate all five engines can express. sqlite-vec's `vec0` metadata columns have no array-contains operator, so a multi-valued tag filter would have meant denormalising the schema for one engine and leaving the others alone.

Selectivities were calibrated by exhaustive search over top-N folders crossed with date quantiles, evaluated against the real corpus distribution. The figures below are measured, not targets:

| filter | target | measured | folders | modified since | matches at 1k / 10k / 100k |
|---|---:|---:|---:|---|---|
| permissive | 80% | **80.003%** | 45 | 2025-01-24 | 813 / 8,001 / 80,003 |
| moderate | 20% | **20.004%** | 40 | 2026-06-01 | 219 / 1,964 / 20,004 |
| restrictive | 1% | **0.994%** | 21 | 2026-08-22 | 8 / 91 / 994 |

### Measurement

Ground truth is exact brute force, computed with a numpy dot product and recomputed for every corpus size and every filter condition. The harness brute-force engine reproduces it at recall 1.000 everywhere, which acts as the correctness gate.

Each run is a fresh Node process, so resident memory is attributable to a single engine. Every configuration was run three times and the median is reported. Rep-to-rep spread on the headline 100k p50 was 1.0% for brute force, 1.4% for sqlite-vec, 1.8% for Orama, 6.3% for LanceDB and 9.6% for Qdrant Edge. None of the conclusions below turn on differences that small.

Latency is p50 and p95 over all 200 queries after a 20-query warmup. A cold pass was measured separately and differed from warm by a few percent for every engine, so only warm figures are tabulated. The corpus contains 567 exact-duplicate vectors out of 100,000, which is 0.57%. That is too few for tie-breaking to place a meaningful ceiling on measured recall.

### Environment

AMD Ryzen 7 5700G with 8 cores and 16 threads, 32 GB RAM, Windows 11 Pro N 10.0.26200, Node v24.17.0, Python 3.12.3, rustc 1.97.1.

| package | version |
|---|---|
| `@orama/orama` | 3.1.18 |
| `sqlite-vec` | 0.1.9 |
| `better-sqlite3` | 13.0.3 |
| `@lancedb/lancedb` | 0.37.1 |
| `qdrant-edge` (crate) | 0.7.2 |

The brief named `orama`, which on npm is a stale 2.0.6. The live package is `@orama/orama`.

---

## 3. Results

### 1,000 chunks

| engine | build | p50 | p95 | recall@10 | RSS | on-disk |
|---|---:|---:|---:|---:|---:|---:|
| brute force | 0.0 s | 0.27 ms | 0.31 ms | exact | 211 MB | 1.6 MB |
| Orama | 0.0 s | 0.47 ms | 0.57 ms | exact | 246 MB | 9.6 MB |
| sqlite-vec | 0.0 s | 0.62 ms | 0.76 ms | exact | 222 MB | 1.7 MB |
| LanceDB (default IVF_PQ) | 0.2 s | 1.43 ms | 2.02 ms | 0.700 | 287 MB | 2.0 MB |
| LanceDB + refineFactor(10) | 0.2 s | 3.23 ms | 3.90 ms | exact | 289 MB | 2.0 MB |
| Qdrant Edge | 0.1 s | 0.05 ms | 0.06 ms | exact | 222 MB | 152.2 MB |

### 10,000 chunks

| engine | build | p50 | p95 | recall@10 | RSS | on-disk |
|---|---:|---:|---:|---:|---:|---:|
| brute force | 0.0 s | 3.28 ms | 3.69 ms | exact | 223 MB | 15.7 MB |
| Orama | 0.2 s | 6.24 ms | 7.14 ms | exact | 433 MB | 141.1 MB |
| sqlite-vec | 0.3 s | 13.60 ms | 14.39 ms | exact | 251 MB | 16.5 MB |
| LanceDB (default IVF_PQ) | 2.5 s | 1.70 ms | 2.20 ms | 0.500 | 429 MB | 16.2 MB |
| LanceDB + refineFactor(10) | 2.4 s | 3.74 ms | 4.41 ms | exact | 430 MB | 16.2 MB |
| Qdrant Edge | 0.9 s | 0.15 ms | 0.24 ms | exact | 296 MB | 193.7 MB |

### 100,000 chunks

| engine | build | p50 | p95 | recall@10 | RSS | on-disk |
|---|---:|---:|---:|---:|---:|---:|
| brute force | 0.0 s | 32.66 ms | 33.71 ms | exact | 306 MB | 156.9 MB |
| Orama | 1.9 s | 89.78 ms | 103.14 ms | exact | 2,200 MB | **cannot persist** |
| sqlite-vec | 2.6 s | 133.81 ms | 139.15 ms | exact | 339 MB | 161.0 MB |
| LanceDB (default IVF_PQ) | 26.0 s | 3.60 ms | 4.31 ms | 0.500 | 646 MB | 158.7 MB |
| LanceDB + refineFactor(10) | 26.0 s | 5.97 ms | 6.75 ms | exact | 645 MB | 158.7 MB |
| LanceDB (IVF_FLAT) | 5.4 s | 10.33 ms | 11.25 ms | exact | 1,397 MB | 309.5 MB |
| Qdrant Edge | 15.2 s | 0.52 ms | 0.68 ms | exact | 526 MB | 549.7 MB |

Recall is reported as "exact" where it is 1.000 to three decimal places. On-disk is everything the engine wrote into its own directory, measured after close so that write-ahead logs have been checkpointed. For the two in-memory engines it is what a plugin would have to persist: raw float32 plus metadata for brute force, and Orama's own `save()` output for Orama.

---

## 4. Filtered search

The realistic query is not a bare similarity search. It is closer to "find chunks similar to this, from notes in these folders, changed since some date".

#### permissive (~80.0% of corpus)

| engine | 1,000 p50 / recall | 10,000 p50 / recall | 100,000 p50 / recall |
|---|---:|---:|---:|
| brute force | 0.23 ms / exact | 2.49 ms / exact | 25.27 ms / exact |
| Orama | 0.47 ms / exact | 8.34 ms / exact | 121.35 ms / exact |
| sqlite-vec | 0.88 ms / exact | 16.12 ms / exact | 156.17 ms / exact |
| LanceDB (default IVF_PQ) | 1.80 ms / 0.700 | 2.15 ms / 0.500 | 5.89 ms / 0.500 |
| LanceDB + refineFactor(10) | 3.35 ms / exact | 4.42 ms / exact | 8.06 ms / exact |
| LanceDB (IVF_FLAT) | n/a | n/a | 11.42 ms / exact |
| Qdrant Edge | 0.21 ms / exact | 0.36 ms / exact | 1.25 ms / exact |

#### moderate (~20.0% of corpus)

| engine | 1,000 p50 / recall | 10,000 p50 / recall | 100,000 p50 / recall |
|---|---:|---:|---:|
| brute force | 0.07 ms / exact | 0.62 ms / exact | 9.88 ms / exact |
| Orama | 0.13 ms / exact | 1.39 ms / exact | 30.91 ms / exact |
| sqlite-vec | 0.70 ms / exact | 14.18 ms / exact | 138.74 ms / exact |
| LanceDB (default IVF_PQ) | 1.83 ms / 0.700 | 1.98 ms / 0.600 | 4.29 ms / 0.500 |
| LanceDB + refineFactor(10) | 3.34 ms / exact | 4.17 ms / exact | 6.45 ms / exact |
| LanceDB (IVF_FLAT) | n/a | n/a | 6.48 ms / exact |
| Qdrant Edge | 0.08 ms / exact | 0.32 ms / exact | 0.89 ms / exact |

#### restrictive (~1.0% of corpus)

| engine | 1,000 p50 / recall | 10,000 p50 / recall | 100,000 p50 / recall |
|---|---:|---:|---:|
| brute force | 0.01 ms / exact | 0.07 ms / exact | 0.71 ms / exact |
| Orama | 0.02 ms / exact | 0.06 ms / exact | 0.72 ms / exact |
| sqlite-vec | 0.60 ms / exact | 13.49 ms / exact | 130.66 ms / exact |
| LanceDB (default IVF_PQ) | 1.64 ms / exact | 1.81 ms / 0.600 | 4.24 ms / 0.400 |
| LanceDB + refineFactor(10) | 2.43 ms / exact | 3.77 ms / exact | 6.49 ms / exact |
| LanceDB (IVF_FLAT) | n/a | n/a | 3.44 ms / exact |
| Qdrant Edge | 0.04 ms / exact | 0.07 ms / exact | 0.20 ms / exact |

#### Results returned (k=10 requested)

Mean results returned at 100,000 chunks. A count below 10 where more than 10 chunks match the filter is the signature of search-then-discard. Those matches are lost, and not simply ranked lower.

| engine | permissive | moderate | restrictive |
|---|---:|---:|---:|
| brute force | 10.0 | 10.0 | 10.0 |
| Orama | 10.0 | 10.0 | 10.0 |
| sqlite-vec | 10.0 | 10.0 | 10.0 |
| LanceDB (default IVF_PQ) | 10.0 | 10.0 | 10.0 |
| LanceDB + refineFactor(10) | 10.0 | 10.0 | 10.0 |
| LanceDB (IVF_FLAT) | 10.0 | 10.0 | 10.0 |
| Qdrant Edge | 10.0 | 10.0 | 10.0 |

### Which strategy is each engine using?

The discriminating evidence is how latency responds to selectivity. An engine that filters first and scans the survivors should spend about 1% of a full scan on a 1% filter. An engine that searches the whole index and applies the predicate along the way should show no benefit from selectivity at all.

The table gives the ratio of restrictive-filter p50 to unfiltered p50, so a lower number means the filter is pruning work. The strategy column is inference from these ratios, not something read off documentation or source.

| engine | 1,000 | 10,000 | 100,000 | inferred strategy |
|---|---:|---:|---:|---|
| brute force | 0.028 | 0.020 | 0.022 | filter first, scan survivors |
| Orama | 0.035 | 0.010 | 0.008 | filter first, scan survivors |
| Qdrant Edge | 0.809 | 0.464 | 0.378 | index traversal with partial pruning |
| sqlite-vec | 0.971 | 0.992 | 0.976 | full scan, predicate applied inline |
| LanceDB (IVF_PQ) | 1.146 | 1.065 | 1.177 | full index search, no pruning benefit |

Brute force and Orama track selectivity almost exactly, which is what filter-then-scan predicts.

sqlite-vec's ratio sits near 0.98 at every size. The filter saves it nothing, and under the permissive setting the predicate makes queries slower than running with no filter at all: 156.17 ms against 133.81 ms at 100k.

LanceDB comes out worse than neutral. Its filtered queries consistently cost more than its unfiltered ones and gain nothing from selectivity.

Qdrant Edge lands between the two groups. That is consistent with a planner estimating cardinality and pruning partially, though the ratios alone cannot establish the mechanism, and I did not read its planner source to confirm it.

### The failure mode that did not appear

A naive implementation searches the index for k results and then discards the ones that fail the filter, which silently loses recall when the filter is selective. None of the five engines did this.

Every engine returned a full 10 results at every selectivity where 10 matches existed, and exactly 8 in the one configuration where only 8 chunks matched the filter, at 1k with the restrictive filter. Filtered recall was 1.000 for every engine at every size, with one exception: LanceDB's default index. That loss is quantisation error, since the same index loses the same recall on unfiltered queries.

This is a negative result worth recording. The failure predicted at 1% selectivity is absent from all five engines at the versions tested.

---

## 5. The crossover

There is no size inside the tested range at which unfiltered search requires a real index. There is only a size at which one becomes convenient.

Brute force scales linearly at 0.327 ms per 1,000 chunks: 0.271 ms at 1k, 3.285 ms at 10k, 32.665 ms at 100k. Extrapolating that line puts it at 50 ms around 153,000 chunks, 100 ms around 306,000, and 200 ms around 612,000.

A vault would need roughly 150,000 chunks before a plain array stopped feeling instant, and roughly 300,000 before it became irritating. The 100,000-chunk corpus used here is 35 million tokens of prose, which is already well past what most people accumulate.

Three places where a real index does pay:

- Qdrant Edge is faster at every size, by 5.5x at 1k and 62.5x at 100k, and stays exact throughout. Its costs are 15.2 s to build at 100k, 550 MB on disk, and a Rust toolchain.
- LanceDB with `refineFactor(10)` is 5.5x faster than brute force at 100k and exact, for 26 s of build time.
- Neither sqlite-vec nor Orama is faster than the plain array at any size above 1k.

The result that runs against expectation is that the two engines a plugin author is most likely to reach for are the two that lose to the array they would replace. Orama gets picked because it is JS-native and sqlite-vec because SQLite feels like the responsible option. At 100k Orama is 2.75x slower than brute force and holds 2.2 GB resident, and sqlite-vec is 4.10x slower.

sqlite-vec's result is surprising for a C extension, so I tested the obvious explanation instead of assuming one. It is not the distance metric. Rebuilding the table with `distance_metric=L2`, which gives equivalent ranking for normalised vectors, produced 139 ms against cosine's 152 ms in the same session. That is a marginal difference and not a fourfold one. I did not establish what does account for the gap.

### Where each engine stops being viable

**Orama cannot be persisted at 100k.** `JSON.stringify(save(db))` throws `RangeError: Invalid string length`, because the serialised index exceeds V8's maximum string length. Its latency and recall at 100k appear in the table because they were measured before persistence was attempted, but a plugin could not save this index without a binary persistence format. At 10k it does persist, to a 141 MB JSON file holding 15.7 MB of vectors.

**LanceDB's default index is lossy from the first size tested.** It reports recall 0.700 at 1k and 0.500 at both 10k and 100k. This is IVF_PQ product quantisation working as designed, but 50% recall is not a reasonable default for semantic search, so I treated it as pathological under the brief's carve-out and measured two documented remedies. `refineFactor(10)` restores recall to 1.000 for a 1.7x latency cost. `Index.ivfFlat()` also restores it, at 10.33 ms, 1.4 GB resident and 309 MB on disk. Both are one-line changes and neither is on by default.

**Qdrant Edge preallocates disk aggressively.** At 1,000 chunks it occupies 152 MB, dominated by two 32 MiB write-ahead-log segments plus 32 MiB preallocated payload and vector pages. `du` confirms these blocks are genuinely allocated and not sparse. At 100k it reaches 550 MB to hold 154 MB of vectors.

**Qdrant Edge stays exact below its indexing threshold.** At 1k it reported no HNSW index built, the default threshold being expressed in KB of segment size. It builds one at 10k and 100k. It returned recall 1.000 at both indexed sizes, so HNSW cost nothing in accuracy here at k=10.

---

## 6. Setup cost

For a plugin author choosing a store, installation is part of the comparison.

| engine | install | measured |
|---|---|---|
| brute force | none | 0 s, it is a `Float32Array` |
| Orama | `npm i @orama/orama` | part of the 78 s below |
| sqlite-vec | `npm i sqlite-vec better-sqlite3` | part of the 78 s below |
| LanceDB | `npm i @lancedb/lancedb` | part of the 78 s below |
| Qdrant Edge | `cargo` plus a hand-written napi binding | see below |

A cold-cache `npm install` of all four JavaScript engines together took 78 s and produced a prebuilt native binary for each. No compiler was needed.

sqlite-vec cost one non-obvious debugging session worth recording. `better-sqlite3` binds JavaScript numbers as SQLite `REAL`, and `vec0` strictly requires `INTEGER` for its primary key and integer metadata columns. Every id, every `k` and every integer filter bound has to be a `BigInt`, or inserts fail with `Only integers are allows for primary key values`. Nothing in either package's documentation connects those two facts.

### The Qdrant Edge caveat, revised

The brief states that Qdrant Edge cannot be driven from Node without building `uniffi-bindgen-react-native` from its uniffi 0.32 branch and using `@ubjs/core` from a local checkout, and that its numbers are therefore not independently reproducible.

For this benchmark's purposes that is no longer the constraint. `qdrant-edge` is published on crates.io at version 0.7.2 under Apache-2.0, and it fetches and builds with an ordinary `cargo add qdrant-edge`. The published `uniffi-bindgen-react-native` and `@ubjs/core` packages are indeed still at 0.31.0-5, so the brief was accurate about that path. It is no longer the only path to a Node binding.

What I did instead was write a napi-rs binding of about 120 lines in `engines-native/qe-napi`, exposing three operations. Filters are constructed from Qdrant's own JSON filter syntax instead of its internal Rust types, which keeps the binding independent of the crate's type layout.

Qdrant Edge's numbers here are therefore independently reproducible from published packages by anyone with a Rust toolchain. The remaining setup cost is real but ordinary:

- A clean `cargo build --release` of the binding took 500 s, about 8m20s, compiling 370 crates. Qdrant Edge vendors a large slice of the Qdrant engine. Dependency fetch beforehand took well under a minute. Against 78 s and no compiler for all four npm engines together, that is roughly a sixfold wait, paid once.
- Writing the binding took about 120 lines of Rust and required familiarity with napi-rs and with Qdrant's `EdgeShard`, `CollectionUpdateOperations` and `CoreSearchRequest` types. This is the real cost, and it differs in kind from `npm install`. It is engineering time, not waiting time.
- The binding was compiled with `lto = true`, which is not the default release profile. That setting favours Qdrant Edge and lengthens the build considerably. It is disclosed here for that reason.

One further asymmetry: the Qdrant Edge adapter receives vectors as a single packed buffer, while the JavaScript engines receive JavaScript arrays or objects because that is their documented interface. Its build times are a lower bound relative to the others.

---

## 7. What this does not tell you

- **One machine.** A single Ryzen 7 5700G with 32 GB of RAM running Windows 11. Relative ordering on a low-power ARM laptop, or on a machine where 2.2 GB resident triggers swapping, could differ. Orama's memory figure would be disqualifying on a smaller machine long before its latency was.
- **One embedding model at one dimensionality.** 384-dimensional MiniLM vectors. Approximate indexes behave differently at 768 or 1536 dimensions, and PQ compression that costs half the recall at 384 dimensions may cost less higher up.
- **One chunking scheme.** 256 tokens at stride 224. Chunk length affects the duplicate rate and the clustering structure of the embedding space, which is what approximate indexes are sensitive to.
- **Read-heavy with no concurrent writes.** Every index was built once and then queried. A vault is updated incrementally as notes change, and no engine here was tested for incremental insert or delete cost. This is the most likely place these conclusions break down. An engine that queries slowly but updates cheaply may still be the right choice.
- **k=10 only.** Recall@10 with `limit=10`. Approximate indexes generally degrade at larger k, so Qdrant Edge's perfect recall here should not be read as perfect recall at k=100.
- **Documented defaults, with one exception.** No engine was tuned beyond its documented options except LanceDB, where the default was treated as pathological and two documented remedies were measured and labelled as such.
- **No hybrid search.** BM25 combined with vectors was out of scope, and it is where several of these engines differentiate themselves. Orama and Qdrant Edge both ship full-text capability that went entirely unmeasured.

---

## 8. Prior art

`photostructure/node-vector-bench` benchmarks sqlite-vec, USearch, LanceDB and DuckDB VSS through Node bindings from 1k to 2M vectors. It is a more thorough performance harness than this one and covers larger scales.

It differs in three ways that matter for the question asked here. It uses synthetic vectors drawn from a Gaussian mixture in place of real embeddings. It does not cover Orama or Qdrant Edge. It does not measure filtered search at controlled selectivities, which is the case this report exists to examine.

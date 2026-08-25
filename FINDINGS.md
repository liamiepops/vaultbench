# Embedded vector search at personal-vault scale

2026-08-25

Obsidian semantic-search plugins store their embeddings in a JavaScript-native vector store: Orama, `sql.js` with FTS5, or a plain `embeddings.json` at the vault root. Whether that choice costs anything at the corpus sizes a personal vault reaches does not appear to have been measured. This report measures it, across five embedded engines at 1,000, 10,000 and 100,000 chunks of real markdown, scoring each engine for recall against exact brute force so its latency can be read against how often it returns the right answer.

The short version is that below 100,000 chunks the choice does not matter, and at 100,000 chunks a plain `Float32Array` beats two of the four real engines. Latency is p50 over 200 queries, median of three runs, at 100,000 chunks. The filtered column is a filter matching 0.994% of the corpus.

| Engine | Query (ms) | Recall@10 | Filtered (ms) | Filtered recall | RSS (MB) | On disk (MB) |
|---|---:|---:|---:|---:|---:|---:|
| brute force | 32.66 | exact | 0.71 | exact | 306 | 156.9 |
| Orama | 89.78 | exact | 0.72 | exact | 2200 | cannot persist |
| sqlite-vec | 133.81 | exact | 130.66 | exact | 339 | 161.0 |
| LanceDB (default) | 3.60 | 0.50 | 4.24 | 0.40 | 646 | 158.7 |
| LanceDB + `refineFactor(10)` | 5.97 | exact | 6.49 | exact | 645 | 158.7 |
| Qdrant Edge | 0.52 | exact | 0.20 | exact | 526 | 549.7 |

Four results carry the report. Brute force is faster than both Orama and sqlite-vec at 100,000 chunks and exact by construction. sqlite-vec gets no benefit at all from a filter, so a 1% filter costs it what an unfiltered query costs. LanceDB's default index returns half the correct results, and a documented one-line option restores them. Qdrant Edge is faster than everything at every size and stays exact, at the price of a Rust toolchain and 550 MB on disk.

---

## 1. Background

A plugin that offers semantic search over a vault has to keep a vector per chunk of text and find the nearest ones to a query. The engines available to it in Node fall into two groups. Some scan every vector and are exact. Others build an index and trade recall for speed.

The reason to measure this at vault scale specifically is that the published benchmarks for these engines target much larger corpora, where an index is obviously necessary. A vault is small. The question is whether the threshold at which an index starts to pay falls inside the range a person's notes actually reach, and if it does, which engines pay off there.

Filtered search is the case that motivated the work. A plugin rarely runs a bare similarity search. It runs something closer to "find chunks like this one, from notes tagged `work`, changed in the last year". There are three broad ways to serve that query, and they diverge sharply when the filter is selective. An engine can apply the filter first and scan the survivors. It can search the index and discard non-matching results afterwards, which quietly loses recall. It can traverse the index with the filter applied during the walk.

---

## 2. Test design

### Corpus

Real markdown from four public documentation repositories, pinned by commit. Generated text was avoided because its embedding distributions are unrealistic and flatter approximate indexes.

| Source | Commit | Files | Tokens | Chunks |
|---|---|---:|---:|---:|
| [mdn/content](https://github.com/mdn/content) (`files/en-us`) | `6cee0131` | 13,117 | 9,411,622 | 45,763 |
| [dotnet/docs](https://github.com/dotnet/docs) (`docs`) | `414c7826` | 11,974 | 10,787,529 | 51,565 |
| [kubernetes/website](https://github.com/kubernetes/website) (`content/en`) | `5836bf49` | 1,814 | 4,421,367 | 20,266 |
| [home-assistant.io](https://github.com/home-assistant/home-assistant.io) (`source`) | `ee175d3e` | 3,734 | 10,535,140 | 48,115 |
| **Total** | | **30,639** | **35,155,658** | **165,709** |

MDN on its own yields about 46,000 chunks, which is why four sources were combined. The 165,709 chunks were shuffled with seed 42 and the first 100,000 kept. The 1k and 10k corpora are nested prefixes of the 100k corpus, so every measured filter selectivity holds at all three sizes.

Front matter is parsed out. MDN and Liquid macros, fenced code blocks and raw HTML are stripped. Files with under 400 characters of prose are dropped.

### Chunking

Chunks are 256 WordPiece tokens at a stride of 224, giving 32 tokens of overlap. The conventional size is 512. It was not used here because `all-MiniLM-L6-v2` declares `max_seq_length: 256`, so a 512-token chunk would be cut in half before the model saw it, and the benchmark would then be scoring retrieval over text that was never embedded. One chunker was run once and all engines consume identical chunks.

### Embeddings

The model is `sentence-transformers/all-MiniLM-L6-v2`: 384 dimensions, mean pooling, L2-normalised, run under ONNX Runtime 1.29.0 on CPU.

Vectors were computed once, written to disk as raw float32, and loaded byte-identically into every engine. No measurement in this report includes embedding cost. Because the vectors are normalised, cosine similarity reduces to a dot product.

The pipeline was checked against the published reference similarities for this model before use. `"That is a happy person"` scored 0.9429 against `"That is a very happy person"`, 0.6946 against `"That is a happy dog"`, and 0.2569 against `"Today is a sunny day"`.

### Queries

200 distinct queries, taken as document titles sampled deterministically with seed 42 from across all four corpora. Titles are short topical noun phrases, which is what people type into a search box. They were embedded once at build time with the same model and cached, so query embedding never enters a measurement.

### Metadata and filters

Each chunk carries a source path, tags derived from that path, and the real last-modified date of its file, read from the originating repository's git history.

Real dates matter for the filtered results. Randomly assigned dates would be uncorrelated with content, which would make date filtering easier than it is in practice. Real dates correlate with topic, because active areas of a documentation set get edited together. Dates span 2018-12-27 to 2026-08-25 and all 100,000 chunks carry one.

Filter predicates take the form `folder IN (...) AND mtime >= T`. A single-valued category was used in place of multi-valued tags because it is the richest predicate all five engines can express. sqlite-vec's `vec0` metadata columns have no array-contains operator, so a multi-valued tag filter would have meant denormalising the schema for one engine and leaving the others alone.

Selectivities were calibrated by exhaustive search over top-N folders crossed with date quantiles, evaluated against the real corpus distribution. The figures below were measured after the fact, and are not targets:

| Filter | Target | Measured | Folders | Modified since | Matches at 1k / 10k / 100k |
|---|---:|---:|---:|---|---|
| permissive | 80% | 80.003% | 45 | 2025-01-24 | 813 / 8,001 / 80,003 |
| moderate | 20% | 20.004% | 40 | 2026-06-01 | 219 / 1,964 / 20,004 |
| restrictive | 1% | 0.994% | 21 | 2026-08-22 | 8 / 91 / 994 |

### Measurement

Ground truth is exact brute force computed with a numpy dot product, recomputed for every corpus size and every filter condition. The brute-force adapter in the harness reproduces it at recall 1.000 in every configuration, which is the correctness gate for the whole harness.

Each run happens in a fresh Node process so that resident memory belongs to one engine. Every configuration was run three times and the median is reported. Rep-to-rep spread on the 100k p50 was 1.0% for brute force, 1.4% for sqlite-vec, 1.8% for Orama, 6.3% for LanceDB and 9.6% for Qdrant Edge. No conclusion here turns on a difference that small.

Latency is p50 and p95 over all 200 queries after a 20-query warmup. A cold pass was measured separately and came within a few percent of warm for every engine, so only warm figures are tabulated.

The corpus holds 567 exact-duplicate vectors out of 100,000, which is 0.57%. That is too few for tie-breaking between identical vectors to place a meaningful ceiling on measured recall.

No engine was tuned beyond its documented options, with one exception described in section 5.2.

### Environment

AMD Ryzen 7 5700G, 8 cores and 16 threads, 32 GB RAM, Windows 11 Pro N 10.0.26200, Node v24.17.0, Python 3.12.3, rustc 1.97.1. All runs were done in one session on an otherwise idle machine.

| Package | Version |
|---|---|
| `@orama/orama` | 3.1.18 |
| `sqlite-vec` | 0.1.9 |
| `better-sqlite3` | 13.0.3 |
| `@lancedb/lancedb` | 0.37.1 |
| `qdrant-edge` (crate) | 0.7.2 |

The brief for this work named `orama`, which on npm is a stale 2.0.6. The live package is `@orama/orama`.

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

Recall reads "exact" where it is 1.000 to three decimal places. On-disk size is everything the engine wrote into its own directory, measured after close so that any write-ahead log has been checkpointed. For the two in-memory engines it is what a plugin would have to persist: raw float32 plus metadata for brute force, and Orama's own `save()` output for Orama.

Build time is wall clock and cold. A plugin pays it once, but the user waits for it.

---

## 4. Filtered search

### How each engine responds to selectivity

The discriminating evidence is what happens to latency when the filter gets tighter. An engine that applies the filter first and scans the survivors should spend about 1% of a full scan on a 1% filter. An engine that searches its whole index and applies the predicate along the way should show no benefit at all.

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

Taking the ratio of restrictive-filter p50 to unfiltered p50 makes the pattern easier to read. A low number means the filter is pruning work.

| Engine | 1,000 | 10,000 | 100,000 |
|---|---:|---:|---:|
| brute force | 0.028 | 0.020 | 0.022 |
| Orama | 0.035 | 0.010 | 0.008 |
| Qdrant Edge | 0.809 | 0.464 | 0.378 |
| sqlite-vec | 0.971 | 0.992 | 0.976 |
| LanceDB (default) | 1.146 | 1.065 | 1.177 |

Brute force and Orama track selectivity almost exactly, which is what filter-first predicts. A 1% filter costs them about 1% of the work.

sqlite-vec sits near 0.98 at every size. The filter buys it nothing. Under the permissive filter the predicate makes queries slower than running unfiltered, 156.17 ms against 133.81 ms at 100,000 chunks, so the cost of evaluating the predicate is visible while the saving is not.

LanceDB comes out worse than neutral at every size. Its filtered queries cost more than its unfiltered ones and gain nothing as the filter tightens.

Qdrant Edge falls between the two groups, pruning some work but well short of proportionally. That is consistent with a planner that estimates cardinality and chooses a strategy, though these ratios cannot establish the mechanism on their own. I did not read the planner source to confirm it, so treat the reading as inference from timings.

### The failure mode that did not appear

The naive implementation of a filtered vector search asks the index for k results and then drops the ones that fail the predicate. When the filter is selective this silently returns fewer than k. The missing entries are lost, and not simply demoted.

None of the five engines did this. Every engine returned a full 10 results at every selectivity where 10 matching chunks existed. In the one configuration where only 8 chunks matched, 1,000 chunks under the restrictive filter, every engine returned exactly 8.

Filtered recall was 1.000 for every engine at every size, with one exception. LanceDB's default index loses recall under filters, but it loses the same recall on unfiltered queries, so the cause is quantisation and not the filtering path.

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

This is a negative result and worth stating as one. The failure mode that motivated much of this test design is absent from all five engines at the versions tested.

---

## 5. Findings by engine

### 5.1 Orama cannot be persisted at 100,000 chunks

`JSON.stringify(save(db))` throws `RangeError: Invalid string length`. The serialised index exceeds V8's maximum string length.

Orama's latency and recall at 100,000 chunks appear in the results table because they were measured before persistence was attempted. A plugin could not ship this configuration without a binary persistence format, since the index cannot be written to disk by the obvious route.

At 10,000 chunks it does persist, to a 141 MB JSON file holding 15.7 MB of vectors.

Resident memory is the other constraint. Orama holds 2.2 GB at 100,000 chunks, against 306 MB for the equivalent flat array. On a machine with less headroom that figure would rule the engine out well before its latency did.

### 5.2 LanceDB's default index returns half the correct results

LanceDB scored recall 0.700 at 1,000 chunks and 0.500 at both 10,000 and 100,000. This is IVF_PQ product quantisation behaving as designed. The vectors are compressed to PQ codes and distances are computed against the compressed form.

A 50% recall floor is not a reasonable default for semantic search, so I treated it as a pathological default under the brief's carve-out and measured two documented remedies:

| Configuration | Build (s) | p50 (ms) | Recall@10 | RSS (MB) | On disk (MB) |
|---|---:|---:|---:|---:|---:|
| default IVF_PQ | 26.0 | 3.60 | 0.500 | 646 | 158.7 |
| `refineFactor(10)` | 26.0 | 5.97 | exact | 645 | 158.7 |
| `Index.ivfFlat()` | 5.4 | 10.33 | exact | 1397 | 309.5 |

`refineFactor(10)` re-ranks candidates against the uncompressed vectors and restores recall to 1.000 for a 1.66x latency cost. `Index.ivfFlat()` skips compression and also restores it, at higher latency, roughly double the resident memory and double the disk.

Both are one-line changes. Neither is on by default, and nothing in the query result signals that recall has been lost.

### 5.3 sqlite-vec is slower than a JavaScript loop, and a filter does not help

sqlite-vec was the slowest engine at 100,000 chunks, at 133.81 ms against 32.66 ms for a flat scan written in JavaScript. It is slower than brute force at every size above 1,000 chunks.

That is surprising for a C extension, so I tested the obvious explanation instead of assuming one. It is not the distance metric. Rebuilding the table with `distance_metric=L2`, which gives an equivalent ranking for normalised vectors, produced 139 ms against cosine's 152 ms in the same session. The difference is marginal and does not account for a fourfold gap. I did not establish what does.

The filtered behaviour compounds this. Because selectivity buys sqlite-vec nothing, its 1% filtered query costs 130.66 ms while brute force answers the same query in 0.71 ms. For a plugin serving filtered search over a large vault, this is the widest gap in the report.

### 5.4 Qdrant Edge preallocates about 150 MB whatever the corpus size

At 1,000 chunks a Qdrant Edge shard occupies 152 MB. The bulk is preallocation: two 32 MiB write-ahead-log segments, a 32 MiB payload storage page, and a 32 MiB vector storage chunk, with further 1 MB and 4 MB pages for the payload indexes.

`du` reports the same figure as the logical size, so these blocks are genuinely allocated on this NTFS volume and not sparse.

At 100,000 chunks the shard reaches 549.7 MB to hold 154 MB of vectors. For a plugin writing into a user's vault directory this is a real cost, and it is close to constant at the small end.

### 5.5 Qdrant Edge is exact below its indexing threshold

At 1,000 chunks Qdrant Edge reported that no HNSW index had been built. Its optimizer builds one only when a segment exceeds an indexing threshold expressed in KB, and 1,000 vectors of 384 dimensions falls under it. Queries at that size are a flat scan.

It builds an index at 10,000 and 100,000 chunks, and returned recall 1.000 at both. HNSW cost nothing in accuracy here at k=10, which stands out because both other indexed engines lost recall.

Its query latency of 0.52 ms at 100,000 chunks is the fastest measured, roughly 63 times faster than a flat scan in JavaScript.

---

## 6. Discussion

### 6.1 Where the crossover is

No corpus size inside the tested range makes an index necessary for unfiltered search. There is only a size at which one becomes convenient.

Brute force scales linearly at 0.327 ms per 1,000 chunks: 0.271 ms at 1,000, 3.285 ms at 10,000 and 32.665 ms at 100,000. Extrapolating that line puts it at 50 ms around 153,000 chunks, 100 ms around 306,000 and 200 ms around 612,000.

A vault would need roughly 150,000 chunks before a flat array stopped feeling instant, and roughly 300,000 before it became irritating. The 100,000-chunk corpus used here is 35 million tokens of prose, already past what most people accumulate.

Two engines do pay off inside the range. Qdrant Edge is faster at every size, by 5.5x at 1,000 chunks and 62.5x at 100,000, and stays exact throughout. LanceDB with `refineFactor(10)` is 5.5x faster than brute force at 100,000 chunks and also exact.

The other two do not. Orama is 2.75x slower than brute force at 100,000 chunks and sqlite-vec is 4.10x slower.

### 6.2 What this means for a plugin author

The engines most likely to be reached for are the two that lose to the array they would replace. Orama gets chosen because it is JS-native and needs no native module. sqlite-vec gets chosen because SQLite feels like the responsible option. Neither assumption survives measurement at these sizes.

For a vault under roughly 100,000 chunks, a plain float32 array is a defensible choice on every axis measured here. It is exact, it has the lowest resident memory of any engine tested, it needs no build step, and under a selective filter it is second only to Qdrant Edge.

Where a plugin needs to beat that, the ranking is clear. Qdrant Edge is the fastest option and stays exact, if the project can carry a Rust toolchain and 550 MB of shard. LanceDB is the fastest option that installs from npm, provided `refineFactor` is set. Shipping LanceDB on its defaults means shipping a search that returns half the right answers.

### 6.3 Setup cost

For anyone choosing a store, what it costs to install is part of the comparison, and the spread here is wide.

A cold-cache `npm install` of all four JavaScript engines together took 78 s and produced a prebuilt native binary for each. No compiler was needed.

sqlite-vec cost one non-obvious debugging session. `better-sqlite3` binds JavaScript numbers as SQLite `REAL`, and `vec0` requires `INTEGER` for its primary key and integer metadata columns. Every id, every `k` and every integer filter bound has to be a `BigInt`, or inserts fail with `Only integers are allows for primary key values`. Nothing in either package's documentation connects those two facts.

Qdrant Edge is the outlier. A clean `cargo build --release` of the binding took 500 s, about 8m20s, compiling 370 crates, because the crate vendors a large slice of the Qdrant engine. Dependency fetch beforehand took well under a minute. Against 78 s and no compiler for all four npm engines together, that is roughly a sixfold wait, paid once.

The larger cost is the binding itself: about 120 lines of Rust, needing familiarity with napi-rs and with Qdrant's `EdgeShard`, `CollectionUpdateOperations` and `CoreSearchRequest` types. That is engineering time and not waiting time, and it is the part a plugin author should weigh.

### 6.4 The Qdrant Edge reproducibility caveat

The brief for this work states that Qdrant Edge cannot be driven from Node without building `uniffi-bindgen-react-native` from its uniffi 0.32 branch and using `@ubjs/core` from a local checkout, and that its numbers are therefore not independently reproducible.

For the purposes of this benchmark that is no longer the binding constraint. `qdrant-edge` is published on crates.io at 0.7.2 under Apache-2.0 and builds from an ordinary `cargo add qdrant-edge`. The published `uniffi-bindgen-react-native` and `@ubjs/core` packages are still at 0.31.0-5, so the brief was accurate about that path. It is not the only path to a Node binding.

What this benchmark used instead is a napi-rs binding of about 120 lines in `engines-native/qe-napi`, exposing three operations. Filters are built from Qdrant's own JSON filter syntax instead of its internal Rust types, which keeps the binding independent of the crate's type layout.

The Qdrant Edge numbers in this report are therefore reproducible from published packages by anyone with a Rust toolchain.

Two caveats attach to them. The binding was compiled with `lto = true`, which is not the default release profile, and that setting favours Qdrant Edge while lengthening the build considerably. The adapter also receives vectors as a single packed buffer, while the JavaScript engines receive JavaScript arrays or objects because that is their documented interface, so Qdrant Edge's build times are a lower bound relative to the others.

---

## 7. What this does not tell you

**One machine.** A single Ryzen 7 5700G with 32 GB of RAM running Windows 11. Ordering could differ on a low-power ARM laptop, or on any machine where 2.2 GB resident starts swapping.

**One embedding model at one dimensionality.** 384-dimensional MiniLM vectors. Approximate indexes behave differently at 768 or 1536 dimensions, and quantisation that costs half the recall at 384 dimensions may cost less higher up.

**One chunking scheme.** 256 tokens at stride 224. Chunk length drives the duplicate rate and the clustering structure of the embedding space, which is what approximate indexes are sensitive to.

**A read-only workload.** Every index was built once and then queried. A vault is maintained incrementally as notes change, and nothing here measures incremental insert, delete or re-index cost. This is the most likely place these conclusions break down. An engine that queries slowly but updates cheaply may still be the right choice for a plugin.

**k=10 only.** Recall was measured at k=10 with `limit=10`. Approximate indexes generally degrade as k grows, so Qdrant Edge's perfect recall here should not be read as perfect recall at k=100.

**Documented defaults, with one exception.** Only LanceDB was tuned, and only after its default was judged pathological. Both remedies are reported under their own names.

**No hybrid search.** Combining BM25 with vectors was out of scope, and it is where several of these engines differentiate themselves. Orama and Qdrant Edge both ship full-text capability that went unmeasured.

---

## 8. Related work

[`photostructure/node-vector-bench`](https://github.com/photostructure/node-vector-bench) benchmarks sqlite-vec, USearch, LanceDB and DuckDB VSS through Node bindings from 1,000 to 2 million vectors. It is a more thorough performance harness than this one and covers larger scales.

It differs in three ways that matter for the question asked here. It uses synthetic vectors drawn from a Gaussian mixture in place of real embeddings. It does not cover Orama or Qdrant Edge. It does not measure filtered search at controlled selectivities, which is the case this report was built to examine.

---

## 9. Reproducing

The repository holds the corpus build script, the harness, one adapter per engine, and the raw per-run JSON behind every figure quoted above. `README.md` gives the commands in order.

Every number in this report can be regenerated from `results/*.json` with `python scripts/06_aggregate.py && python scripts/08_tables.py`.

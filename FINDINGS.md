# Embedded vector search at personal-vault scale

**Question.** At the corpus sizes a personal knowledge base actually reaches,
does a serious vector engine beat a simple one, and where is the crossover?

**Status.** All numbers below were measured on one machine in one session.
Every engine consumed byte-identical vectors and identical chunks. Recall is
measured against exact brute force, so the latency figures mean something.

---

## 1. Summary

At 1,000 and 10,000 chunks the choice of engine does not matter: every engine
answers in under 14 ms, and most in under 7 ms. Nobody will perceive the
difference. At 100,000 chunks the engines separate by more than two orders of
magnitude, but the separation does not favour the serious databases uniformly —
a plain `Float32Array` scanned in JavaScript answers in 32.7 ms with exact
results, while sqlite-vec takes 133.8 ms and Orama 89.8 ms, both also exact.
Only Qdrant Edge (0.52 ms) and LanceDB (3.6–6.0 ms) are genuinely faster, and
LanceDB's default index silently returns half the correct results until a
documented option is switched on.

For filtered search — the case worth building this for — what matters is
whether an engine exploits selectivity, and the plain array exploits it better
than any indexed engine because filtering first turns a restrictive query into
a 1% scan. At 100,000 chunks with a 1% filter it answers in 0.71 ms, beaten
only by Qdrant Edge's 0.20 ms; sqlite-vec takes 130.66 ms for that same query,
having gained nothing from the filter at all. Only at looser filters and the
largest corpus does an index pull ahead: LanceDB with refinement answers the
80% filter in 8.06 ms against brute force's 25.27 ms.

**The plugin authors who chose a plain embeddings array made a defensible
choice for any vault up to about 100,000 chunks.** The ones who reached for
sqlite-vec bought a slower search than the array they replaced.

---

## 2. Method

### Corpus

Real markdown from four public documentation repositories, pinned by commit.
Generated text was avoided deliberately: it has unrealistic embedding
distributions and flatters approximate indexes.

| source | commit | files | tokens | chunks |
|---|---|---:|---:|---:|
| [mdn/content](https://github.com/mdn/content) (`files/en-us`) | `6cee0131` | 13,117 | 9,411,622 | 45,763 |
| [dotnet/docs](https://github.com/dotnet/docs) (`docs`) | `414c7826` | 11,974 | 10,787,529 | 51,565 |
| [kubernetes/website](https://github.com/kubernetes/website) (`content/en`) | `5836bf49` | 1,814 | 4,421,367 | 20,266 |
| [home-assistant.io](https://github.com/home-assistant/home-assistant.io) (`source`) | `ee175d3e` | 3,734 | 10,535,140 | 48,115 |
| **total** | | **30,639** | **35,155,658** | **165,709** |

MDN alone yields only ~46k chunks, which is why four sources were combined. The
165,709 chunks were shuffled with seed 42 and the first 100,000 kept, so the
1k and 10k corpora are strict nested prefixes of the 100k corpus and every
measured filter selectivity holds at all three sizes.

Front matter is parsed, MDN/Liquid macros, fenced code blocks and raw HTML are
stripped, and files with under 400 characters of prose are dropped.

### Chunking

**256 WordPiece tokens, stride 224 (32-token overlap).** This departs from the
conventional 512 for a specific reason: `all-MiniLM-L6-v2` declares
`max_seq_length: 256`. A 512-token chunk would be silently truncated in half
before embedding, and the benchmark would be measuring retrieval over text the
model never saw. One chunker was run once; all engines consume the identical
chunks.

### Embeddings

`sentence-transformers/all-MiniLM-L6-v2` — 384 dimensions, mean pooling,
L2-normalised — run under ONNX Runtime 1.29.0 on CPU. **Computed once,
persisted as raw float32, and loaded byte-identically into every engine**, so
no measurement anywhere includes embedding cost. Because vectors are
normalised, cosine similarity is a plain dot product.

The pipeline was validated against the published reference similarities for
this model: `"That is a happy person"` scores 0.9429 against `"That is a very
happy person"`, 0.6946 against `"That is a happy dog"`, and 0.2569 against
`"Today is a sunny day"`.

### Queries

200 distinct queries, being document titles sampled deterministically (seed 42)
from across all four corpora — short topical noun phrases, which is what people
type into a vault search box. Embedded once at build time with the same model
and cached, so query embedding never enters a measurement.

### Metadata and filters

Each chunk carries a source path, a **real** modification date taken from the
originating repository's git history (`git log`, newest commit touching the
file), and tags derived from its path. Real dates matter: randomly assigned
dates would be uncorrelated with content and would make date filtering
unrealistically easy, whereas real dates correlate with topic. Dates span
2018-12-27 to 2026-08-25; all 100,000 chunks carry one.

Filter predicates are `folder IN (...) AND mtime >= T`. A single-valued
category was used rather than multi-valued tags because that is the richest
predicate all five engines can express — sqlite-vec's `vec0` metadata columns
have no array-contains operator, so multi-valued tag filters would have
required denormalising the schema for one engine and not the others.

Selectivities were calibrated by exhaustive search over (top-N folders ×
date quantile) against the real corpus distribution, so they are measured, not
assumed:

| filter | target | measured | folders | modified since | matches at 1k / 10k / 100k |
|---|---:|---:|---:|---|---|
| permissive | 80% | **80.003%** | 45 | 2025-01-24 | 813 / 8,001 / 80,003 |
| moderate | 20% | **20.004%** | 40 | 2026-06-01 | 219 / 1,964 / 20,004 |
| restrictive | 1% | **0.994%** | 21 | 2026-08-22 | 8 / 91 / 994 |

### Measurement

Ground truth is exact brute force (numpy dot product) recomputed for every
corpus size and every filter condition. Brute force in the harness reproduces
it at recall 1.000 everywhere, which is the correctness gate.

Each run is a fresh Node process, so resident memory is attributable to one
engine. Every configuration was run **3 times**; the median is reported.
Rep-to-rep spread on the headline 100k p50 was 1.0% (brute), 1.4%
(sqlite-vec), 1.8% (Orama), 6.3% (LanceDB) and 9.6% (Qdrant Edge) — small
enough that none of the conclusions turn on it.

Latency is p50/p95 over all 200 queries after a 20-query warmup. A cold pass
was measured separately; it differed from warm by only a few percent for every
engine, so only warm figures are tabulated. The corpus contains 567
exact-duplicate vectors out of 100,000 (0.57%), too few for tie-breaking to
place a meaningful ceiling on measured recall.

### Environment

AMD Ryzen 7 5700G (8 cores / 16 threads), 32 GB RAM, Windows 11 Pro N
10.0.26200, Node v24.17.0, Python 3.12.3, rustc 1.97.1.

| package | version |
|---|---|
| `@orama/orama` | 3.1.18 |
| `sqlite-vec` | 0.1.9 |
| `better-sqlite3` | 13.0.3 |
| `@lancedb/lancedb` | 0.37.1 |
| `qdrant-edge` (crate) | 0.7.2 |

Note that the spec's candidate table named `orama`, which on npm is a stale
2.0.6. The live package is `@orama/orama`.

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

Recall is reported as "exact" where it is 1.000 to three decimal places.
"On-disk" is everything the engine wrote into its own directory, measured
after close so that write-ahead logs have been checkpointed. For the two
in-memory engines it is what a plugin would have to persist: raw float32
plus metadata for brute force, and Orama's own `save()` output for Orama.

---

## 4. Filtered search

This is the case worth building the apparatus for. The realistic query is not
"find similar chunks" but "find chunks similar to this, from notes in these
folders, modified since some date".

#### permissive (~80.0% of corpus)

| engine | 1,000 p50 / recall | 10,000 p50 / recall | 100,000 p50 / recall |
|---|---:|---:|---:|
| brute force | 0.23 ms / exact | 2.49 ms / exact | 25.27 ms / exact |
| Orama | 0.47 ms / exact | 8.34 ms / exact | 121.35 ms / exact |
| sqlite-vec | 0.88 ms / exact | 16.12 ms / exact | 156.17 ms / exact |
| LanceDB (default IVF_PQ) | 1.80 ms / 0.700 | 2.15 ms / 0.500 | 5.89 ms / 0.500 |
| LanceDB + refineFactor(10) | 3.35 ms / exact | 4.42 ms / exact | 8.06 ms / exact |
| LanceDB (IVF_FLAT) | — | — | 11.42 ms / exact |
| Qdrant Edge | 0.21 ms / exact | 0.36 ms / exact | 1.25 ms / exact |

#### moderate (~20.0% of corpus)

| engine | 1,000 p50 / recall | 10,000 p50 / recall | 100,000 p50 / recall |
|---|---:|---:|---:|
| brute force | 0.07 ms / exact | 0.62 ms / exact | 9.88 ms / exact |
| Orama | 0.13 ms / exact | 1.39 ms / exact | 30.91 ms / exact |
| sqlite-vec | 0.70 ms / exact | 14.18 ms / exact | 138.74 ms / exact |
| LanceDB (default IVF_PQ) | 1.83 ms / 0.700 | 1.98 ms / 0.600 | 4.29 ms / 0.500 |
| LanceDB + refineFactor(10) | 3.34 ms / exact | 4.17 ms / exact | 6.45 ms / exact |
| LanceDB (IVF_FLAT) | — | — | 6.48 ms / exact |
| Qdrant Edge | 0.08 ms / exact | 0.32 ms / exact | 0.89 ms / exact |

#### restrictive (~1.0% of corpus)

| engine | 1,000 p50 / recall | 10,000 p50 / recall | 100,000 p50 / recall |
|---|---:|---:|---:|
| brute force | 0.01 ms / exact | 0.07 ms / exact | 0.71 ms / exact |
| Orama | 0.02 ms / exact | 0.06 ms / exact | 0.72 ms / exact |
| sqlite-vec | 0.60 ms / exact | 13.49 ms / exact | 130.66 ms / exact |
| LanceDB (default IVF_PQ) | 1.64 ms / exact | 1.81 ms / 0.600 | 4.24 ms / 0.400 |
| LanceDB + refineFactor(10) | 2.43 ms / exact | 3.77 ms / exact | 6.49 ms / exact |
| LanceDB (IVF_FLAT) | — | — | 3.44 ms / exact |
| Qdrant Edge | 0.04 ms / exact | 0.07 ms / exact | 0.20 ms / exact |

#### Results returned (k=10 requested)

A count below 10 at a selective filter is the signature of search-then-discard: matches are lost, not merely ranked lower.

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

The discriminating evidence is how latency responds to selectivity. If an
engine filters first and scans the survivors, a 1% filter should cost roughly
1% of an unfiltered scan. If it scans the whole index and applies the predicate
along the way, selectivity should not help at all.

Ratio of restrictive-filter p50 to unfiltered p50 (lower = filter is pruning work):

| engine | 1,000 | 10,000 | 100,000 | reading |
|---|---:|---:|---:|---|
| brute force | 0.028 | 0.020 | 0.022 | filter first, scan survivors |
| Orama | 0.035 | 0.010 | 0.008 | filter first, scan survivors |
| Qdrant Edge | 0.809 | 0.464 | 0.378 | index traversal, partial pruning |
| sqlite-vec | 0.971 | 0.992 | 0.976 | full scan, predicate applied inline |
| LanceDB (IVF_PQ) | 1.146 | 1.065 | 1.177 | full index search, no pruning benefit |

Brute force and Orama track selectivity almost exactly — a 1% filter costs
about 1% of the work, which is what filter-then-scan predicts. sqlite-vec's
ratio sits at ~0.98 across every size: the filter saves nothing, and at the
permissive setting the predicate makes queries *slower* than no filter at all
(156.17 ms vs 133.81 ms at 100k). LanceDB is worse than neutral — filtered
queries consistently cost more than unfiltered ones with no latency benefit
from selectivity. Qdrant Edge lands in between, consistent with a planner that
estimates cardinality and prunes partially rather than degenerating to either
extreme.

### The failure mode that did not appear

The classic naive implementation searches the index for k results and then
discards non-matching ones, which silently loses recall when the filter is
selective. **No engine tested did this.** Every engine returned a full 10
results at every selectivity where 10 matches existed, and exactly 8 at the
one configuration where only 8 chunks matched the filter (1k / restrictive).
Filtered recall was 1.000 for every engine at every size, with the sole
exception of LanceDB's default index — and that loss is quantisation error,
not a filtering bug, since it loses the same recall on unfiltered queries.

This is a real negative result: the failure the spec anticipated at 1%
selectivity is not present in any of these five engines as of the versions
tested.

---

## 5. The crossover

**For unfiltered search there is no crossover inside the tested range at which
a real index becomes necessary** — only one at which it becomes nice to have.

Brute force scales precisely linearly at **0.327 ms per 1,000 chunks**
(0.271 ms at 1k, 3.285 ms at 10k, 32.665 ms at 100k). Extrapolating that line:

- 50 ms at ~153,000 chunks
- 100 ms at ~306,000 chunks
- 200 ms at ~612,000 chunks

A vault would need roughly 150,000 chunks before a plain array stopped feeling
instant, and roughly 300,000 before it became annoying. For reference, the
100,000-chunk corpus here is 35 million tokens of prose — far beyond what most
people have.

Where a real index does pay:

- **Qdrant Edge is faster at every size**, by 5x at 1k and 63x at 100k, while
  staying exact. If the goal is the lowest achievable latency, it wins
  outright. It costs 15.2 s to build at 100k, 550 MB on disk, and a Rust
  toolchain.
- **LanceDB with `refineFactor(10)`** is 5.5x faster than brute force at 100k
  and exact, at 26 s build time.
- **sqlite-vec and Orama are both slower than the plain array at every size
  above 1k**, while offering nothing brute force does not.

The counterintuitive result is that the two engines a plugin author is most
likely to reach for — Orama because it is JS-native, sqlite-vec because SQLite
feels like the responsible choice — are the two that lose to the array they
would replace. Orama at 100k is 2.7x slower than brute force and holds 2.2 GB
resident. sqlite-vec at 100k is 4.1x slower.

sqlite-vec's result is worth stating plainly because it is surprising for a C
extension. It is not an artifact of the distance metric: rebuilding the table
with `distance_metric=L2` (equivalent ranking for normalised vectors) gave
139 ms against cosine's 152 ms in the same session — a marginal difference, not
a fourfold one.

### Where each engine stops being viable

- **Orama cannot be persisted at 100k.** `JSON.stringify(save(db))` throws
  `RangeError: Invalid string length` — the serialised index exceeds V8's
  maximum string length. Its latency and recall at 100k are in the table
  because they were measured before persistence was attempted, but a plugin
  could not save this index without a binary persistence format. At 10k it
  persists, but to a 141 MB JSON file for 15.7 MB of vectors.
- **LanceDB's default index is lossy from the very first size tested.** It
  reports recall 0.700 at 1k and 0.500 at 10k and 100k. This is IVF_PQ product
  quantisation working as designed, but 50% recall is not a reasonable default
  for semantic search, so it was treated as pathological under the spec's
  carve-out and remedied with documented options: `refineFactor(10)` restores
  recall to 1.000 for a 1.7x latency cost, and `Index.ivfFlat()` also restores
  it at 10.33 ms, 1.4 GB resident and 309 MB on disk. Both are one-line
  changes; neither is on by default.
- **Qdrant Edge preallocates disk aggressively.** At 1,000 chunks it occupies
  152 MB, dominated by two 32 MiB write-ahead-log segments and 32 MiB
  preallocated payload and vector pages. `du` confirms these are genuinely
  allocated, not sparse. At 100k it reaches 550 MB for 154 MB of vectors.
- **Qdrant Edge stays exact below its indexing threshold.** At 1k it reported
  no HNSW index built (the default threshold is expressed in KB of segment
  size); it builds one at 10k and 100k. Notably it returned recall 1.000 at
  both indexed sizes, so HNSW cost nothing in accuracy here at k=10.

---

## 6. Setup cost

For a plugin author choosing a store, installation is part of the comparison.

| engine | install | measured |
|---|---|---|
| brute force | none | 0 s — it is a `Float32Array` |
| Orama | `npm i @orama/orama` | part of the 78 s below |
| sqlite-vec | `npm i sqlite-vec better-sqlite3` | part of the 78 s below |
| LanceDB | `npm i @lancedb/lancedb` | part of the 78 s below |
| Qdrant Edge | `cargo` + a hand-written napi binding | see below |

A cold-cache `npm install` of all four JavaScript engines together took
**78 s** and produced a prebuilt native binary for each. No compiler was
needed.

sqlite-vec cost one non-obvious debugging session that is worth recording:
`better-sqlite3` binds JavaScript numbers as SQLite `REAL`, and `vec0` strictly
requires `INTEGER` for its primary key and integer metadata columns. Every id,
every `k`, and every integer filter bound has to be a `BigInt` or inserts fail
with `Only integers are allows for primary key values`. Nothing in either
package's documentation connects those two facts.

### The Qdrant Edge caveat, revised

**The spec's premise here is out of date, and the correction is favourable.**
Section 6 states that Qdrant Edge cannot be driven from Node without building
`uniffi-bindgen-react-native` from its uniffi 0.32 branch and using `@ubjs/core`
from a local checkout, and that its numbers are therefore not independently
reproducible.

That is no longer the situation. **`qdrant-edge` is published on crates.io**
(version 0.7.2, Apache-2.0) and fetches and builds with an ordinary
`cargo add qdrant-edge`. No uniffi, no branch builds, no local checkouts, no
private beta. The published `uniffi-bindgen-react-native` and `@ubjs/core`
packages are indeed still at 0.31.0-5, so the spec was accurate about *that*
path — but that path is no longer the only one.

What this benchmark did instead was write a ~120-line napi-rs binding
(`engines-native/qe-napi`) exposing exactly three operations. Filters are
constructed from Qdrant's own JSON filter syntax rather than its internal Rust
types, which keeps the binding independent of the crate's type layout.

**Qdrant Edge's numbers here are therefore independently reproducible**, from
published packages, by anyone with a Rust toolchain. The remaining setup cost
is real but ordinary:

- A clean `cargo build --release` of the binding took **500 s** (8m20s) on
  this machine, compiling 370 crates — Qdrant Edge vendors a large slice of
  the Qdrant engine. Dependency fetch beforehand took well under a minute.
  Against 78 s and no compiler for all four npm engines combined, that is
  roughly a sixfold wait, once.
- Writing the binding: roughly 120 lines of Rust, requiring familiarity with
  napi-rs and with Qdrant's `EdgeShard` / `CollectionUpdateOperations` /
  `CoreSearchRequest` API. This is the genuine cost, and it is a different
  kind of cost from `npm install` — it is not a wait, it is engineering.
- The binding here was compiled with `lto = true`, which is not the default
  release profile. That is favourable to Qdrant Edge and is disclosed here;
  it lengthens the build considerably.

One further caveat: the Qdrant Edge adapter receives vectors as a single packed
buffer, whereas the JavaScript engines receive JavaScript arrays or objects
because that is their documented interface. Its build times are therefore a
lower bound relative to the others.

---

## 7. What this does not tell you

- **One machine.** A single Ryzen 7 5700G with 32 GB of RAM running Windows 11.
  Relative ordering on a low-power ARM laptop, or on a machine where 2.2 GB of
  resident memory triggers swapping, could differ. Orama's memory figure in
  particular would be disqualifying on a smaller machine long before its
  latency was.
- **One embedding model, one dimensionality.** 384-dimensional MiniLM vectors.
  Approximate indexes behave differently at 768 or 1536 dimensions, and PQ
  compression ratios that lose half the recall at 384 dimensions may lose less
  at higher ones.
- **One chunking scheme.** 256 tokens, stride 224. Chunk length affects the
  duplicate rate and the clustering structure of the embedding space, which is
  exactly what approximate indexes are sensitive to.
- **Read-heavy, no concurrent writes.** Every index was built once and then
  queried. A vault is incrementally updated as notes change, and none of these
  engines was tested for incremental insert, delete, or re-index cost. That is
  a real workload difference and it is the most likely place these conclusions
  would break down — an engine that is slow to query but cheap to update
  incrementally may still be the right choice.
- **k=10 only.** Recall@10 with `limit=10`. Approximate indexes generally
  degrade at larger k, and Qdrant Edge's perfect recall here should not be read
  as perfect recall at k=100.
- **Documented defaults, mostly.** No engine was tuned beyond its documented
  options, except LanceDB, where the default was treated as pathological and
  two documented remedies were measured and labelled.
- **No hybrid search.** BM25 combined with vectors is out of scope and is where
  several of these engines differentiate themselves; Orama and Qdrant Edge both
  ship full-text capability that went entirely unmeasured here.

---

## 8. Prior art

`photostructure/node-vector-bench` benchmarks sqlite-vec, USearch, LanceDB and
DuckDB VSS with Node bindings from 1k to 2M vectors. It is a more thorough
performance harness than this one and covers larger scales. It differs in three
ways that matter for the question asked here: it uses **synthetic** vectors
from a Gaussian mixture rather than real embeddings, it does not cover Orama or
Qdrant Edge, and it does not measure filtered search at controlled
selectivities — which is the case this report exists to examine.

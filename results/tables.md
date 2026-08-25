
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

## Filtered search


### permissive (~80.0% of corpus)

| engine | 1,000 p50 / recall | 10,000 p50 / recall | 100,000 p50 / recall |
|---|---:|---:|---:|
| brute force | 0.23 ms / exact | 2.49 ms / exact | 25.27 ms / exact |
| Orama | 0.47 ms / exact | 8.34 ms / exact | 121.35 ms / exact |
| sqlite-vec | 0.88 ms / exact | 16.12 ms / exact | 156.17 ms / exact |
| LanceDB (default IVF_PQ) | 1.80 ms / 0.700 | 2.15 ms / 0.500 | 5.89 ms / 0.500 |
| LanceDB + refineFactor(10) | 3.35 ms / exact | 4.42 ms / exact | 8.06 ms / exact |
| LanceDB (IVF_FLAT) | — | — | 11.42 ms / exact |
| Qdrant Edge | 0.21 ms / exact | 0.36 ms / exact | 1.25 ms / exact |

### moderate (~20.0% of corpus)

| engine | 1,000 p50 / recall | 10,000 p50 / recall | 100,000 p50 / recall |
|---|---:|---:|---:|
| brute force | 0.07 ms / exact | 0.62 ms / exact | 9.88 ms / exact |
| Orama | 0.13 ms / exact | 1.39 ms / exact | 30.91 ms / exact |
| sqlite-vec | 0.70 ms / exact | 14.18 ms / exact | 138.74 ms / exact |
| LanceDB (default IVF_PQ) | 1.83 ms / 0.700 | 1.98 ms / 0.600 | 4.29 ms / 0.500 |
| LanceDB + refineFactor(10) | 3.34 ms / exact | 4.17 ms / exact | 6.45 ms / exact |
| LanceDB (IVF_FLAT) | — | — | 6.48 ms / exact |
| Qdrant Edge | 0.08 ms / exact | 0.32 ms / exact | 0.89 ms / exact |

### restrictive (~1.0% of corpus)

| engine | 1,000 p50 / recall | 10,000 p50 / recall | 100,000 p50 / recall |
|---|---:|---:|---:|
| brute force | 0.01 ms / exact | 0.07 ms / exact | 0.71 ms / exact |
| Orama | 0.02 ms / exact | 0.06 ms / exact | 0.72 ms / exact |
| sqlite-vec | 0.60 ms / exact | 13.49 ms / exact | 130.66 ms / exact |
| LanceDB (default IVF_PQ) | 1.64 ms / exact | 1.81 ms / 0.600 | 4.24 ms / 0.400 |
| LanceDB + refineFactor(10) | 2.43 ms / exact | 3.77 ms / exact | 6.49 ms / exact |
| LanceDB (IVF_FLAT) | — | — | 3.44 ms / exact |
| Qdrant Edge | 0.04 ms / exact | 0.07 ms / exact | 0.20 ms / exact |

### Results returned (k=10 requested)

Mean results returned at 100,000 chunks. A count below 10 where more than 10 chunks match the filter is the signature of search-then-discard: matches are lost, not merely ranked lower.

| engine | permissive | moderate | restrictive |
|---|---:|---:|---:|
| brute force | 10.0 | 10.0 | 10.0 |
| Orama | 10.0 | 10.0 | 10.0 |
| sqlite-vec | 10.0 | 10.0 | 10.0 |
| LanceDB (default IVF_PQ) | 10.0 | 10.0 | 10.0 |
| LanceDB + refineFactor(10) | 10.0 | 10.0 | 10.0 |
| LanceDB (IVF_FLAT) | 10.0 | 10.0 | 10.0 |
| Qdrant Edge | 10.0 | 10.0 | 10.0 |
"""Exact top-k ground truth by brute force, for every size x filter combo.

Vectors are L2-normalised, so cosine similarity is a plain dot product.
Also reports the duplicate-vector rate, because exact ties place a real
ceiling on any engine's measured recall.
"""
import json, sys
import numpy as np

DIM, K = 384, 10
SIZES = [1_000, 10_000, 100_000]

V = np.fromfile("data/vectors.f32", dtype=np.float32).reshape(-1, DIM)
Q = np.fromfile("data/queries.f32", dtype=np.float32).reshape(-1, DIM)
print(f"vectors {V.shape}  queries {Q.shape}", file=sys.stderr)

m = json.load(open("data/meta.json"))
vocab = np.array(m["folderVocab"])
folders = vocab[np.array(m["folderCode"], dtype=np.int32)]
mtimes = np.array(m["mtime"], dtype=np.int64)

filters = json.load(open("data/filters.json"))

# duplicate detection on the full set
uniq = len({V[i].tobytes() for i in range(min(len(V), 100_000))})
print(f"exact-duplicate vectors: {max(SIZES)-uniq:,} of {max(SIZES):,}", file=sys.stderr)

def mask_for(spec, n):
    m = np.ones(n, dtype=bool)
    if spec.get("folders"):
        m &= np.isin(folders[:n], spec["folders"])
    if spec.get("mtimeFrom"):
        m &= mtimes[:n] >= spec["mtimeFrom"]
    return m

gt = {"k": K, "sizes": {}, "duplicates": int(max(SIZES) - uniq)}
for n in SIZES:
    Vn = V[:n]
    S = Q @ Vn.T                                    # [nq, n] cosine
    entry = {}
    idx = np.argsort(-S, axis=1, kind="stable")[:, :K]
    entry["none"] = idx.tolist()
    for fname, spec in filters.items():
        m = mask_for(spec, n)
        cnt = int(m.sum())
        Sm = np.where(m[None, :], S, -np.inf)
        fi = np.argsort(-Sm, axis=1, kind="stable")[:, :K]
        # trim to however many actually matched
        entry[fname] = [[int(x) for x in row[:min(K, cnt)]] for row in fi]
        print(f"  n={n:,} filter={fname:12} matched={cnt:,} ({cnt/n:.2%})", file=sys.stderr)
    gt["sizes"][str(n)] = entry
    print(f"n={n:,} done", file=sys.stderr)

json.dump(gt, open("data/groundtruth.json", "w"))
print("WROTE data/groundtruth.json", file=sys.stderr)

"""Calibrate three filter predicates to ~80% / ~20% / ~1% selectivity.

Predicates are (folder IN set) AND (mtime >= cutoff): a single-valued
category plus a date range, which every engine under test can express.
Selected by an exhaustive search over (top-N folders x mtime quantile),
evaluated against the real corpus distribution via a 2-D cumulative
histogram, so the reported selectivities are measured, not assumed.
"""
import json, sys, datetime as dt
import numpy as np
from collections import Counter

N = 100_000
folders, mtimes = [], []
for i, line in enumerate(open("data/chunks.jsonl", encoding="utf-8")):
    if i >= N: break
    c = json.loads(line)
    folders.append(c["folder"]); mtimes.append(c["mtime"] or 0)
n = len(folders)
mt = np.array(mtimes, dtype=np.int64)

fc = Counter(folders)
order = [f for f, _ in fc.most_common()]
rank_of = {f: i for i, f in enumerate(order)}
rank = np.array([rank_of[f] for f in folders], dtype=np.int32)

print(f"chunks: {n:,}   distinct folders: {len(order)}", file=sys.stderr)
print(f"largest folders: {fc.most_common(6)}", file=sys.stderr)
print(f"mtime range: {dt.datetime.utcfromtimestamp(mt.min()).date()} .. "
      f"{dt.datetime.utcfromtimestamp(mt.max()).date()}", file=sys.stderr)

# candidate mtime cutoffs = percentiles of the real date distribution
QS = np.unique(np.percentile(mt, np.arange(0, 100, 0.5)).astype(np.int64))
bucket = np.searchsorted(QS, mt, side="right") - 1          # 0..len(QS)-1
nF, nQ = len(order), len(QS)

H = np.zeros((nF, nQ), dtype=np.int64)
np.add.at(H, (rank, bucket), 1)
# counts[nf, q] = rows with folder-rank < nf AND mtime >= QS[q]
counts = np.cumsum(np.cumsum(H[:, ::-1], axis=1)[:, ::-1], axis=0)

out = {}
for name, target in [("permissive", 0.80), ("moderate", 0.20), ("restrictive", 0.01)]:
    sel = counts / n
    d = np.abs(sel - target)
    nf_i, q_i = np.unravel_index(np.argmin(d), d.shape)
    nf, cut, s = nf_i + 1, int(QS[q_i]), float(sel[nf_i, q_i])
    out[name] = {"folders": order[:nf], "mtimeFrom": cut if q_i > 0 else None,
                 "target": target, "measured": s}
    print(f"{name:12} target={target:>5.0%}  measured={s:>7.3%}  "
          f"folders={nf:>3}  mtime>={dt.datetime.utcfromtimestamp(cut).date()}", file=sys.stderr)

json.dump(out, open("data/filters.json", "w"), indent=1)
print("WROTE data/filters.json", file=sys.stderr)

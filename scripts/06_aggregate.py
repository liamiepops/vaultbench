"""Collapse repetitions into median + spread and emit the results tables."""
import json, glob, statistics as st, sys, collections

runs = collections.defaultdict(list)
for fp in glob.glob("results/*.json"):
    if fp.replace("\\","/").endswith("results/summary.json"): continue
    r = json.load(open(fp))
    runs[(r["engine"], r["size"])].append(r)

def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return None, None
    return st.median(vals), (max(vals) - min(vals))

rows = []
for (eng, size), rs in sorted(runs.items(), key=lambda x: (x[0][1], x[0][0])):
    row = {"engine": eng, "size": size, "reps": len(rs)}
    for key in ("buildMs", "p50", "p95", "coldP50", "recall", "rssBytes", "diskBytes"):
        m, sp = agg([r.get(key) for r in rs])
        row[key], row[key + "_spread"] = m, sp
    row["indexed"] = rs[0].get("indexed")
    row["indexError"] = rs[0].get("indexError")
    row["persistError"] = rs[0].get("persistError")
    row["filtered"] = {}
    for fname in rs[0].get("filtered", {}):
        sub = {}
        for key in ("p50", "p95", "recall", "recallMean", "selectivity", "returnedMean"):
            m, sp = agg([r["filtered"][fname].get(key) for r in rs])
            sub[key], sub[key + "_spread"] = m, sp
        row["filtered"][fname] = sub
    rows.append(row)

json.dump(rows, open("results/summary.json", "w"), indent=1)
print(f"{'engine':<12}{'size':>8}{'build_s':>9}{'p50_ms':>9}{'p95_ms':>9}{'recall':>8}{'rss_MB':>8}{'disk_MB':>9}")
for r in rows:
    print(f"{r['engine']:<12}{r['size']:>8,}{(r['buildMs'] or 0)/1000:>9.2f}"
          f"{r['p50'] or 0:>9.3f}{r['p95'] or 0:>9.3f}{r['recall'] if r['recall'] is not None else float('nan'):>8.3f}"
          f"{(r['rssBytes'] or 0)/1e6:>8.0f}{(r['diskBytes'] or 0)/1e6:>9.1f}")
print()
for fname in ("permissive", "moderate", "restrictive"):
    print(f"--- {fname} ---")
    print(f"{'engine':<12}{'size':>8}{'p50_ms':>9}{'p95_ms':>9}{'recall':>8}{'ret':>7}")
    for r in rows:
        f = r["filtered"].get(fname)
        if not f: continue
        print(f"{r['engine']:<12}{r['size']:>8,}{f['p50'] or 0:>9.3f}{f['p95'] or 0:>9.3f}"
              f"{f['recall'] if f['recall'] is not None else float('nan'):>8.3f}{f['returnedMean'] or 0:>7.1f}")
    print()

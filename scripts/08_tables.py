"""Emit the FINDINGS.md results tables straight from results/summary.json."""
import json

rows = json.load(open("results/summary.json"))
ORDER = ["brute", "orama", "sqlite-vec", "lancedb", "lancedb-refine", "lancedb-flat", "qdrant-edge"]
LABEL = {"brute": "brute force", "orama": "Orama", "sqlite-vec": "sqlite-vec",
         "lancedb": "LanceDB (default IVF_PQ)", "lancedb-refine": "LanceDB + refineFactor(10)",
         "lancedb-flat": "LanceDB (IVF_FLAT)", "qdrant-edge": "Qdrant Edge"}
sizes = sorted({r["size"] for r in rows})
def get(e, s):
    for r in rows:
        if r["engine"] == e and r["size"] == s: return r
    return None
def f(x, d=2, dash="—"):
    return dash if x is None else f"{x:,.{d}f}"

out = []
for s in sizes:
    out.append(f"\n### {s:,} chunks\n")
    out.append("| engine | build | p50 | p95 | recall@10 | RSS | on-disk |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for e in ORDER:
        r = get(e, s)
        if not r: continue
        rec = "exact" if r["recall"] is not None and r["recall"] >= 0.9995 else f(r["recall"], 3)
        disk = ("**cannot persist**" if r.get("persistError")
                else f"{f((r['diskBytes'] or 0)/1e6,1)} MB")
        out.append(f"| {LABEL[e]} | {f((r['buildMs'] or 0)/1000,1)} s | {f(r['p50'],2)} ms | "
                   f"{f(r['p95'],2)} ms | {rec} | {f((r['rssBytes'] or 0)/1e6,0)} MB | {disk} |")

out.append("\n## Filtered search\n")
for fname in ("permissive", "moderate", "restrictive"):
    any_r = next((r["filtered"].get(fname) for r in rows if r["filtered"].get(fname)), None)
    selp = f"{any_r['selectivity']*100:.1f}%" if any_r else "?"
    out.append(f"\n### {fname} (~{selp} of corpus)\n")
    out.append("| engine | " + " | ".join(f"{s:,} p50 / recall" for s in sizes) + " |")
    out.append("|---|" + "---:|" * len(sizes))
    for e in ORDER:
        cells = []
        for s in sizes:
            r = get(e, s)
            d = r["filtered"].get(fname) if r else None
            if not d: cells.append("—"); continue
            rec = "exact" if d["recall"] is not None and d["recall"] >= 0.9995 else f(d["recall"], 3)
            cells.append(f"{f(d['p50'],2)} ms / {rec}")
        out.append(f"| {LABEL[e]} | " + " | ".join(cells) + " |")

out.append("\n### Results returned (k=10 requested)\n")
out.append("A count below 10 at a selective filter is the signature of "
           "search-then-discard: matches are lost, not merely ranked lower.\n")
out.append("| engine | " + " | ".join(f"{n}" for n in ("permissive","moderate","restrictive")) + " |")
out.append("|---|" + "---:|" * 3)
big = sizes[-1]
for e in ORDER:
    r = get(e, big)
    if not r: continue
    cells = [f(r["filtered"].get(n, {}).get("returnedMean"), 1) for n in ("permissive","moderate","restrictive")]
    out.append(f"| {LABEL[e]} | " + " | ".join(cells) + " |")

txt = "\n".join(out)
open("results/tables.md", "w", encoding="utf-8").write(txt)
print(txt)

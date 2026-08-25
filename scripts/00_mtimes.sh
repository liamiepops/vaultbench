#!/usr/bin/env bash
# Extract real last-modified time per file from git history (newest commit touching it).
# --no-renames avoids blob fetches on a blob:none partial clone.
set -e
REPO="$1"; SUB="$2"; OUT="$3"
cd "$REPO"
git log --format='C|%at' --name-only --no-renames --diff-filter=AM -- "$SUB" > "../$OUT.raw"
cd ..
python - "$OUT" "$SUB" <<'PY'
import json,sys,datetime as d
out,sub = sys.argv[1],sys.argv[2]
mt,cur = {},None
for line in open(out+".raw",encoding="utf-8",errors="replace"):
    line=line.rstrip("\n")
    if line.startswith("C|"): cur=int(line[2:]); continue
    if not line or cur is None: continue
    if line.startswith(sub+"/") and line.rsplit(".",1)[-1] in ("md","markdown"):
        rel=line[len(sub)+1:]
        if rel not in mt: mt[rel]=cur      # git log is newest-first
json.dump(mt,open(out,"w"))
vs=sorted(mt.values())
print(f"{out}: {len(mt):,} files with real mtime")
for p in (0,25,50,75,100):
    if vs:
        i=min(len(vs)-1,int(len(vs)*p/100))
        print(f"   p{p:<3} {d.datetime.utcfromtimestamp(vs[i]).date()}")
PY

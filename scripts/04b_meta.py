"""Compact metadata sidecar: only the fields the harness needs.

Keeps benchmark processes from loading the 114 MB chunk file, which would
otherwise inflate every engine's resident-memory figure by the same amount
and obscure the differences being measured.
"""
import json
ids, folders, mtimes = [], [], []
for line in open("data/chunks.jsonl", encoding="utf-8"):
    c = json.loads(line)
    ids.append(c["id"]); folders.append(c["folder"]); mtimes.append(c["mtime"] or 0)
vocab = sorted(set(folders))
vi = {f: i for i, f in enumerate(vocab)}
json.dump({"n": len(ids), "folderVocab": vocab,
           "folderCode": [vi[f] for f in folders], "mtime": mtimes},
          open("data/meta.json", "w"))
print(f"wrote data/meta.json  n={len(ids):,}  folders={len(vocab)}")

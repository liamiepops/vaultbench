"""Chunk real markdown documentation into fixed-size token windows.

One chunker, run once; every engine consumes the identical chunk set.
Window/stride are in MiniLM WordPiece tokens so a chunk never exceeds the
embedding model's real 256-token context (see FINDINGS: deviation from spec).
"""
import json, os, re, sys, random
from tokenizers import Tokenizer

WINDOW, STRIDE = 256, 224          # 32-token overlap
MAX_CHUNKS = 100_000
SEED = 42

# (source name, repo dir, subdir under repo, mtimes json)
SOURCES = [
    ("mdn",     "corpus-src",    "files/en-us", "data/mtimes_mdn.json"),
    ("dotnet",  "corpus-dotnet", "docs",        "data/mtimes_dotnet.json"),
    ("k8s",     "corpus-k8s",    "content/en",  "data/mtimes_k8s.json"),
    ("hass",    "corpus-ha",     "source",      "data/mtimes_ha.json"),
]

tok = Tokenizer.from_file("models/all-MiniLM-L6-v2/tokenizer.json")
tok.no_truncation(); tok.no_padding()
FM = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n", re.S)

def front_matter(txt):
    m = FM.match(txt)
    if not m: return {}, txt
    meta = {}
    for line in m.group(1).split("\n"):
        if ":" in line and not line.startswith((" ", "\t", "-", "#")):
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"\'')
    return meta, txt[m.end():]

def clean(txt):
    txt = re.sub(r"\{\{[^}]*\}\}", " ", txt)          # MDN / jekyll macros
    txt = re.sub(r"\{%[^%]*%\}", " ", txt)            # liquid tags
    txt = re.sub(r"```.*?```", " ", txt, flags=re.S)  # fenced code
    txt = re.sub(r"<[^>]+>", " ", txt)                # raw html
    txt = re.sub(r"[ \t]+", " ", txt)
    return re.sub(r"\n{3,}", "\n\n", txt).strip()

all_chunks, stats = [], {}
for name, repo, sub, mtf in SOURCES:
    root = os.path.join(repo, sub)
    if not os.path.isdir(root):
        print(f"!! missing source {name} ({root}) - skipped", file=sys.stderr); continue
    mtimes = json.load(open(mtf)) if os.path.exists(mtf) else {}
    files = []
    for dp, _, ns in os.walk(root):
        for n in ns:
            if n.endswith((".md", ".markdown")):
                files.append(os.path.join(dp, n).replace("\\", "/"))
    files.sort()
    used = toks = nch = 0
    for fp in files:
        try: raw = open(fp, encoding="utf-8").read()
        except Exception: continue
        meta, body = front_matter(raw)
        body = clean(body)
        if len(body) < 400: continue
        rel = os.path.relpath(fp, root).replace("\\", "/")
        ids = tok.encode(body, add_special_tokens=False).ids
        if len(ids) < 64: continue
        used += 1; toks += len(ids)
        parts = [p for p in rel.split("/")[:-1] if p]
        tags = sorted({name} | {p.lower() for p in parts[:2]})
        # single-valued category (top-level "folder"), expressible as a filter
        # predicate in every engine under test - unlike multi-valued tags.
        folder = f"{name}/{parts[0].lower()}" if parts else f"{name}/_root"
        mt = mtimes.get(rel)
        for i, st in enumerate(range(0, max(1, len(ids) - WINDOW + STRIDE), STRIDE)):
            piece = ids[st:st + WINDOW]
            if len(piece) < 48: break
            all_chunks.append({
                "source": name, "path": f"{name}/{rel}", "title": meta.get("title", "")[:200],
                "folder": folder,
                "tags": tags, "mtime": mt, "chunk_ix": i, "n_tokens": len(piece),
                "text": tok.decode(piece),
            })
            nch += 1
    stats[name] = (used, toks, nch, len(mtimes))
    print(f"{name:8} files={used:6,}  tokens={toks:12,}  chunks={nch:7,}  mtimes={len(mtimes):,}", file=sys.stderr)

print(f"TOTAL raw chunks: {len(all_chunks):,}", file=sys.stderr)
random.Random(SEED).shuffle(all_chunks)
sel = all_chunks[:MAX_CHUNKS]
with open("data/chunks.jsonl", "w", encoding="utf-8") as f:
    for i, c in enumerate(sel):
        c["id"] = i
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
have_mt = sum(1 for c in sel if c["mtime"])
print(f"WROTE {len(sel):,} chunks ({have_mt:,} with real mtime) -> data/chunks.jsonl", file=sys.stderr)
json.dump({k: {"files": v[0], "tokens": v[1], "chunks": v[2]} for k, v in stats.items()},
          open("data/corpus_stats.json", "w"), indent=1)

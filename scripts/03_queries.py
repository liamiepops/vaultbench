"""Build a fixed query set from real MDN document titles.

Titles are what a person actually types into a vault search box: short,
noun-phrase, topical. Sampled deterministically, deduplicated.
"""
import json, random, sys
SEED, N = 42, 200
titles, seen = [], set()
for line in open("data/chunks.jsonl", encoding="utf-8"):
    c = json.loads(line)
    t = (c["title"] or "").strip()
    if 3 <= len(t) <= 120 and t.lower() not in seen:
        seen.add(t.lower()); titles.append(t)
titles.sort()
random.Random(SEED).shuffle(titles)
sel = titles[:N]
with open("data/queries.jsonl","w",encoding="utf-8") as f:
    for i,t in enumerate(sel):
        f.write(json.dumps({"qid":i,"q":t}, ensure_ascii=False)+"\n")
print(f"wrote {len(sel)} queries; e.g. {sel[:5]}", file=sys.stderr)

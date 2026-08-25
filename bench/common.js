import fs from 'node:fs';
export const DIM = 384;

export function loadVectors(n) {
  const buf = fs.readFileSync('data/vectors.f32');
  const all = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  return all.subarray(0, n * DIM);
}
export function loadQueryVectors() {
  const buf = fs.readFileSync('data/queries.f32');
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}
export function loadMeta(n) {
  const out = [];
  const rl = fs.readFileSync('data/chunks.jsonl', 'utf8').split('\n');
  for (let i = 0; i < n; i++) {
    const c = JSON.parse(rl[i]);
    out.push({ id: c.id, path: c.path, tags: c.tags, folder: c.folder, mtime: c.mtime ?? 0, source: c.source });
  }
  return out;
}
export const loadFilters = () => JSON.parse(fs.readFileSync('data/filters.json', 'utf8'));
export const loadGroundTruth = () => JSON.parse(fs.readFileSync('data/groundtruth.json', 'utf8'));

/** Does a chunk satisfy a filter spec? Shared definition of truth. */
export function matches(m, f) {
  if (f.folders && !f.folders.includes(m.folder)) return false;
  if (f.mtimeFrom != null && !(m.mtime >= f.mtimeFrom)) return false;
  return true;
}

/** recall@k of `got` against exact `truth` (both arrays of ids). */
export function recallAt(got, truth, k) {
  if (!truth.length) return null;
  const t = new Set(truth.slice(0, k));
  let hit = 0;
  for (const g of got.slice(0, k)) if (t.has(g)) hit++;
  return hit / Math.min(k, t.size);
}

export function pct(sorted, p) {
  if (!sorted.length) return null;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p / 100 * sorted.length) - 1));
  return sorted[i];
}
export const median = a => pct([...a].sort((x, y) => x - y), 50);

export function dirSize(dir) {
  let total = 0;
  const walk = d => {
    for (const e of fs.readdirSync(d, { withFileTypes: true })) {
      const p = `${d}/${e.name}`;
      if (e.isDirectory()) walk(p);
      else try { total += fs.statSync(p).size; } catch {}
    }
  };
  try { fs.statSync(dir).isDirectory() ? walk(dir) : total = fs.statSync(dir).size; } catch {}
  return total;
}

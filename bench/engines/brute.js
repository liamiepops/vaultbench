/** Exact cosine over a flat Float32Array — the floor, and the source of truth.
 *  Vectors are L2-normalised at embed time, so cosine == dot product.
 *  Filters are evaluated inline against typed side-arrays, which is what a
 *  competent plain-array implementation does. */
export function createBrute() {
  let V = null, N = 0, folderCode = null, mtime = null, codeOf = null;
  const D = 384;
  return {
    name: 'brute',
    async build(vectors, meta) {
      V = vectors; N = meta.length;
      codeOf = new Map();
      folderCode = new Int32Array(N);
      mtime = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        let c = codeOf.get(meta[i].folder);
        if (c === undefined) { c = codeOf.size; codeOf.set(meta[i].folder, c); }
        folderCode[i] = c; mtime[i] = meta[i].mtime;
      }
      return { dir: null };
    },
    async query(q, k) { return topk(q, k, null, -Infinity); },
    async queryFiltered(q, k, f) {
      let allowed = null;
      if (f.folders) {
        allowed = new Uint8Array(codeOf.size);
        for (const name of f.folders) { const c = codeOf.get(name); if (c !== undefined) allowed[c] = 1; }
      }
      return topk(q, k, allowed, f.mtimeFrom ?? -Infinity);
    },
    /** What a plain-array plugin actually writes to disk. Reported as raw
     *  float32 + JSON metadata; note that an embeddings.json storing vectors
     *  as JSON numbers would be several times larger. */
    async persist(path) {
      const fs = await import('node:fs');
      fs.writeFileSync(`${path}.f32`, Buffer.from(V.buffer, V.byteOffset, V.length * 4));
      fs.writeFileSync(`${path}.meta.json`, JSON.stringify(
        Array.from({ length: N }, (_, i) => ({ i, f: folderCode[i], m: mtime[i] }))));
      return fs.statSync(`${path}.f32`).size + fs.statSync(`${path}.meta.json`).size;
    },
    async close() { V = null; folderCode = null; mtime = null; codeOf = null; },
  };
  function topk(q, k, allowed, minMtime) {
    const ids = new Int32Array(k).fill(-1), sc = new Float64Array(k).fill(-Infinity);
    let worst = -Infinity, filled = 0;
    for (let i = 0; i < N; i++) {
      if (allowed !== null && allowed[folderCode[i]] === 0) continue;
      if (mtime[i] < minMtime) continue;
      const off = i * D;
      let s = 0;
      for (let d = 0; d < D; d++) s += V[off + d] * q[d];
      if (filled === k && s <= worst) continue;
      let j = Math.min(filled, k - 1);
      while (j > 0 && sc[j - 1] < s) { sc[j] = sc[j - 1]; ids[j] = ids[j - 1]; j--; }
      sc[j] = s; ids[j] = i;
      if (filled < k) filled++;
      worst = sc[filled - 1];
    }
    const out = [];
    for (let i = 0; i < filled; i++) if (ids[i] !== -1) out.push(ids[i]);
    return out;
  }
}

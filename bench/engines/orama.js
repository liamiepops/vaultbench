/** Orama 3.x — JS-native search library with vector support.
 *  Used by at least one shipping Obsidian plugin. */
import { create, insertMultiple, searchVector, save } from '@orama/orama';

export function createOrama() {
  let db = null, idOf = null;
  return {
    name: 'orama',
    async build(vectors, meta) {
      db = create({
        schema: { embedding: 'vector[384]', tags: 'string[]', mtime: 'number' },
      });
      const docs = new Array(meta.length);
      for (let i = 0; i < meta.length; i++) {
        docs[i] = {
          id: String(meta[i].id),
          embedding: Array.from(vectors.subarray(i * 384, i * 384 + 384)),
          tags: meta[i].tags,
          mtime: meta[i].mtime,
        };
      }
      await insertMultiple(db, docs, 5000);
      idOf = null;
      return { dir: null };
    },
    async query(q, k) {
      const r = await searchVector(db, { vector: { value: Array.from(q), property: 'embedding' }, similarity: -1, limit: k, includeVectors: false });
      return r.hits.map(h => Number(h.id));
    },
    async queryFiltered(q, k, f) {
      const where = {};
      if (f.tags) where.tags = f.tags;
      if (f.mtimeFrom != null) where.mtime = { gte: f.mtimeFrom };
      const r = await searchVector(db, { vector: { value: Array.from(q), property: 'embedding' }, similarity: -1, limit: k, where, includeVectors: false });
      return r.hits.map(h => Number(h.id));
    },
    /** Orama's own persistence format, which is what a plugin would write. */
    async persist(path) {
      const fs = await import('node:fs');
      fs.writeFileSync(`${path}.orama.json`, JSON.stringify(save(db)));
      return fs.statSync(`${path}.orama.json`).size;
    },
    async close() { db = null; },
  };
}

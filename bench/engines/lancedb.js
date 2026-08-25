/** LanceDB — embedded vector database, Node bindings via napi.
 *  Vector index is IVF_PQ (LanceDB's documented default for createIndex).
 *  Scalar indices are added on the filter columns so filtered search can be
 *  pushed down rather than brute-forced. */
import * as lancedb from '@lancedb/lancedb';
import fs from 'node:fs';

export function createLanceDb({ index = 'ivfPq', refineFactor = 0 } = {}) {
  let db = null, tbl = null, dir = null, indexed = false;
  return {
    name: `lancedb`,
    async build(vectors, meta, d) {
      dir = `${d}/lance`;
      fs.rmSync(dir, { recursive: true, force: true });
      db = await lancedb.connect(dir);
      const rows = new Array(meta.length);
      for (let i = 0; i < meta.length; i++) {
        rows[i] = {
          id: meta[i].id,
          vector: Array.from(vectors.subarray(i * 384, i * 384 + 384)),
          folder: meta[i].folder,
          mtime: meta[i].mtime,
        };
      }
      tbl = await db.createTable('chunks', rows);
      // IVF_PQ needs enough rows to train; below that LanceDB stays exact/flat.
      try {
        await tbl.createIndex('vector', { config: lancedb.Index[index]({ distanceType: 'cosine' }) });
        indexed = true;
      } catch (e) {
        indexed = false;
        this.indexError = String(e.message || e).slice(0, 200);
      }
      try {
        await tbl.createIndex('folder', { config: lancedb.Index.bitmap() });
        await tbl.createIndex('mtime', { config: lancedb.Index.btree() });
      } catch { /* scalar index optional */ }
      return { dir, indexed };
    },
    async query(q, k) {
      let qq = tbl.search(Array.from(q)).distanceType('cosine').limit(k).select(['id']);
      if (refineFactor) qq = qq.refineFactor(refineFactor);
      const r = await qq.toArray();
      return r.map(x => Number(x.id));
    },
    async queryFiltered(q, k, f) {
      const cond = [];
      if (f.folders) cond.push(`folder IN (${f.folders.map(s => `'${s.replace(/'/g, "''")}'`).join(',')})`);
      if (f.mtimeFrom != null) cond.push(`mtime >= ${f.mtimeFrom}`);
      let qq = tbl.search(Array.from(q)).distanceType('cosine').limit(k).select(['id']);
      if (refineFactor) qq = qq.refineFactor(refineFactor);
      if (cond.length) qq = qq.where(cond.join(' AND '));   // prefilter is LanceDB's default
      const r = await qq.toArray();
      return r.map(x => Number(x.id));
    },
    async close() { tbl = null; db = null; },
  };
}

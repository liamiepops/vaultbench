/** sqlite-vec: loadable SQLite extension, driven through better-sqlite3.
 *  vec0 virtual tables keep a flat (exact) vector store plus filterable
 *  metadata columns, so this is an exact engine, not an approximate one. */
import Database from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import fs from 'node:fs';

export function createSqliteVec() {
  let db = null, file = null, qStmt = null;
  return {
    name: 'sqlite-vec',
    async build(vectors, meta, dir) {
      file = `${dir}/vec.db`;
      fs.rmSync(file, { force: true });
      db = new Database(file);
      sqliteVec.load(db);
      db.pragma('journal_mode = WAL');
      db.exec(`CREATE VIRTUAL TABLE v USING vec0(
        id INTEGER PRIMARY KEY,
        embedding float[384] distance_metric=cosine,
        folder TEXT,
        mtime INTEGER
      )`);
      const ins = db.prepare('INSERT INTO v(id, embedding, folder, mtime) VALUES (?,?,?,?)');
      const tx = db.transaction(ms => {
        for (let i = 0; i < ms.length; i++) {
          const sub = vectors.subarray(i * 384, i * 384 + 384);
          // better-sqlite3 binds JS numbers as REAL; vec0 rejects non-INTEGER
          // for its primary key and INTEGER metadata columns, so bind BigInt.
          ins.run(BigInt(ms[i].id), Buffer.from(sub.buffer, sub.byteOffset, 384 * 4), ms[i].folder, BigInt(ms[i].mtime));
        }
      });
      tx(meta);
      qStmt = db.prepare('SELECT id FROM v WHERE embedding MATCH ? AND k = ?');
      return { dir: file };
    },
    async query(q, k) {
      return qStmt.all(Buffer.from(q.buffer, q.byteOffset, 384 * 4), BigInt(k)).map(r => Number(r.id));
    },
    async queryFiltered(q, k, f) {
      const cond = [], params = [Buffer.from(q.buffer, q.byteOffset, 384 * 4), BigInt(k)];
      if (f.folders) { cond.push(`folder IN (${f.folders.map(() => '?').join(',')})`); params.push(...f.folders); }
      if (f.mtimeFrom != null) { cond.push('mtime >= ?'); params.push(BigInt(f.mtimeFrom)); }
      const sql = `SELECT id FROM v WHERE embedding MATCH ? AND k = ?${cond.length ? ' AND ' + cond.join(' AND ') : ''}`;
      return db.prepare(sql).all(...params).map(r => Number(r.id));
    },
    async close() { try { db?.close(); } catch {} db = null; },
  };
}

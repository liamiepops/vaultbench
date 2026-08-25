/** Qdrant Edge, via a thin napi-rs binding over the published `qdrant-edge`
 *  crate (see engines-native/qe-napi). HNSW is built by the shard optimizer,
 *  so build() is followed by an explicit optimize() pass. */
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);

export function createQdrantEdge() {
  let idx = null, native = null;
  return {
    name: 'qdrant-edge',
    async build(vectors, meta, dir) {
      native = require('../../engines-native/qe-napi/qe_napi.node');
      const buf = Buffer.from(vectors.buffer, vectors.byteOffset, vectors.length * 4);
      idx = native.QeIndex.build(
        `${dir}/shard`, buf,
        meta.map(m => m.id), meta.map(m => m.folder), meta.map(m => m.mtime), 384,
      );
      return { dir: `${dir}/shard` };
    },
    async optimize() { this.indexed = idx.optimize(); },
    async query(q, k) {
      return idx.search(Buffer.from(q.buffer, q.byteOffset, q.length * 4), k);
    },
    async queryFiltered(q, k, f) {
      return idx.searchFiltered(
        Buffer.from(q.buffer, q.byteOffset, q.length * 4), k,
        f.folders ?? [], f.mtimeFrom ?? 0);
    },
    async close() { idx = null; },
  };
}

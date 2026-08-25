/** One engine, one corpus size, one repetition. Runs in its own process so
 *  resident memory is attributable to this engine alone. */
import fs from 'node:fs';
import { loadVectors, loadQueryVectors, loadMeta, loadFilters, loadGroundTruth,
         recallAt, pct, median, dirSize, DIM } from './common.js';

const arg = k => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? null : process.argv[i + 1]; };
const ENGINE = arg('engine'), SIZE = Number(arg('size')), REP = Number(arg('rep') ?? 0);
const K = 10, WARMUP = 20;

const factories = {
  brute:        () => import('./engines/brute.js').then(m => m.createBrute()),
  orama:        () => import('./engines/orama.js').then(m => m.createOrama()),
  'sqlite-vec': () => import('./engines/sqlitevec.js').then(m => m.createSqliteVec()),
  lancedb:      () => import('./engines/lancedb.js').then(m => m.createLanceDb()),
  // Variants probing whether LanceDB's lossy default index is remediable
  // with documented options rather than genuine tuning.
  'lancedb-refine': () => import('./engines/lancedb.js').then(m => m.createLanceDb({ refineFactor: 10 })),
  'lancedb-flat':   () => import('./engines/lancedb.js').then(m => m.createLanceDb({ index: 'ivfFlat' })),
  'qdrant-edge':() => import('./engines/qdrantedge.js').then(m => m.createQdrantEdge()),
};

const nowMs = () => Number(process.hrtime.bigint()) / 1e6;

async function timeQueries(fn, Q, nq) {
  const lat = [], res = [];
  for (let i = 0; i < nq; i++) {
    const q = Q.subarray(i * DIM, i * DIM + DIM);
    const t0 = nowMs();
    const r = await fn(q);
    lat.push(nowMs() - t0);
    res.push(r);
  }
  return { lat, res };
}

const main = async () => {
  const vectors = loadVectors(SIZE);
  const meta = loadMeta(SIZE);
  const Q = loadQueryVectors();
  const nq = Q.length / DIM;
  const filters = loadFilters();
  const gt = loadGroundTruth().sizes[String(SIZE)];

  const dir = `idx/${ENGINE}_${SIZE}_${REP}`;
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });

  const eng = await factories[ENGINE]();
  const tBuild = nowMs();
  const info = await eng.build(vectors, meta, dir);
  if (eng.optimize) await eng.optimize();
  const buildMs = nowMs() - tBuild;

  const out = {
    engine: ENGINE, size: SIZE, rep: REP, k: K, nQueries: nq,
    buildMs, indexed: info?.indexed ?? eng.indexed ?? null, indexError: eng.indexError ?? null,
  };

  // cold pass (index just built, caches unwarmed)
  const cold = await timeQueries(q => eng.query(q, K), Q, Math.min(nq, 50));
  out.coldP50 = pct([...cold.lat].sort((a, b) => a - b), 50);

  for (let i = 0; i < WARMUP; i++) await eng.query(Q.subarray(i * DIM, i * DIM + DIM), K);

  const warm = await timeQueries(q => eng.query(q, K), Q, nq);
  const sorted = [...warm.lat].sort((a, b) => a - b);
  out.p50 = pct(sorted, 50); out.p95 = pct(sorted, 95); out.mean = warm.lat.reduce((a, b) => a + b, 0) / nq;
  out.recall = median(warm.res.map((r, i) => recallAt(r, gt.none[i], K)).filter(x => x != null));

  out.filtered = {};
  for (const [fname, spec] of Object.entries(filters)) {
    for (let i = 0; i < Math.min(WARMUP, nq); i++) await eng.queryFiltered(Q.subarray(i * DIM, i * DIM + DIM), K, spec);
    const fr = await timeQueries(q => eng.queryFiltered(q, K, spec), Q, nq);
    const fs_ = [...fr.lat].sort((a, b) => a - b);
    const rec = fr.res.map((r, i) => recallAt(r, gt[fname][i], K)).filter(x => x != null);
    out.filtered[fname] = {
      selectivity: spec.measured,
      p50: pct(fs_, 50), p95: pct(fs_, 95),
      recall: median(rec),
      recallMean: rec.reduce((a, b) => a + b, 0) / Math.max(rec.length, 1),
      returnedMean: fr.res.reduce((a, r) => a + r.length, 0) / fr.res.length,
    };
  }

  // In-memory engines are asked to write what a plugin would actually persist;
  // on-disk engines already have. Size is taken after close so that anything
  // still sitting in a write-ahead log has been checkpointed.
  if (eng.persist) {
    try { await eng.persist(`${dir}/persisted`); }
    catch (e) { out.persistError = String(e.message || e).slice(0, 200); }
  }

  if (global.gc) global.gc();
  out.rssBytes = process.memoryUsage.rss();
  out.heapBytes = process.memoryUsage().heapUsed;

  await eng.close();
  out.diskBytes = dirSize(dir);
  fs.mkdirSync('results', { recursive: true });
  fs.writeFileSync(`results/${ENGINE}_${SIZE}_${REP}.json`, JSON.stringify(out, null, 1));
  console.log(JSON.stringify({ engine: ENGINE, size: SIZE, rep: REP, buildMs: +buildMs.toFixed(0),
    p50: +out.p50.toFixed(3), p95: +out.p95.toFixed(3), recall: out.recall,
    rssMB: +(out.rssBytes / 1e6).toFixed(0), diskMB: +(out.diskBytes / 1e6).toFixed(1),
    filtered: Object.fromEntries(Object.entries(out.filtered).map(([k, v]) =>
      [k, { p50: +v.p50.toFixed(3), recall: +v.recall.toFixed(3) }])) }));
};
main().catch(e => { console.error('FAIL', ENGINE, SIZE, e.stack || e.message); process.exit(1); });

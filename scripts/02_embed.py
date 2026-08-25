"""Embed all chunks and all queries once with all-MiniLM-L6-v2 (ONNX, CPU).

Vectors are persisted as raw float32 [N,384] and reused by every engine, so
no measurement ever includes embedding cost.
"""
import json, sys, time, os
import numpy as np, onnxruntime as ort
from tokenizers import Tokenizer

MODEL="models/all-MiniLM-L6-v2"; DIM=384; MAXLEN=256; BATCH=64

tok = Tokenizer.from_file(f"{MODEL}/tokenizer.json")
tok.enable_truncation(MAXLEN); tok.enable_padding(length=None)

so = ort.SessionOptions()
so.intra_op_num_threads = 8
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(f"{MODEL}/onnx/model.onnx", so, providers=["CPUExecutionProvider"])
INPUTS = {i.name for i in sess.get_inputs()}

def embed(texts):
    out = np.empty((len(texts), DIM), dtype=np.float32)
    t0 = time.time()
    for s in range(0, len(texts), BATCH):
        batch = texts[s:s+BATCH]
        enc = tok.encode_batch(batch)
        ids  = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in INPUTS:
            feed["token_type_ids"] = np.zeros_like(ids)
        last = sess.run(None, feed)[0]                       # [B,T,384]
        m = mask[..., None].astype(np.float32)
        vec = (last * m).sum(1) / np.clip(m.sum(1), 1e-9, None)   # mean pool
        vec /= np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12, None)
        out[s:s+len(batch)] = vec.astype(np.float32)
        if s % (BATCH*50) == 0:
            done = s+len(batch); el = time.time()-t0
            print(f"  {done:,}/{len(texts):,}  {done/max(el,1e-9):.1f}/s  eta {(len(texts)-done)/max(done/max(el,1e-9),1e-9)/60:.1f}m", file=sys.stderr, flush=True)
    return out

what = sys.argv[1] if len(sys.argv)>1 else "chunks"

if what == "chunks":
    texts=[]
    for line in open("data/chunks.jsonl", encoding="utf-8"):
        texts.append(json.loads(line)["text"])
    print(f"embedding {len(texts):,} chunks", file=sys.stderr)
    v = embed(texts)
    v.tofile("data/vectors.f32")
    print(f"WROTE data/vectors.f32 shape={v.shape} norm0={np.linalg.norm(v[0]):.6f}", file=sys.stderr)
else:
    qs=[json.loads(l)["q"] for l in open("data/queries.jsonl", encoding="utf-8")]
    print(f"embedding {len(qs)} queries", file=sys.stderr)
    v = embed(qs)
    v.tofile("data/queries.f32")
    print(f"WROTE data/queries.f32 shape={v.shape}", file=sys.stderr)

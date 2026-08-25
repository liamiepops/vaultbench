#!/usr/bin/env bash
# Every engine, every size, N repetitions. One process per run so resident
# memory is attributable. Failures are recorded, not fatal.
REPS=${REPS:-3}
SIZES=${SIZES:-"1000 10000 100000"}
ENGINES=${ENGINES:-"brute orama sqlite-vec lancedb qdrant-edge"}
NODE_FLAGS="--expose-gc --max-old-space-size=20480"
mkdir -p results logs idx
for size in $SIZES; do
  for eng in $ENGINES; do
    for rep in $(seq 0 $((REPS-1))); do
      echo "=== $eng size=$size rep=$rep ==="
      if ! node $NODE_FLAGS bench/run.js --engine "$eng" --size "$size" --rep "$rep" \
           > "logs/${eng}_${size}_${rep}.log" 2>&1; then
        echo "  FAILED (see logs/${eng}_${size}_${rep}.log)"
        tail -3 "logs/${eng}_${size}_${rep}.log" | sed 's/^/    /'
      else
        tail -1 "logs/${eng}_${size}_${rep}.log" | sed 's/^/    /'
      fi
      rm -rf "idx/${eng}_${size}_${rep}"
    done
  done
done

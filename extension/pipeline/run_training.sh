#!/bin/bash
# On the GPU pod, from the repo root, after setup_pod.sh (extension/docs/RUN.md, step 3). Builds the training text
# (skipped if already built), runs a 5-step trial, then post-trains seed_0, seed_1, seed_2 back to back
# (DECISIONS D-46). Writes its own log to extension/results/training/training.log. Safe to rerun after a crash:
# finished seeds are skipped and the interrupted one continues from its last checkpoint (D-47). Training saves
# only on this pod; with HF_REPO set, each finished seed's adapter (not its checkpoints) is then copied to that
# private HuggingFace repo (D-50).
set -euo pipefail
mkdir -p extension/results/training
exec > >(tee -a extension/results/training/training.log) 2>&1
PY=${PY:-.venv/bin/python}
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== $(date)  HF_REPO=${HF_REPO:-<not set: adapters stay on this pod only>}"

[ -f extension/data/dolma/replay.jsonl ] || $PY -m extension.pipeline.fetch_dolma   # stops if it differs from extension/data/dolma.json
for s in 0 1 2; do
  [ -f extension/data/seed_$s/train.jsonl ] || $PY -m extension.pipeline.build_data --seed $s   # stops if it differs from extension/data/seed_$s/build.json
done

if [ ! -f extension/results/training/trial/run.json ]; then
  echo "=== trial: 5 steps of seed_0"
  $PY -m extension.pipeline.train --seed 0 --max-steps 5 --log-steps 1 --out extension/results/training/trial
  rm -rf extension/results/training/trial/checkpoint-*
fi

for s in 0 1 2; do
  echo "=== seed_$s"
  $PY -m extension.pipeline.train --seed $s
  rm -rf extension/results/training/seed_$s/checkpoint-*   # finished: the adapter is saved, crash-recovery copies no longer needed
  if [ -n "${HF_REPO:-}" ]; then
    "$(dirname "$PY")/hf" upload "$HF_REPO" extension/results/training/seed_$s seed_$s --private --exclude "checkpoint-*" \
      || echo "WARNING: upload of seed_$s to $HF_REPO failed; the adapter is still in extension/results/training/seed_$s"
  fi
done
echo "=== all three seeds trained $(date)"

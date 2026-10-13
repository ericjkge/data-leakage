#!/bin/bash
# On the GPU pod, from the repo root, after setup_pod.sh (extension/RUN.md). Step 4 of PLAN.md (D-20 to D-23, D-44,
# details in D-53): attack prompts and generations for each seed's model, written to results/ext/<model>/<attack>/.
# Infini-gram matching is not run here (laptop, step 5). With 3+ GPUs the seeds run side by side, one GPU each;
# otherwise one after another. Safe to rerun: scripts.run and ea_sample resume where they stopped.
set -euo pipefail
PY=${PY:-.venv/bin/python}
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
OUT=results/ext
LOGS=${LOGS:-/workspace/attack_logs}   # not committed
mkdir -p $LOGS

# Shared prompts (D-44): RSA and RWA copied from the base model's reproduction run; RA rebuilt with the D-21 prompt.
mkdir -p $OUT/olmo-7b/ra
[ -s $OUT/olmo-7b/ra/prompts.jsonl ] || $PY -c "
import json
from src.attacks.ra import RA
with open('$OUT/olmo-7b/ra/prompts.jsonl', 'w') as f:
    for i, row in enumerate(RA().make_prompts(None, None, 500)):
        f.write(json.dumps({'id': i, **row}) + '\n')"
for s in 0 1 2; do
  for a in rsa rwa; do mkdir -p $OUT/olmo-7b-pano-seed_$s/$a; cp -n results/olmo-7b/$a/prompts.jsonl $OUT/olmo-7b-pano-seed_$s/$a/; done
  mkdir -p $OUT/olmo-7b-pano-seed_$s/ra; cp -n $OUT/olmo-7b/ra/prompts.jsonl $OUT/olmo-7b-pano-seed_$s/ra/
done

one_seed() {  # $1 = seed; CIA last, as the slowest (PLAN.md step 4)
  local m=olmo-7b-pano-seed_$1
  for a in rsa rwa ra; do echo "=== $m $a"; $PY -m scripts.run --model $m --attack $a --stage generate --out $OUT; done
  echo "=== $m ea_sample"; $PY -m src.attacks.ea_sample --model $m --seed 0 --out $OUT
  echo "=== $m cia"; $PY -m scripts.run --model $m --attack cia --stage attack --n 100 --out $OUT
  $PY -m scripts.run --model $m --attack cia --stage generate --n 100 --out $OUT
  echo "=== $m done"
}

base_ra() {  # D-21: the reproduction's RA on base OLMo, rerun with the corrected prompt
  echo "=== olmo-7b ra"; $PY -m scripts.run --model olmo-7b --attack ra --stage generate --out $OUT
}

if [ "$(nvidia-smi -L | wc -l)" -ge 3 ]; then
  for s in 0 1 2; do
    (export CUDA_VISIBLE_DEVICES=$s; [ $s = 0 ] && base_ra; one_seed $s) > $LOGS/seed_$s.log 2>&1 &
  done
  wait
else
  (base_ra; for s in 0 1 2; do one_seed $s; done) > $LOGS/all.log 2>&1
fi
echo "=== all attacks done"

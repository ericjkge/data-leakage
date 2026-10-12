#!/bin/bash
# One-time setup on a fresh GPU pod, run from the repo root (extension/RUN.md, step 2): a Python 3.12
# environment with the library versions we tested, then OLMo-7B-0424 (27.6 GB) into the HuggingFace cache.
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}

command -v uv >/dev/null || pip install uv
[ -x .venv/bin/python ] || uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/hf download allenai/OLMo-7B-0424-hf --revision e3c82a0871580b424d74e68acee866bb007df7df

.venv/bin/python - <<'EOF'
import torch, transformers, peft
assert torch.cuda.is_available(), "no GPU visible"
free, total = torch.cuda.mem_get_info()
print(f"GPU: {torch.cuda.get_device_name()}, {total / 1e9:.0f} GB; torch {torch.__version__}, "
      f"transformers {transformers.__version__}, peft {peft.__version__}")
assert total > 70e9, "needs an 80 GB GPU"
EOF
echo "setup done"

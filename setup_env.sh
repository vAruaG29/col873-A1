#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-col873}"
PY_VER="${PY_VER:-3.10}"
CUDA_TAG="${CUDA_TAG:-cu124}"
MAKE_ENV=1
[[ "${1:-}" == "--no-env" ]] && MAKE_ENV=0

if [[ $MAKE_ENV -eq 1 ]]; then
  echo "== creating conda env '$ENV_NAME' (python $PY_VER) =="
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
fi

echo "== python: $(which python) ($(python -V 2>&1)) =="
python -m pip install --upgrade pip wheel

echo "== installing torch ($CUDA_TAG) =="
python -m pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "== installing the rest =="
python -m pip install -r requirements.txt

echo "== optional: spaCy pipeline for Presidio =="
python -m spacy download en_core_web_lg || \
  echo "[warn] en_core_web_lg failed; use --spacy_model en_core_web_sm or skip --system presidio"

echo "== pre-downloading the two models (so runs do not need the network) =="
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
for m in ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B"):
    print("fetching", m, flush=True)
    AutoTokenizer.from_pretrained(m)
    AutoModelForCausalLM.from_pretrained(m)
print("models cached under", __import__("os").environ.get("HF_HOME", "~/.cache/huggingface"))
PY

echo
echo "== sanity check =="
python - <<'PY'
import torch, transformers, peft
print("torch       ", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("peft        ", peft.__version__)
if torch.cuda.is_available():
    print("gpu         ", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
try:
    import bitsandbytes; print("bitsandbytes", bitsandbytes.__version__)
except Exception as e:
    print("bitsandbytes MISSING ->", e, "(run A6/QLoRA will fail)")
PY
echo
echo "done. activate with:  conda activate $ENV_NAME"

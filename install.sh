#!/usr/bin/env bash
# One-shot installer for Paper2MD (Linux / WSL, CPU only).
#
#   ./install.sh                 full install: light deps + PaddleOCR-VL
#   ./install.sh --no-paddle     DeepSeek engine only (no Paddle, no models)
#
# With --no-paddle the app is ~30 MB of dependencies, runs in ~300 MB of RAM,
# and needs OCR_ENGINE=deepseek plus a DEEPSEEK_API_KEY. Without it you also get
# the local PaddleOCR-VL engine, which is free and works offline but needs
# PaddlePaddle plus 2-4 GB of models and ~9 GB of RAM.
#
# Override the package index or Paddle version if you like:
#   PIP_INDEX=https://pypi.org/simple/ PADDLE_VERSION=3.3.1 ./install.sh
set -euo pipefail
cd "$(dirname "$0")"

# Tsinghua by default: it served the PaddlePaddle wheel at ~7 MB/s where
# Aliyun's mirror managed ~115 KB/s for the same file.
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple/}"
PADDLE_VERSION="${PADDLE_VERSION:-3.3.0}"
PYTHON="${PYTHON:-python3}"

WITH_PADDLE=1
for arg in "$@"; do
  case "$arg" in
    --no-paddle|--deepseek-only) WITH_PADDLE=0 ;;
    -h|--help)
      sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

echo "==> creating .venv"
"$PYTHON" -m venv .venv 2>/dev/null || "$PYTHON" -m venv --without-pip .venv

# Some minimal distros ship python3 without ensurepip (python3-venv missing).
if [ ! -x .venv/bin/pip ]; then
  echo "==> bootstrapping pip (ensurepip unavailable)"
  curl -sSL -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
  .venv/bin/python /tmp/get-pip.py
fi

cat > .venv/pip.conf <<EOF
[global]
index-url = $PIP_INDEX
timeout = 120
retries = 5
EOF

echo "==> installing application dependencies"
.venv/bin/pip install -r requirements.txt

if [ "$WITH_PADDLE" = "1" ]; then
  echo "==> installing PaddlePaddle $PADDLE_VERSION (CPU)"
  .venv/bin/pip install "paddlepaddle==$PADDLE_VERSION"

  echo "==> installing PaddleOCR-VL (first run downloads the models, ~2-4 GB)"
  .venv/bin/pip install "paddleocr[doc-parser]"
else
  echo "==> skipping PaddleOCR-VL (--no-paddle)"
  echo "    set OCR_ENGINE=deepseek and DEEPSEEK_API_KEY in .env"
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> wrote .env from .env.example"
fi

if [ "$WITH_PADDLE" = "0" ]; then
  # Keep the config honest: without Paddle, auto has nothing to fall back to.
  if grep -q '^OCR_ENGINE=' .env; then
    sed -i 's/^OCR_ENGINE=.*/OCR_ENGINE=deepseek/' .env
    echo "==> set OCR_ENGINE=deepseek in .env"
  fi
  echo "==> set DEEPSEEK_API_KEY in .env to finish setup"
else
  echo "==> set IMGBB_API_KEY to enable image hosting"
fi

echo
echo "Done. Start the app with:  ./run.sh"
echo "Then open:                 http://localhost:8000"

#!/usr/bin/env bash
# One-shot installer for Paper2MD (Linux / WSL, CPU only).
#
#   ./install.sh
#
# Override the package index or Paddle version if you like:
#   PIP_INDEX=https://pypi.org/simple/ PADDLE_VERSION=3.3.1 ./install.sh
set -euo pipefail
cd "$(dirname "$0")"

PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
PADDLE_VERSION="${PADDLE_VERSION:-3.3.0}"
PYTHON="${PYTHON:-python3}"

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

echo "==> installing PaddlePaddle $PADDLE_VERSION (CPU)"
.venv/bin/pip install "paddlepaddle==$PADDLE_VERSION"

echo "==> installing PaddleOCR-VL (first run downloads the models, ~2-4 GB)"
.venv/bin/pip install "paddleocr[doc-parser]"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> wrote .env from .env.example — set IMGBB_API_KEY to enable image hosting"
fi

echo
echo "Done. Start the app with:  ./run.sh"
echo "Then open:                 http://localhost:8000"

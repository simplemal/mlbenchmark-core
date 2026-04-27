#!/usr/bin/env bash
# setup.sh — creates the venv and installs all dependencies for mlbenchmark-core.
#
# Prerequisites:
#   - Python 3.12 (from python.org — not the system Python on macOS)
#   - Xcode Command Line Tools  →  xcode-select --install
#   - CMake  →  brew install cmake
#
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# After setup:
#   source .venv/bin/activate
#   python3 cli.py info

set -euo pipefail

PYTHON="${PYTHON:-python3.12}"

echo ""
echo "ML Benchmark Core — Setup"
echo "════════════════════════════════════════"

# ── Check Python ───────────────────────────────────────────────────────────
if ! command -v "$PYTHON" &>/dev/null; then
    echo "✗  $PYTHON not found."
    echo "   Download Python 3.12 from https://www.python.org/downloads/"
    exit 1
fi

PY_VER=$("$PYTHON" --version 2>&1)
echo "▶ Python: $PY_VER"

# ── Create venv ────────────────────────────────────────────────────────────
if [[ -d .venv ]]; then
    echo "▶ .venv already exists — updating packages…"
else
    echo "▶ Creating .venv…"
    "$PYTHON" -m venv .venv
fi

source .venv/bin/activate
pip install --quiet --upgrade pip

# ── Standard packages ──────────────────────────────────────────────────────
echo "▶ Installing standard dependencies…"
pip install --quiet -r requirements.txt

# ── llama-cpp-python with Metal acceleration ───────────────────────────────
echo "▶ Installing llama-cpp-python (Metal GPU)…"
CMAKE_ARGS="-DGGML_METAL=on" \
    pip install --quiet "llama-cpp-python==0.3.17" \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal

# ── mlc-llm + tvm — pinned wheels from mlbenchmark.app ────────────────────
echo "▶ Installing mlc-llm (pinned wheels, Metal)…"
# The official nightlies from mlc.ai change frequently and often break
# compatibility (e.g. missing BlockBuilder._func_stack with LLVM 19).
# We use wheels tested and hosted on mlbenchmark.app to guarantee stability.
WHEELS_BASE="https://mlbenchmark.app/api/v1/wheels"
pip install --quiet \
    "${WHEELS_BASE}/mlc_ai_nightly_cpu-0.24.dev0-py3-none-macosx_13_0_arm64.whl" \
    "${WHEELS_BASE}/mlc_llm_nightly_cpu-0.20.dev147-py3-none-macosx_13_0_arm64.whl"

# ── Hugging Face token ────────────────────────────────────────────────────
# Required to download models (some are gated on HF).
# Get your token at https://huggingface.co/settings/tokens
echo ""
echo "────────────────────────────────────────"
echo "Hugging Face token"
echo ""
echo "Models are downloaded from Hugging Face."
echo "Some are gated and require an HF account with approved access."
echo "Get your token at: https://huggingface.co/settings/tokens"
echo ""

if command -v huggingface-cli &>/dev/null; then
    # Check if a token is already saved
    EXISTING_TOKEN=$(huggingface-cli whoami 2>/dev/null | head -1 || true)
    if [[ -n "$EXISTING_TOKEN" && "$EXISTING_TOKEN" != *"Not logged in"* ]]; then
        echo "✓  Already logged in as: $EXISTING_TOKEN"
        read -rp "   Do you want to update the token? [y/N] " UPDATE_TOKEN
        if [[ "$UPDATE_TOKEN" =~ ^[Yy]$ ]]; then
            huggingface-cli login
        fi
    else
        echo "Enter your HF token (it will be saved in ~/.cache/huggingface/):"
        huggingface-cli login
    fi
else
    echo "⚠  huggingface-cli not found in PATH — skipping login."
    echo "   Run manually: huggingface-cli login"
fi

echo ""
echo "════════════════════════════════════════"
echo "✓  Setup complete."
echo ""
echo "Activate the venv:   source .venv/bin/activate"
echo "Check:               python3 cli.py info"
echo "Test MLC:            python3 cli.py jit light"
echo ""

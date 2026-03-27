#!/usr/bin/env bash
# setup.sh — crea il venv e installa tutte le dipendenze per mlbenchmark-core.
#
# Prerequisiti:
#   - Python 3.12 (da python.org — non il Python di sistema macOS)
#   - Xcode Command Line Tools  →  xcode-select --install
#   - CMake  →  brew install cmake
#
# Uso:
#   chmod +x setup.sh && ./setup.sh
#
# Dopo il setup:
#   source .venv/bin/activate
#   python3 cli.py info

set -euo pipefail

PYTHON="${PYTHON:-python3.12}"

echo ""
echo "ML Benchmark Core — Setup"
echo "════════════════════════════════════════"

# ── Verifica Python ────────────────────────────────────────────────────────
if ! command -v "$PYTHON" &>/dev/null; then
    echo "✗  $PYTHON non trovato."
    echo "   Scarica Python 3.12 da https://www.python.org/downloads/"
    exit 1
fi

PY_VER=$("$PYTHON" --version 2>&1)
echo "▶ Python: $PY_VER"

# ── Crea venv ─────────────────────────────────────────────────────────────
if [[ -d .venv ]]; then
    echo "▶ .venv già esistente — aggiorno i pacchetti…"
else
    echo "▶ Creo .venv…"
    "$PYTHON" -m venv .venv
fi

source .venv/bin/activate
pip install --quiet --upgrade pip

# ── Pacchetti standard ─────────────────────────────────────────────────────
echo "▶ Installo dipendenze standard…"
pip install --quiet -r requirements.txt

# ── llama-cpp-python con accelerazione Metal ───────────────────────────────
echo "▶ Installo llama-cpp-python (Metal GPU)…"
CMAKE_ARGS="-DGGML_METAL=on" \
    pip install --quiet "llama-cpp-python==0.3.17" \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal

# ── mlc-llm + tvm da mlc.ai (nightly, Apple Silicon) ──────────────────────
echo "▶ Installo mlc-llm (nightly, Metal)…"
# Il pacchetto nightly include sia TVM che il runtime Metal per Apple Silicon.
# Se il comando fallisce, visita https://mlc.ai/package/ per la versione aggiornata.
pip install --quiet --pre -U \
    mlc-llm-nightly-cpu \
    mlc-ai-nightly-cpu \
    -f https://mlc.ai/wheels

# ── Hugging Face token ────────────────────────────────────────────────────
# Necessario per scaricare i modelli (alcuni sono gated su HF).
# Ottieni il tuo token su https://huggingface.co/settings/tokens
echo ""
echo "────────────────────────────────────────"
echo "Hugging Face token"
echo ""
echo "I modelli vengono scaricati da Hugging Face."
echo "Alcuni sono gated e richiedono un account HF con accesso approvato."
echo "Ottieni il tuo token su: https://huggingface.co/settings/tokens"
echo ""

if command -v huggingface-cli &>/dev/null; then
    # Controlla se c'è già un token salvato
    EXISTING_TOKEN=$(huggingface-cli whoami 2>/dev/null | head -1 || true)
    if [[ -n "$EXISTING_TOKEN" && "$EXISTING_TOKEN" != *"Not logged in"* ]]; then
        echo "✓  Già loggato come: $EXISTING_TOKEN"
        read -rp "   Vuoi aggiornare il token? [y/N] " UPDATE_TOKEN
        if [[ "$UPDATE_TOKEN" =~ ^[Yy]$ ]]; then
            huggingface-cli login
        fi
    else
        echo "Inserisci il tuo HF token (verrà salvato in ~/.cache/huggingface/):"
        huggingface-cli login
    fi
else
    echo "⚠  huggingface-cli non trovato nel PATH — salto il login."
    echo "   Esegui manualmente: huggingface-cli login"
fi

echo ""
echo "════════════════════════════════════════"
echo "✓  Setup completato."
echo ""
echo "Attiva il venv:   source .venv/bin/activate"
echo "Verifica:         python3 cli.py info"
echo "Test MLC:         python3 cli.py jit nano"
echo ""

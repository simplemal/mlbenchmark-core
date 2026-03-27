#!/usr/bin/env python3
r"""
ml-benchmark CLI — diagnosi e test mirati dei backend di inferenza.

Uso:
    python3 cli.py info                          # hardware + tier disponibili
    python3 cli.py test mlc nano                 # test completo MLC su Nano
    python3 cli.py test mlx nano                 # test MLX
    python3 cli.py test gguf nano                # test GGUF
    python3 cli.py jit nano                      # solo diagnostica JIT MLC (senza inference)
    python3 cli.py prompt "ciao" --backend mlx --tier nano  # prompt singolo

Esegui sempre con il venv dell'app:
    ~/Library/Application\ Support/MLBenchmark/venv/bin/python3 cli.py ...
"""

import sys
import os
import argparse
import platform
import time
import traceback
from pathlib import Path

# ── Aggiungi app/ al path in modo che gli import (utils, model_runner_*) funzionino ──
CLI_DIR = Path(__file__).resolve().parent
APP_DIR = CLI_DIR / "app"
sys.path.insert(0, str(APP_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

TIERS = ["nano", "entry", "standard", "advanced", "extreme"]

TIER_KEYS = {
    "nano":     "Nano__Llama-3.2-3B-Instruct",
    "entry":    "Entry__Phi-3.5-Mini-Instruct",
    "standard": "Standard__Gemma-2-9B-Instruct",
    "advanced": "Advanced__Qwen2.5-14B-Instruct",
    "extreme":  "Extreme__Qwen2.5-32B-Instruct",
}

BACKENDS = ["mlx", "gguf", "mlc"]


def models_dir() -> Path:
    """Ritorna la directory modelli — Application Support se siamo in bundle, altrimenti locale."""
    app_support = Path.home() / "Library" / "Application Support" / "MLBenchmark" / "models"
    if app_support.exists():
        return app_support
    local = CLI_DIR / "models"
    if local.exists():
        return local
    raise FileNotFoundError(
        f"Nessuna directory modelli trovata.\n"
        f"Cercato in:\n  {app_support}\n  {local}"
    )


def model_path(tier: str, backend: str) -> Path:
    key = f"{TIER_KEYS[tier]}__{backend.upper()}"
    return models_dir() / key


def hr(char="─", width=60):
    print(char * width)


def section(title: str):
    print()
    hr()
    print(f"  {title}")
    hr()


def ok(msg): print(f"  ✓  {msg}")
def fail(msg): print(f"  ✗  {msg}")
def info(msg): print(f"     {msg}")
def warn(msg): print(f"  ⚠  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Comando: info
# ─────────────────────────────────────────────────────────────────────────────

def cmd_info(_args):
    section("Sistema")
    info(f"Python      {sys.version}")
    info(f"macOS       {platform.mac_ver()[0]}  (darwin {platform.release()})")
    info(f"Arch        {platform.machine()}")

    section("Venv / pacchetti")
    info(f"Executable  {sys.executable}")
    for pkg in ["mlc_llm", "mlx", "llama_cpp", "tvm"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            ok(f"{pkg}  {ver}")
        except ImportError as e:
            fail(f"{pkg}  — {e}")

    section("Modelli su disco")
    try:
        mdir = models_dir()
        info(f"Directory: {mdir}")
        for tier in TIERS:
            for backend in BACKENDS:
                p = model_path(tier, backend)
                if p.exists():
                    size_gb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9
                    ok(f"{tier:10s} {backend.upper():5s}  {size_gb:.1f} GB  →  {p.name}")
                else:
                    info(f"{tier:10s} {backend.upper():5s}  non scaricato")
    except FileNotFoundError as e:
        fail(str(e))

    section("Hardware (da utils.py)")
    try:
        from utils import detect_apple_chip, detect_memory_gb, get_available_tiers
        chip = detect_apple_chip()
        ram  = detect_memory_gb()
        ok(f"Chip: {chip}  |  RAM: {ram} GB")
        tiers = get_available_tiers()
        ok(f"Tier disponibili: {[t['name'] for t in tiers]}")
    except Exception as e:
        fail(f"utils.py non disponibile: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Comando: jit  (solo diagnostica compilazione MLC, senza inference)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_jit(args):
    tier = args.tier.lower()
    section(f"Diagnostica JIT MLC — {tier}")

    # 1. Import
    info("1/5  Import mlc_llm…")
    try:
        import mlc_llm
        ok(f"mlc_llm importato  ({mlc_llm.__file__})")
    except ImportError as e:
        fail(f"mlc_llm non disponibile: {e}")
        return

    # 2. Import TVM
    info("2/5  Import tvm…")
    try:
        import tvm
        ok(f"tvm  {getattr(tvm, '__version__', '?')}")
        from tvm.relax.block_builder import BlockBuilder
        bb = BlockBuilder()
        ok("BlockBuilder() istanziato senza errori")
    except AttributeError as e:
        fail(f"BlockBuilder AttributeError → {e}")
        fail("Questo è il bug che causa il crash JIT su macOS 26!")
        info("Il tvm installato nel venv non è compatibile con darwin25.x")
        info("Dettagli e aggiornamenti: https://mlbenchmark.app/docs#W001")
        return
    except Exception as e:
        fail(f"TVM error: {type(e).__name__}: {e}")
        return

    # 3. Verifica path modello
    info("3/5  Verifica path modello…")
    p = model_path(tier, "mlc")
    if not p.exists():
        fail(f"Modello non trovato: {p}")
        return
    ok(f"Path: {p}")
    cfg = p / "mlc-chat-config.json"
    if cfg.exists():
        ok("mlc-chat-config.json presente")
    else:
        warn("mlc-chat-config.json mancante")

    # 4. Target detection
    info("4/5  Rilevamento target MLC…")
    try:
        from mlc_llm.interface import jit as mlc_jit
        import inspect
        ok(f"mlc_llm.interface.jit caricato da {mlc_jit.__file__}")
    except Exception as e:
        fail(f"Impossibile caricare jit.py: {e}")

    # 5. JIT compilation test
    info("5/5  Tentativo JIT compilation (questa è la parte critica)…")
    info("     Stdout/stderr del processo di compilazione seguono:")
    print()
    try:
        from mlc_llm.interface.jit import jit as do_jit
        t0 = time.time()
        result = do_jit(
            model_path=p,
            overrides={},
            device="metal",
        )
        elapsed = time.time() - t0
        ok(f"JIT completato in {elapsed:.1f}s  →  {result}")
    except RuntimeError as e:
        elapsed = time.time() - t0
        fail(f"JIT fallito dopo {elapsed:.1f}s: {e}")
        traceback.print_exc()
    except Exception as e:
        fail(f"Errore inatteso: {type(e).__name__}: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Comando: test  (carica il modello + warm-up + prompt di prova)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_test(args):
    tier    = args.tier.lower()
    backend = args.backend.lower()
    section(f"Test {backend.upper()} — {tier}")

    p = model_path(tier, backend)
    if not p.exists():
        fail(f"Modello non trovato: {p}")
        info("Scarica il modello prima avviando il benchmark dall'app.")
        return

    # Import runner corretto
    info(f"Caricamento runner {backend.upper()}…")
    try:
        if backend == "mlx":
            import model_runner_mlx as runner
            repo = str(p)
        elif backend == "gguf":
            import model_runner_gguf as runner
            gguf_files = list(p.glob("*.gguf"))
            if not gguf_files:
                fail(f"Nessun file .gguf in {p}")
                return
            repo = str(gguf_files[0])
        elif backend == "mlc":
            import model_runner_mlc as runner
            repo = str(p.resolve())
            # Controlla compatibilità JIT prima di procedere
            if getattr(runner, "MLC_JIT_ERROR", None):
                fail(f"MLC non compatibile su questo Mac")
                info(f"  {runner.MLC_JIT_ERROR}")
                warn("Il backend MLC viene saltato su questa configurazione.")
                info("Per dettagli: https://mlbenchmark.app/docs#W001")
                return
        else:
            fail(f"Backend sconosciuto: {backend}")
            return
        ok("Runner importato")
    except Exception as e:
        fail(f"Import runner fallito: {type(e).__name__}: {e}")
        traceback.print_exc()
        return

    # Warm-up
    WARM_UP = "Say hello in one word."
    info(f"Warm-up (repo={repo})…")
    t0 = time.time()
    try:
        out, reason, ntok = runner.run_prompt(WARM_UP, 16, repo, ctx_max=4096)
        elapsed = time.time() - t0
        ok(f"Warm-up OK in {elapsed:.1f}s  →  '{out[:80].strip()}'  ({reason}, {ntok} tok)")
    except Exception as e:
        elapsed = time.time() - t0
        fail(f"Warm-up fallito dopo {elapsed:.1f}s: {type(e).__name__}: {e}")
        traceback.print_exc()
        return

    # Prompt di prova
    TEST_PROMPT = "What is 2 + 2? Answer with just the number."
    info(f"Prompt: '{TEST_PROMPT}'")
    t0 = time.time()
    try:
        out, reason, ntok = runner.run_prompt(TEST_PROMPT, 64, repo, ctx_max=4096)
        elapsed = time.time() - t0
        tps = ntok / elapsed if elapsed > 0 else 0
        ok(f"Output: '{out[:120].strip()}'")
        ok(f"Velocità: {tps:.1f} t/s  ({ntok} tok in {elapsed:.1f}s, finish={reason})")
    except Exception as e:
        elapsed = time.time() - t0
        fail(f"Prompt fallito dopo {elapsed:.1f}s: {type(e).__name__}: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Comando: prompt  (prompt singolo libero)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_prompt(args):
    tier    = args.tier.lower()
    backend = args.backend.lower()
    text    = args.text
    section(f"Prompt singolo — {backend.upper()} / {tier}")

    p = model_path(tier, backend)
    if not p.exists():
        fail(f"Modello non trovato: {p}")
        return

    try:
        if backend == "mlx":
            import model_runner_mlx as runner
            repo = str(p)
        elif backend == "gguf":
            import model_runner_gguf as runner
            gguf_files = list(p.glob("*.gguf"))
            if not gguf_files:
                fail("Nessun .gguf trovato")
                return
            repo = str(gguf_files[0])
        elif backend == "mlc":
            import model_runner_mlc as runner
            repo = str(p.resolve())
        else:
            fail(f"Backend sconosciuto: {backend}")
            return
    except Exception as e:
        fail(f"Import fallito: {e}")
        return

    info(f"Prompt: {text}")
    t0 = time.time()
    try:
        out, reason, ntok = runner.run_prompt(text, args.max_tokens, repo, ctx_max=4096)
        elapsed = time.time() - t0
        tps = ntok / elapsed if elapsed > 0 else 0
        print()
        print(out)
        print()
        ok(f"{tps:.1f} t/s  ({ntok} tok in {elapsed:.1f}s, finish={reason})")
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ml-benchmark CLI — diagnosi e test backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # info
    sub.add_parser("info", help="Hardware, pacchetti, modelli su disco")

    # jit
    p_jit = sub.add_parser("jit", help="Diagnostica JIT MLC step by step")
    p_jit.add_argument("tier", choices=TIERS)

    # test
    p_test = sub.add_parser("test", help="Carica modello, warm-up, prompt di prova")
    p_test.add_argument("backend", choices=BACKENDS)
    p_test.add_argument("tier", choices=TIERS)

    # prompt
    p_prompt = sub.add_parser("prompt", help="Prompt singolo libero")
    p_prompt.add_argument("text")
    p_prompt.add_argument("--backend", choices=BACKENDS, required=True)
    p_prompt.add_argument("--tier",    choices=TIERS,    required=True)
    p_prompt.add_argument("--max-tokens", type=int, default=256, dest="max_tokens")

    args = parser.parse_args()

    dispatch = {"info": cmd_info, "jit": cmd_jit, "test": cmd_test, "prompt": cmd_prompt}
    dispatch[args.cmd](args)
    print()


if __name__ == "__main__":
    main()

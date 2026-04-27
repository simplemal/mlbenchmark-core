#!/usr/bin/env python3
r"""
ml-benchmark CLI — diagnostics and targeted tests for the inference backends.

Usage:
    python3 cli.py info                            # hardware + available tiers
    python3 cli.py test mlc light                  # full MLC test on Light
    python3 cli.py test mlx flash                  # MLX test
    python3 cli.py test gguf blaze                 # GGUF test
    python3 cli.py jit light                       # MLC JIT diagnostics only
    python3 cli.py prompt "hi" --backend mlx --tier light   # single prompt

Tiers are loaded dynamically from shared/repository.json
(currently: light, speed, flash, blaze, ultra).

Always run with the app's venv:
    ~/Library/Application\ Support/MLBenchmark/venv/bin/python3 cli.py ...
"""

import sys
import os
import argparse
import contextlib
import platform
import time
import traceback
from pathlib import Path

# ── Add app/ to sys.path so imports (utils, model_runner_*) work ──────────
CLI_DIR = Path(__file__).resolve().parent
APP_DIR = CLI_DIR / "app"
sys.path.insert(0, str(APP_DIR))


@contextlib.contextmanager
def _suppress_native_io():
    """Redirect C-level stdout/stderr to /dev/null while inside the block.
    Used to silence benign TVM/LLVM logs printed during mlc_llm import
    (e.g. 'Using LLVM 19.1.7 with -mcpu=apple-m1 is not valid…').
    Python-level exceptions still propagate normally."""
    # Flush any buffered Python output BEFORE swapping the fds, otherwise
    # pending writes get redirected to /dev/null and lost.
    sys.stdout.flush()
    sys.stderr.flush()

    saved_out = os.dup(1)
    saved_err = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        # Flush again so anything written inside the block (which went to
        # devnull at the C level) is also drained at the Python level.
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_tier_map() -> dict:
    """Build {lower_tier_id: 'TierName__ModelName'} dynamically from
    shared/repository.json. Insertion order from the JSON is preserved
    (Light, Speed, Flash, Blaze, Ultra) so argparse help reads naturally."""
    import json
    repo_path = Path(__file__).resolve().parent / "shared" / "repository.json"
    if not repo_path.exists():
        return {}
    try:
        with open(repo_path, encoding="utf-8") as f:
            repo = json.load(f)
    except Exception:
        return {}
    out = {}
    for key in repo:
        parts = key.split("__")
        if len(parts) >= 3:
            tier_id = parts[0]                     # e.g. "Light"
            prefix  = "__".join(parts[:2])         # e.g. "Light__Qwen2.5-0.5B-Instruct"
            out.setdefault(tier_id.lower(), prefix)
    return out


TIER_KEYS = _load_tier_map()
TIERS     = list(TIER_KEYS.keys())  # e.g. ["light", "speed", "flash", "blaze", "ultra"]

BACKENDS = ["mlx", "gguf", "mlc"]


def models_dir() -> Path:
    """Return the models directory — Application Support if bundled, otherwise local."""
    app_support = Path.home() / "Library" / "Application Support" / "MLBenchmark" / "models"
    if app_support.exists():
        return app_support
    local = CLI_DIR / "models"
    if local.exists():
        return local
    raise FileNotFoundError(
        f"No models directory found.\n"
        f"Searched in:\n  {app_support}\n  {local}"
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
# Command: info
# ─────────────────────────────────────────────────────────────────────────────

def cmd_info(_args):
    section("System")
    info(f"Python      {sys.version}")
    info(f"macOS       {platform.mac_ver()[0]}  (darwin {platform.release()})")
    info(f"Arch        {platform.machine()}")

    section("Venv / packages")
    info(f"Executable  {sys.executable}")
    for pkg in ["mlc_llm", "mlx", "llama_cpp", "tvm"]:
        try:
            with _suppress_native_io():
                mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            ok(f"{pkg}  {ver}")
        except ImportError as e:
            fail(f"{pkg}  — {e}")

    section("Models on disk")
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
                    info(f"{tier:10s} {backend.upper():5s}  not downloaded")
    except FileNotFoundError as e:
        fail(str(e))

    section("Hardware (from utils.py)")
    try:
        from utils import detect_apple_chip, detect_memory_gb, get_available_tiers
        chip = detect_apple_chip()
        ram  = detect_memory_gb()
        ok(f"Chip: {chip}  |  RAM: {ram} GB")
        tiers = get_available_tiers()
        ok(f"Available tiers: {[t['name'] for t in tiers]}")
    except Exception as e:
        fail(f"utils.py unavailable: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Command: jit  (MLC compile diagnostics only — no inference)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_jit(args):
    tier = args.tier.lower()
    section(f"MLC JIT diagnostics — {tier}")

    # 1. Import
    info("1/5  Import mlc_llm…")
    try:
        with _suppress_native_io():
            import mlc_llm
        ok(f"mlc_llm imported  ({mlc_llm.__file__})")
    except ImportError as e:
        fail(f"mlc_llm not available: {e}")
        return

    # 2. Import TVM
    info("2/5  Import tvm…")
    try:
        with _suppress_native_io():
            import tvm
            from tvm.relax.block_builder import BlockBuilder
            bb = BlockBuilder()
        ok(f"tvm  {getattr(tvm, '__version__', '?')}")
        ok("BlockBuilder() instantiated without errors")
    except AttributeError as e:
        fail(f"BlockBuilder AttributeError → {e}")
        fail("This is the bug that causes JIT crashes on macOS 26!")
        info("The tvm installed in the venv is incompatible with darwin25.x")
        info("Details and updates: https://mlbenchmark.app/docs#W001")
        return
    except Exception as e:
        fail(f"TVM error: {type(e).__name__}: {e}")
        return

    # 3. Verify model path
    info("3/5  Verify model path…")
    p = model_path(tier, "mlc")
    if not p.exists():
        fail(f"Model not found: {p}")
        return
    ok(f"Path: {p}")
    cfg = p / "mlc-chat-config.json"
    if cfg.exists():
        ok("mlc-chat-config.json present")
    else:
        warn("mlc-chat-config.json missing")

    # 4. Target detection
    info("4/5  MLC target detection…")
    try:
        from mlc_llm.interface import jit as mlc_jit
        import inspect
        ok(f"mlc_llm.interface.jit loaded from {mlc_jit.__file__}")
    except Exception as e:
        fail(f"Cannot load jit.py: {e}")

    # 5. JIT compilation test
    info("5/5  Attempting JIT compilation (this is the critical step)…")
    info("     Stdout/stderr from the compiler follow:")
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
        ok(f"JIT completed in {elapsed:.1f}s  →  {result}")
    except RuntimeError as e:
        elapsed = time.time() - t0
        fail(f"JIT failed after {elapsed:.1f}s: {e}")
        traceback.print_exc()
    except Exception as e:
        fail(f"Unexpected error: {type(e).__name__}: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Command: test  (load model + warm-up + sample prompt)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_test(args):
    tier    = args.tier.lower()
    backend = args.backend.lower()
    section(f"Test {backend.upper()} — {tier}")

    p = model_path(tier, backend)
    if not p.exists():
        fail(f"Model not found: {p}")
        info("Download the model first by running the benchmark from the app.")
        return

    # Import the right runner
    info(f"Loading {backend.upper()} runner…")
    try:
        if backend == "mlx":
            import model_runner_mlx as runner
            repo = str(p)
        elif backend == "gguf":
            import model_runner_gguf as runner
            gguf_files = list(p.glob("*.gguf"))
            if not gguf_files:
                fail(f"No .gguf file in {p}")
                return
            repo = str(gguf_files[0])
        elif backend == "mlc":
            with _suppress_native_io():
                import model_runner_mlc as runner
            repo = str(p.resolve())
            # Check JIT compatibility before proceeding
            if getattr(runner, "MLC_JIT_ERROR", None):
                fail(f"MLC not compatible on this Mac")
                info(f"  {runner.MLC_JIT_ERROR}")
                warn("The MLC backend is skipped on this configuration.")
                info("Details: https://mlbenchmark.app/docs#W001")
                return
        else:
            fail(f"Unknown backend: {backend}")
            return
        ok("Runner imported")
    except Exception as e:
        fail(f"Runner import failed: {type(e).__name__}: {e}")
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
        fail(f"Warm-up failed after {elapsed:.1f}s: {type(e).__name__}: {e}")
        traceback.print_exc()
        return

    # Sample prompt
    TEST_PROMPT = "What is 2 + 2? Answer with just the number."
    info(f"Prompt: '{TEST_PROMPT}'")
    t0 = time.time()
    try:
        out, reason, ntok = runner.run_prompt(TEST_PROMPT, 64, repo, ctx_max=4096)
        elapsed = time.time() - t0
        tps = ntok / elapsed if elapsed > 0 else 0
        ok(f"Output: '{out[:120].strip()}'")
        ok(f"Speed: {tps:.1f} t/s  ({ntok} tok in {elapsed:.1f}s, finish={reason})")
    except Exception as e:
        elapsed = time.time() - t0
        fail(f"Prompt failed after {elapsed:.1f}s: {type(e).__name__}: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Command: prompt  (free-form single prompt)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_prompt(args):
    tier    = args.tier.lower()
    backend = args.backend.lower()
    text    = args.text
    section(f"Single prompt — {backend.upper()} / {tier}")

    p = model_path(tier, backend)
    if not p.exists():
        fail(f"Model not found: {p}")
        return

    try:
        if backend == "mlx":
            import model_runner_mlx as runner
            repo = str(p)
        elif backend == "gguf":
            import model_runner_gguf as runner
            gguf_files = list(p.glob("*.gguf"))
            if not gguf_files:
                fail("No .gguf file found")
                return
            repo = str(gguf_files[0])
        elif backend == "mlc":
            with _suppress_native_io():
                import model_runner_mlc as runner
            repo = str(p.resolve())
        else:
            fail(f"Unknown backend: {backend}")
            return
    except Exception as e:
        fail(f"Import failed: {e}")
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
        description="ml-benchmark CLI — backend diagnostics and tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # info
    sub.add_parser("info", help="Hardware, packages, models on disk")

    # jit
    p_jit = sub.add_parser("jit", help="Step-by-step MLC JIT diagnostics")
    p_jit.add_argument("tier", choices=TIERS)

    # test
    p_test = sub.add_parser("test", help="Load model, warm-up, sample prompt")
    p_test.add_argument("backend", choices=BACKENDS)
    p_test.add_argument("tier", choices=TIERS)

    # prompt
    p_prompt = sub.add_parser("prompt", help="Free-form single prompt")
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

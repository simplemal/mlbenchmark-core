#!/usr/bin/env python3
"""
benchmark_runner.py — subprocess interface for MLBenchmark Xcode app.

Communicates via JSON lines on stdout (one JSON object per line).
All debug/verbose output goes to stderr so it doesn't pollute the protocol.

Usage:
  python3 benchmark_runner.py --info
      Emits hardware info and available tiers, then exits.

  python3 benchmark_runner.py --tiers Entry Standard Advanced
      Runs the selected tiers and streams progress events.

Cancel: write "cancel" to stdin at any time during a run.

JSON event types:
  {"event": "hardware",          "chip": str, "ram_gb": int, "available_ram_gb": float,
                                 "available_disk_gb": float, "cpu_cores": int, "gpu_cores": int}
  {"event": "tiers",             "available": [str, ...], "all": [{name, min_ram_gb, model_size}, ...]}
  {"event": "ready"}

  {"event": "tier_start",        "tier": str, "model": str}
  {"event": "download_start",    "tier": str, "backend": str, "key": str, "size_gb": float}
  {"event": "download_progress", "key": str, "downloaded_gb": float, "total_gb": float}
  {"event": "download_done",     "key": str}
  {"event": "download_failed",   "key": str, "message": str}
  {"event": "backend_start",     "tier": str, "backend": str}
  {"event": "prompt_result",     "tier": str, "backend": str, "tps": float,
                                 "prompt_index": int, "prompt_total": int}
  {"event": "backend_done",      "tier": str, "backend": str, "tps": float, "success": bool}
  {"event": "tier_done",         "tier": str}
  {"event": "complete",          "file_path": str, "tier_results": {tier: {backend: tps}}}
  {"event": "cancelled"}
  {"event": "error",             "message": str}
"""

import json
import sys
import os
import gc
import math
import time
import signal
import threading
import queue as _queue
import argparse
from statistics import mean
from pathlib import Path

# ── Redirect stderr early so debug prints from imported modules go there ──────
# (stdout is reserved for JSON protocol)
_real_print = print

def _stderr_print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("flush", True)
    _real_print(*args, **kwargs)

import builtins
builtins.print = _stderr_print  # all module-level prints → stderr

# ── App imports (after redirecting print) ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import globals_state as gs
from utils import (
    detect_apple_chip,
    detect_mac_model,
    detect_memory_gb,
    detect_gpu_cores,
    detect_ne_cores,
    detect_os_name,
    detect_os_version,
    get_available_ram_gb,
    get_available_tiers,
    get_benchmark_plan,
    get_models_for_tier,
    get_models,
    get_models_dir,
    get_free_disk_space_gb,
    get_folder_size,
    get_prompts,
    get_token_steps,
    get_prompt_warm_up,
    is_model_ready,
    BENCHMARK_TIERS,
)
from download_model import download_model, cleanup_partial_downloads
from benchmark_analyzer import BenchmarkAnalyzer
from model_runner_timeout import model_runner_timeout

# ── JSON emit ─────────────────────────────────────────────────────────────────

def emit(obj: dict):
    """Write one JSON line to stdout."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


# ── Model runners ─────────────────────────────────────────────────────────────

from functools import lru_cache

@lru_cache(maxsize=None)
def get_runner(fmt: str):
    if fmt == "MLX":
        import model_runner_mlx as r
    elif fmt == "GGUF":
        import model_runner_gguf as r
    elif fmt == "MLC":
        import model_runner_mlc as r
    else:
        raise ValueError(f"Unknown format: {fmt}")
    return r


_MEMORY_ERROR_KEYWORDS = (
    "memory", "malloc", "unable to allocate", "out of memory",
    "allocation failed", "cannot allocate", "enomem",
)


def _is_memory_error(exc: Exception) -> bool:
    """Return True if the exception looks like an out-of-memory condition."""
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in _MEMORY_ERROR_KEYWORDS)


def _error_tag(exc: Exception) -> str:
    """Return a prefixed error string: 'memory_insufficient: …' or 'TypeName: …'."""
    if _is_memory_error(exc):
        return f"memory_insufficient: {exc}"
    return f"{type(exc).__name__}: {exc}"


def release_memory():
    for fmt in ("MLX", "GGUF", "MLC"):
        try:
            get_runner(fmt).release_model()
        except Exception:
            pass
    gc.collect()


# ── GGUF file selection (mirror of BenchmarkML.select_gguf_file) ──────────────

_QUANT_MAP = {
    "4": ["Q4_K_M", "Q4_K_S", "Q4_0"],
    "8": ["Q8_0"],
    "F16": ["F16"],
}

def _select_gguf_file(model_dir: Path, quantization: str) -> str:
    gguf_files = [f for f in os.listdir(model_dir) if f.endswith(".gguf")]
    if not gguf_files:
        raise FileNotFoundError(f"No GGUF files in {model_dir}")
    if len(gguf_files) == 1:
        return os.path.join(model_dir, gguf_files[0])
    quant = str(quantization).upper()
    prefs = _QUANT_MAP.get(quant, [quant])
    for pref in prefs:
        for f in gguf_files:
            if pref in f.upper():
                return os.path.join(model_dir, f)
    return os.path.join(model_dir, gguf_files[0])


# ── Cancel + response routing via stdin ───────────────────────────────────────
# Single thread owns stdin reads. "cancel" → sets flag immediately.
# Everything else (e.g. "delete", "skip") → queued for the main thread.

_stdin_queue: _queue.Queue = _queue.Queue()


def _watch_stdin():
    try:
        for line in sys.stdin:
            stripped = line.strip().lower()
            if stripped in ("cancel", "cancel_keep", "cancel_delete"):
                gs.cancel_requested = True
                gs.cancel_delete_downloads = (stripped == "cancel_delete")
            else:
                _stdin_queue.put(stripped)
    except Exception:
        pass


threading.Thread(target=_watch_stdin, daemon=True).start()


def _read_stdin_response(timeout: float = 60) -> str:
    """Block until a non-cancel stdin line arrives (or timeout). Returns "" on timeout."""
    try:
        return _stdin_queue.get(timeout=timeout)
    except _queue.Empty:
        return ""


# ── Info mode ─────────────────────────────────────────────────────────────────

def cmd_info():
    cpu_cores = os.cpu_count() or 0
    try:
        gpu_cores = int(detect_gpu_cores())
    except Exception:
        gpu_cores = 0
    ne_cores = detect_ne_cores()

    mac = detect_mac_model()
    emit({
        "event": "hardware",
        "chip": detect_apple_chip(),
        "mac_name": mac["name"],
        "mac_chip": mac["chip"],
        "mac_year": mac["year"],
        "ram_gb": int(detect_memory_gb()),
        "available_ram_gb": round(get_available_ram_gb(), 1),
        "available_disk_gb": round(get_free_disk_space_gb(), 1),
        "cpu_cores": cpu_cores,
        "gpu_cores": gpu_cores,
        "ne_cores": ne_cores,
        "os_name": detect_os_name(),
        "os_version": detect_os_version(),
    })

    available = get_available_tiers()
    emit({
        "event": "tiers",
        "available": [t["name"] for t in available],
        "all": BENCHMARK_TIERS,
    })

    emit({"event": "ready"})


# ── Benchmark single backend ──────────────────────────────────────────────────

def _log_ram(label: str):
    """Print available RAM at a given point."""
    try:
        available = get_available_ram_gb()
        print(f"[RAM] {label}: {available:.1f} GB available")
    except Exception as e:
        print(f"[RAM] Could not read RAM at '{label}': {e}")


def _run_backend(tier_name: str, key: str, model: dict) -> tuple[float, bool, dict]:
    """Run one backend. Returns (avg_tps, success, prompt_details). Emits prompt_result events."""
    fmt = model["format"]
    if gs.cancel_requested or gs.ram_tier_drop:
        print(f"[{fmt}] _run_backend skipped before start (cancel={gs.cancel_requested}, ram_drop={gs.ram_tier_drop})")
        return 0.0, False, {}

    try:
        runner = get_runner(fmt)
    except Exception as e:
        emit({"event": "backend_done", "tier": tier_name, "backend": fmt, "tps": 0.0, "success": False,
              "error": _error_tag(e)})
        return 0.0, False, {}

    ctx_max = min(model.get("ctx_max", 4096), 8192)
    model_dir = get_models_dir() / key

    # ── MLC JIT compatibility check ───────────────────────────────────────────
    if fmt == "MLC":
        jit_err = getattr(runner, "MLC_JIT_ERROR", None)
        if jit_err:
            import shutil
            if model_dir.exists():
                shutil.rmtree(str(model_dir), ignore_errors=True)
                print(f"[MLC] Model deleted (JIT incompatible): {model_dir}")
            emit({"event": "backend_done", "tier": tier_name, "backend": fmt,
                  "tps": 0.0, "success": False, "error": jit_err})
            return 0.0, False, {}
    # ─────────────────────────────────────────────────────────────────────────

    try:
        if fmt == "MLX":
            repo = str(model_dir)
        elif fmt == "MLC":
            repo = os.path.abspath(str(model_dir))
        else:
            repo = _select_gguf_file(model_dir, model.get("quantization"))
    except Exception as e:
        print(f"[{fmt}] Repo setup failed: {e}")
        emit({"event": "backend_done", "tier": tier_name, "backend": fmt, "tps": 0.0, "success": False,
              "error": f"Model not found: {e}"})
        return 0.0, False, {}

    basic_tokens, _ = get_token_steps()
    token_count = basic_tokens[0] if basic_tokens else 128

    # Warm-up
    _log_ram(f"{fmt} before warm-up")
    try:
        runner.run_prompt(get_prompt_warm_up(), 16, repo, ctx_max)
        _log_ram(f"{fmt} after warm-up")
    except Exception as e:
        import traceback
        print(f"[{fmt}] Warm-up FAILED: {type(e).__name__}: {e}")
        print(f"[{fmt}] Warm-up traceback:\n{traceback.format_exc()}")
        _log_ram(f"{fmt} after warm-up failure")
        emit({"event": "backend_done", "tier": tier_name, "backend": fmt, "tps": 0.0, "success": False,
              "error": _error_tag(e)})
        return 0.0, False, {}

    if gs.cancel_requested or gs.ram_tier_drop:
        print(f"[{fmt}] Stopped after warm-up (cancel={gs.cancel_requested}, ram_drop={gs.ram_tier_drop})")
        return 0.0, False, {}

    # Post-warmup RAM check: if RAM is critically low, the model loaded but
    # there's not enough headroom for inference — skip to avoid native crashes.
    post_warmup_ram = get_available_ram_gb()
    if post_warmup_ram < 0.5:
        print(f"[{fmt}] RAM critically low after warm-up ({post_warmup_ram:.1f} GB) — skipping prompts to avoid crash")
        _log_ram(f"{fmt} skipped (critically low RAM)")
        try:
            runner.release_model()
        except Exception:
            pass
        emit({"event": "backend_done", "tier": tier_name, "backend": fmt, "tps": 0.0, "success": False,
              "error": f"memory_insufficient: only {post_warmup_ram:.1f} GB RAM free after loading model"})
        return 0.0, False, {}

    prompts = get_prompts("basic")
    total = len(prompts)
    speeds = []
    prompt_details: dict = {}

    for idx, (prompt_key, prompt) in enumerate(prompts.items()):
        if gs.cancel_requested or gs.ram_tier_drop:
            print(f"[{fmt}] Loop interrupted before prompt {idx+1}/{total} '{prompt_key}' "
                  f"(cancel={gs.cancel_requested}, ram_drop={gs.ram_tier_drop})")
            _log_ram(f"{fmt} at loop interruption")
            break

        gs.current_prompt_index = idx + 1
        gs.current_task = prompt_key
        _log_ram(f"{fmt} before prompt {idx+1}/{total} '{prompt_key}'")

        try:
            start = time.time()
            output, finish_reason, real_tokens = model_runner_timeout(
                runner.run_prompt, 60, prompt, token_count, repo, ctx_max
            )
            elapsed = time.time() - start

            if real_tokens > 0 and finish_reason not in ("timeout", "error"):
                tps = real_tokens / elapsed
                speeds.append(tps)
                gs.current_tps = tps
                print(f"[{fmt}] prompt {idx+1}/{total} '{prompt_key}': {tps:.2f} t/s "
                      f"({real_tokens} tokens in {elapsed:.2f}s, finish={finish_reason})")
                prompt_details[prompt_key] = {
                    "tps": round(tps, 2),
                    "elapsed": round(elapsed, 3),
                    "real_tokens": real_tokens,
                    "finish_reason": finish_reason,
                }
                emit({
                    "event": "prompt_result",
                    "tier": tier_name,
                    "backend": fmt,
                    "tps": round(tps, 2),
                    "prompt_index": idx + 1,
                    "prompt_total": total,
                    "prompt_name": prompt_key,
                })
            elif finish_reason == "timeout":
                print(f"[{fmt}] prompt {idx+1}/{total} '{prompt_key}': TIMEOUT after {elapsed:.1f}s")
                _log_ram(f"{fmt} after timeout on '{prompt_key}'")
                # First prompt timeout = backend can't run properly, abort early
                if idx == 0:
                    print(f"[{fmt}] First prompt timed out — skipping remaining prompts")
                    break
            elif finish_reason == "error":
                print(f"[{fmt}] prompt {idx+1}/{total} '{prompt_key}': ERROR returned by runner "
                      f"(real_tokens={real_tokens}, elapsed={elapsed:.2f}s)")
                _log_ram(f"{fmt} after error on '{prompt_key}'")
            else:
                print(f"[{fmt}] prompt {idx+1}/{total} '{prompt_key}': SKIPPED "
                      f"(finish={finish_reason}, real_tokens={real_tokens})")

        except Exception as e:
            import traceback
            print(f"[{fmt}] prompt {idx+1}/{total} '{prompt_key}': EXCEPTION {type(e).__name__}: {e}")
            print(f"[{fmt}] Exception traceback:\n{traceback.format_exc()}")
            _log_ram(f"{fmt} after exception on '{prompt_key}'")

    avg_tps = round(mean(speeds), 2) if speeds else 0.0
    success = len(speeds) > 0
    print(f"[{fmt}] backend finished: {len(speeds)}/{total} prompts succeeded, "
          f"avg_tps={avg_tps}, success={success}")
    _log_ram(f"{fmt} after backend completed")
    return avg_tps, success, prompt_details


# ── Run mode ──────────────────────────────────────────────────────────────────

def _migrate_model_dirs():
    """Rename legacy tier folders (Nano→Light, Entry→Speed, etc.) so existing
    downloads are found under the new names. Runs once, harmless if already done."""
    renames = {"Nano": "Light", "Entry": "Speed", "Standard": "Flash",
               "Advanced": "Blaze", "Extreme": "Ultra"}
    base = get_models_dir()
    if not base.exists():
        return
    for old_prefix, new_prefix in renames.items():
        for d in base.iterdir():
            if d.is_dir() and d.name.startswith(old_prefix + "__"):
                new_name = new_prefix + "__" + d.name.split("__", 1)[1]
                new_path = base / new_name
                if not new_path.exists():
                    d.rename(new_path)
                    print(f"[MIGRATE] {d.name} → {new_name}")


def cmd_run_backend(tier_name: str, backend: str):
    """Run a single backend for a single tier. One process per backend — if it
    crashes, the frontend can launch the next one independently."""
    _migrate_model_dirs()
    gs.cancel_requested = False
    gs.ram_tier_drop = False

    tier_def = next((t for t in BENCHMARK_TIERS if t["name"] == tier_name), None)
    if tier_def is None:
        emit({"event": "error", "message": f"Unknown tier: {tier_name}"})
        return

    models = get_models_for_tier(tier_name)
    key = None
    model = None
    for k, m in models.items():
        if m["format"] == backend:
            key, model = k, m
            break

    if not key:
        emit({"event": "error", "message": f"No {backend} model found for {tier_name}"})
        return

    fmt = backend
    gs.current_tier = tier_name

    # Pre-run RAM check: usa size_gb del modello + 50% buffer,
    # non min_ram_gb del tier (che è il minimo totale del computer, non la RAM libera necessaria).
    available_ram = get_available_ram_gb()
    model_size_gb = float(model.get("size_gb", tier_def.get("min_ram_gb", 1)))
    ram_needed = model_size_gb * 1.5
    if available_ram < ram_needed:
        emit({"event": "backend_done", "tier": tier_name, "backend": fmt,
              "tps": 0.0, "success": False,
              "error": f"memory_insufficient: {available_ram:.1f} GB free, need {ram_needed:.1f} GB"})
        return

    # MLC JIT pre-check
    if fmt == "MLC":
        _mlc = get_runner("MLC")
        _jit_err = getattr(_mlc, "MLC_JIT_ERROR", None)
        if _jit_err:
            import shutil
            _model_dir = get_models_dir() / key
            if _model_dir.exists():
                shutil.rmtree(str(_model_dir), ignore_errors=True)
            emit({"event": "backend_done", "tier": tier_name, "backend": fmt,
                  "tps": 0.0, "success": False, "error": f"mlc_jit_incompatible: {_jit_err}"})
            return

    model_name = model.get("name", tier_name)
    emit({"event": "tier_start", "tier": tier_name, "model": model_name})

    # Download if needed
    if not is_model_ready(key):
        emit({"event": "download_start", "tier": tier_name, "backend": fmt,
              "key": key, "size_gb": model.get("size_gb", 0)})

        def dl_progress(downloaded_bytes, total_bytes, _key=key):
            emit({
                "event": "download_progress",
                "key": _key,
                "downloaded_gb": round(downloaded_bytes / (1024**3), 2),
                "total_gb": round(total_bytes / (1024**3), 2) if total_bytes > 0 else model.get("size_gb", 0),
            })

        dl_result = download_model(key, dl_progress)

        if gs.cancel_requested:
            cleanup_partial_downloads([key])
            emit({"event": "cancelled"})
            return

        if not dl_result["success"]:
            emit({"event": "download_failed", "key": key, "message": dl_result["message"]})
            emit({"event": "backend_done", "tier": tier_name, "backend": fmt,
                  "tps": 0.0, "success": False})
            return

        emit({"event": "download_done", "key": key})

    if gs.cancel_requested:
        emit({"event": "cancelled"})
        return

    # Run
    emit({"event": "backend_start", "tier": tier_name, "backend": fmt})
    avg_tps, success, prompt_details = _run_backend(tier_name, key, model)

    release_memory()

    # Emit result
    emit({"event": "backend_done", "tier": tier_name, "backend": fmt,
          "tps": avg_tps, "success": success})

    if gs.cancel_requested:
        if gs.cancel_delete_downloads:
            import shutil
            model_dir = get_models_dir() / key
            if model_dir.exists():
                try:
                    shutil.rmtree(model_dir)
                except Exception:
                    pass
        emit({"event": "cancelled"})
        return

    # Emit complete with this single backend's data
    emit({
        "event": "backend_complete",
        "tier": tier_name,
        "backend": fmt,
        "tps": avg_tps,
        "success": success,
        "prompts": prompt_details,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

def cmd_jit_check():
    """Emits {"event":"jit_check","mlc_compatible":bool,"error":str|null} and exits."""
    from model_runner_mlc import MLC_JIT_ERROR, MLC_AVAILABLE
    compatible = MLC_AVAILABLE and MLC_JIT_ERROR is None
    emit({
        "event": "jit_check",
        "mlc_compatible": compatible,
        "error": MLC_JIT_ERROR if MLC_JIT_ERROR else (None if MLC_AVAILABLE else "MLC not installed"),
    })


def cmd_save(file_path=None):
    """Read aggregated results JSON from file (or stdin as fallback) and save to CSV."""
    if file_path:
        with open(file_path, "r") as f:
            raw = f.read()
    else:
        import sys
        raw = sys.stdin.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        emit({"event": "error", "message": f"Invalid JSON: {e}"})
        return

    run_data = data.get("run_data", {})
    duration = data.get("duration", 0)

    tier_results = BenchmarkAnalyzer.compute_tier_results(run_data)
    if not tier_results:
        emit({"event": "error", "message": "No successful benchmark runs completed."})
        return

    prompt_results = {}
    for t_name, backends in run_data.items():
        prompt_results[t_name] = {}
        for bk, bk_data in backends.items():
            if bk_data.get("success") and bk_data.get("tps", 0) > 0:
                prompt_results[t_name][bk] = bk_data.get("prompts", {})

    scores = BenchmarkAnalyzer.compute_scores(tier_results)
    file_path, benchmark_id = BenchmarkAnalyzer.save_result(tier_results, prompt_results, duration)
    emit({
        "event": "complete",
        "file_path": str(file_path),
        "tier_results": tier_results,
        "scores": scores,
        "prompt_results": prompt_results,
        "benchmark_id": benchmark_id,
        "duration_seconds": round(duration),
    })


def main():
    parser = argparse.ArgumentParser(description="MLBenchmark subprocess runner")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--info", action="store_true",
                      help="Emit hardware info and available tiers, then exit")
    mode.add_argument("--run", nargs=2, metavar=("TIER", "BACKEND"),
                      help="Run a single backend for a single tier (e.g. --run Light MLX)")
    mode.add_argument("--save", nargs="?", const=True, default=None,
                      metavar="FILE",
                      help="Save results from JSON file (or stdin if no file given)")
    mode.add_argument("--jit-check", action="store_true",
                      help="Check MLC JIT compatibility and exit")
    args = parser.parse_args()

    if args.info:
        cmd_info()
    elif args.jit_check:
        cmd_jit_check()
    elif args.run:
        cmd_run_backend(args.run[0], args.run[1])
    elif args.save is not None:
        file_path = args.save if isinstance(args.save, str) else None
        cmd_save(file_path)


if __name__ == "__main__":
    main()

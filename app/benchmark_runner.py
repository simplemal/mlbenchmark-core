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
    start_ram_monitor,
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
              "error": f"{type(e).__name__}: {e}"})
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
              "error": f"{type(e).__name__}: {e}"})
        return 0.0, False, {}

    if gs.cancel_requested or gs.ram_tier_drop:
        print(f"[{fmt}] Stopped after warm-up (cancel={gs.cancel_requested}, ram_drop={gs.ram_tier_drop})")
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
    emit({"event": "backend_done", "tier": tier_name, "backend": fmt,
          "tps": avg_tps, "success": success})
    return avg_tps, success, prompt_details


# ── Run mode ──────────────────────────────────────────────────────────────────

def cmd_run(selected_tier_names: list[str]):
    gs.cancel_requested = False
    gs.ram_tier_drop = False

    if selected_tier_names:
        # User explicitly selected these tiers — build plan directly without
        # an upfront RAM filter. Right after Apple Intelligence the Foundation
        # Models API may still hold memory, causing psutil to under-report
        # available RAM and producing a false "no tiers" error.
        # The per-tier RAM check below (before each tier runs) handles any
        # genuine shortage at runtime.
        plan = []
        for name in selected_tier_names:
            tier_def = next((t for t in BENCHMARK_TIERS if t["name"] == name), None)
            if tier_def is None:
                continue
            models = get_models_for_tier(tier_def["name"])
            if models:
                plan.append({"tier": tier_def, "models": models})
    else:
        plan = get_benchmark_plan()

    if not plan:
        emit({"event": "error", "message": "No tiers available with current RAM."})
        return

    start_ram_monitor()

    backend_order = ["MLX", "GGUF", "MLC"]
    run_data = {}
    benchmark_start = time.time()

    try:
        for plan_item in plan:
            if gs.cancel_requested:
                break

            tier = plan_item["tier"]
            tier_name = tier["name"]
            models = plan_item["models"]

            # Pre-tier RAM check
            available_ram = get_available_ram_gb()
            if available_ram < tier["min_ram_gb"]:
                print(f"[RAM] Skipping {tier_name}: {available_ram:.1f} GB < {tier['min_ram_gb']} GB")
                break

            gs.current_tier = tier_name
            run_data[tier_name] = {}

            # Get a representative model name for this tier
            model_name = next(iter(models.values()), {}).get("name", tier_name)
            emit({"event": "tier_start", "tier": tier_name, "model": model_name})

            sorted_models = sorted(
                models.items(),
                key=lambda kv: backend_order.index(kv[1]["format"])
                if kv[1]["format"] in backend_order else 99,
            )

            for key, model in sorted_models:
                if gs.cancel_requested or gs.ram_tier_drop:
                    break

                fmt = model["format"]

                # Per-backend RAM check
                available_ram = get_available_ram_gb()
                if available_ram < tier["min_ram_gb"]:
                    print(f"[RAM] Skipping {fmt}: {available_ram:.1f} GB available")
                    run_data[tier_name][fmt] = {"tps": 0.0, "success": False}
                    emit({"event": "backend_done", "tier": tier_name, "backend": fmt,
                          "tps": 0.0, "success": False})
                    continue

                gs.current_tps = 0.0
                gs.current_task = f"{tier_name} — {fmt}"

                # MLC JIT pre-check — skip download entirely if incompatible
                if fmt == "MLC":
                    _mlc = get_runner("MLC")
                    _jit_err = getattr(_mlc, "MLC_JIT_ERROR", None)
                    if _jit_err:
                        import shutil
                        _model_dir = get_models_dir() / key
                        if _model_dir.exists():
                            shutil.rmtree(str(_model_dir), ignore_errors=True)
                            print(f"[MLC] Model deleted (JIT incompatible): {_model_dir}")
                        run_data[tier_name][fmt] = {"tps": 0.0, "success": False}
                        emit({"event": "backend_done", "tier": tier_name, "backend": fmt,
                              "tps": 0.0, "success": False, "error": _jit_err})
                        continue

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

                    gs.cancel_requested = False
                    dl_result = download_model(key, dl_progress)

                    if gs.cancel_requested:
                        cleanup_partial_downloads([key])
                        break

                    if not dl_result["success"]:
                        print(f"[{fmt}] Download failed: {dl_result['message']}")
                        emit({"event": "download_failed", "key": key, "message": dl_result["message"]})
                        run_data[tier_name][fmt] = {"tps": 0.0, "success": False}
                        emit({"event": "backend_done", "tier": tier_name, "backend": fmt,
                              "tps": 0.0, "success": False})
                        continue

                    emit({"event": "download_done", "key": key})

                if gs.cancel_requested or gs.ram_tier_drop:
                    break

                emit({"event": "backend_start", "tier": tier_name, "backend": fmt})
                avg_tps, success, prompt_details = _run_backend(tier_name, key, model)
                run_data[tier_name][fmt] = {"tps": avg_tps, "success": success, "prompts": prompt_details}

                release_memory()

                # Disk space management between models
                next_not_ready = [
                    k2 for item2 in plan
                    for k2 in item2["models"].keys()
                    if not is_model_ready(k2) and k2 != key
                ]
                if next_not_ready:
                    next_key = next_not_ready[0]
                    next_needed_gb = get_models().get(next_key, {}).get("size_gb", 0)
                    free_gb = get_free_disk_space_gb()
                    if free_gb < next_needed_gb:
                        model_size_gb = get_folder_size(get_models_dir() / key) / (1024**3)
                        # Emit a disk_space event — Xcode UI decides whether to delete
                        emit({
                            "event": "disk_space_low",
                            "free_gb": round(free_gb, 1),
                            "needed_gb": round(next_needed_gb, 1),
                            "current_key": key,
                            "current_size_gb": round(model_size_gb, 1),
                        })
                        # Wait for response on stdin: "delete" or "skip"
                        response = _read_stdin_response(timeout=120)
                        if gs.cancel_requested:
                            break
                        if response == "delete":
                            from download_model import delete_model
                            delete_model(key, lambda: None)
                        else:
                            gs.cancel_requested = True
                            break

            emit({"event": "tier_done", "tier": tier_name})

            if gs.ram_tier_drop:
                available = get_available_ram_gb()
                print(f"[RAM] Dropped during {tier_name} — {available:.1f} GB available. Stopping.")
                emit({
                    "event": "ram_drop",
                    "tier": tier_name,
                    "available_gb": round(available, 1),
                })
                gs.ram_tier_drop = False
                gs.cancel_requested = False
                break

    except Exception as e:
        emit({"event": "error", "message": str(e)})
        return

    if gs.cancel_requested:
        if gs.cancel_delete_downloads:
            import shutil
            for item in plan:
                for key in item["models"]:
                    model_dir = get_models_dir() / key
                    if model_dir.exists():
                        try:
                            shutil.rmtree(model_dir)
                        except Exception as e:
                            print(f"[cancel] Could not delete {key}: {e}")
        emit({"event": "cancelled"})
        return

    # Save results
    try:
        tier_results = BenchmarkAnalyzer.compute_tier_results(run_data)
        if not tier_results:
            emit({"event": "error", "message": "No successful benchmark runs completed."})
            return
        benchmark_duration = round(time.time() - benchmark_start, 1)
        # Build prompt_results: tier → backend → prompt_name → detail
        prompt_results: dict = {}
        for t_name, backends in run_data.items():
            prompt_results[t_name] = {}
            for bk, bk_data in backends.items():
                if bk_data.get("success") and bk_data.get("tps", 0) > 0:
                    prompt_results[t_name][bk] = bk_data.get("prompts", {})
        file_path, benchmark_id = BenchmarkAnalyzer.save_result(tier_results, prompt_results, benchmark_duration)
        emit({
            "event": "complete",
            "file_path": str(file_path),
            "tier_results": tier_results,
            "prompt_results": prompt_results,
            "benchmark_id": benchmark_id,
        })
    except Exception as e:
        emit({"event": "error", "message": f"Failed to save results: {e}"})


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLBenchmark subprocess runner")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--info", action="store_true",
                      help="Emit hardware info and available tiers, then exit")
    mode.add_argument("--tiers", nargs="*", metavar="TIER",
                      help="Run benchmark for the given tier names (all available if empty)")
    args = parser.parse_args()

    if args.info:
        cmd_info()
    else:
        cmd_run(args.tiers or [])


if __name__ == "__main__":
    main()

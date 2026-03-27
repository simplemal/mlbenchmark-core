import os
import gc
import re
import sys
import math
import time
import traceback
from utils import log_exception_context, print_memory_usage, get_models

try:
    from mlc_llm import MLCEngine
    MLC_AVAILABLE = True
except ImportError:
    MLC_AVAILABLE = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_model = None
_repo = None

# Flag per indicare se la funzione è attualmente in esecuzione
_is_running = False


def load_model_if_needed(repo: str):
    """Carica il modello se necessario"""
    if not MLC_AVAILABLE:
        raise RuntimeError("mlc_llm is not installed on this system. MLC backend unavailable.")
    global _model, _repo
    if _model is not None and repo == _repo:
        print(f"[MLC] Model already loaded from: {repo}")
        return

    release_model()

    print(f"[MLC] Loading model from: {repo}")

    if not os.path.exists(repo):
        raise FileNotFoundError(f"Model not found at: {repo}")

    try:
        print_memory_usage("Before model load")
        _model = MLCEngine(repo)
        _repo = repo
        print_memory_usage("After model load")
        print(f"[MLC] Successfully loaded model from: {repo}")
    except Exception as e:
        print(f"[MLC] Exception loading model: {str(e)}")
        log_exception_context("load_model_if_needed failed")
        print(f"[MLC] Exception stacktrace: {traceback.format_exc()}")
        raise


def is_model_loaded() -> bool:
    """Controlla se il modello è caricato"""
    return _model is not None


def run_prompt(
    prompt: str, max_tokens: int, repo: str, ctx_max: int
) -> tuple[str, str, int]:
    """Esegue un prompt e restituisce (output, finish_reason, token_count)"""
    global _is_running

    MLC_HARD_LIMIT = 2000

    if _is_running:
        print("[MLC] WARNING: run_prompt called while already running — possible re-entrancy issue")

    _is_running = True

    try:
        # Stato modello prima del caricamento
        print(f"[MLC] run_prompt start — model_loaded={_model is not None}, repo_match={_repo == repo}")
        print_memory_usage("[MLC] before load_model_if_needed")

        load_model_if_needed(repo)
        print(f"[MLC] model ready — _model type: {type(_model).__name__}")
        print_memory_usage("[MLC] after load_model_if_needed")

        prompt_tokens = get_token_count(prompt)

        if prompt_tokens + max_tokens > MLC_HARD_LIMIT:
            adjusted_max_tokens = max(MLC_HARD_LIMIT - prompt_tokens, 16)
            print(f"[MLC] Capping max_tokens {max_tokens} → {adjusted_max_tokens} (hard limit {MLC_HARD_LIMIT})")
            max_tokens = adjusted_max_tokens

        ctx_requested = prompt_tokens + max_tokens
        if ctx_requested > ctx_max:
            adjusted_max_tokens = max(ctx_max - prompt_tokens, 16)
            print(f"[MLC] Capping max_tokens {max_tokens} → {adjusted_max_tokens} (ctx_max={ctx_max})")
            max_tokens = adjusted_max_tokens

        print(f"[MLC] Inference start — prompt_tokens={prompt_tokens}, max_out={max_tokens}, ctx_max={ctx_max}")

        if _model is None:
            raise RuntimeError("[MLC] _model is None after load_model_if_needed — cannot run inference")

        response = _model.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=repo,
            stream=True,
            max_tokens=max_tokens,
        )

        output = ""
        chunk_count = 0
        start_time = time.time()
        slow_chunk_count = 0
        last_chunk_time = start_time

        print("[MLC] Streaming response...")
        for chunk in response:
            current_time = time.time()
            chunk_gap = current_time - last_chunk_time
            last_chunk_time = current_time

            if chunk_gap > 0.5:
                slow_chunk_count += 1
                if slow_chunk_count > 3:
                    print(f"[MLC] WARNING: {slow_chunk_count} slow chunks (last gap={chunk_gap:.2f}s)")
            else:
                slow_chunk_count = 0

            if chunk.choices and chunk.choices[0].delta.content:
                output += chunk.choices[0].delta.content
                chunk_count += 1
                if chunk_count % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"[MLC] {chunk_count} chunks in {elapsed:.2f}s ({chunk_count/elapsed:.1f} t/s so far)")

        real_tokens = get_token_count(output)
        finish_reason = "length" if real_tokens >= max_tokens else "stop"
        elapsed_total = time.time() - start_time

        print(f"[MLC] Inference done — {real_tokens} tokens in {elapsed_total:.2f}s "
              f"({real_tokens/elapsed_total:.2f} t/s), finish={finish_reason}")
        print_memory_usage("[MLC] after inference")

        return output, finish_reason, real_tokens

    except Exception as e:
        print(f"[MLC] EXCEPTION in run_prompt: {type(e).__name__}: {e}")
        print(f"[MLC] Full traceback:\n{traceback.format_exc()}")
        print_memory_usage("[MLC] after exception")

        try:
            print("[MLC] Attempting emergency model reset after exception")
            release_model()
            print("[MLC] Emergency reset successful — model released")
        except Exception as reset_err:
            print(f"[MLC] Emergency reset also failed: {type(reset_err).__name__}: {reset_err}")

        raise
    finally:
        _is_running = False
        print("[MLC] run_prompt exited")


def get_token_count(prompt: str, margin: float = 0.15) -> int:
    """Stima il numero di token in un prompt"""
    try:
        if _model is not None:
            # Usare il modello per una stima più accurata
            tokens = len(prompt.split())
            return math.ceil(tokens * (1 + margin))
    except Exception as e:
        print(f"[MLC] Error in token counting with model: {e}")

    # Fallback
    words = re.findall(r"\b\w+\b", prompt)
    base_count = len(words)
    estimated = base_count * (1 + margin)
    return math.ceil(estimated)


def release_model():
    """Rilascia completamente il modello"""
    global _model, _repo

    if _model is not None:
        print("[MLC] Releasing model resources")
        try:
            _model.terminate()
            print("[MLC] Model terminated successfully")
        except Exception as e:
            print(f"[MLC] Error terminating model: {e}")
            print(f"[MLC] Exception stacktrace: {traceback.format_exc()}")
        finally:
            _model = None
            _repo = None
            gc.collect()
            print("[MLC] Model resources released and garbage collected")

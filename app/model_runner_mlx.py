import os
import gc
import re
import math
import mlx.core as mx
from utils import log_exception_context, print_memory_usage

from mlx_lm import load, generate as generate_block

try:
    from mlx_lm.generate_stream import generate as generate_stream

    STREAM_AVAILABLE = True
except ImportError:
    STREAM_AVAILABLE = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_model = None
_tokenizer = None
_repo = None


def load_model_if_needed(repo: str):
    global _model, _tokenizer, _repo
    if _model is not None and repo == _repo:
        return

    release_model()

    print(f"[MLX] Loading model from: {repo}")
    os.environ["MLX_CACHE_DIR"] = "/tmp/mlx_cache"
    try:
        print_memory_usage("Before model load")
        _model, _tokenizer = load(repo)
        print_memory_usage("After model load")
        _repo = repo
    except Exception:
        log_exception_context("load_model_if_needed failed")
        raise


def is_model_loaded() -> bool:
    return _model is not None


def run_prompt(
    prompt: str, max_tokens: int, repo: str, ctx_max: int
) -> tuple[str, str, int]:
    load_model_if_needed(repo)

    prompt_tokens = get_token_count(prompt)
    ctx_requested = prompt_tokens + max_tokens

    if ctx_requested > ctx_max:
        adjusted_max_tokens = max(ctx_max - prompt_tokens, 16)
        print(
            f"[WARN] Requested {ctx_requested} > ctx_max {ctx_max}. "
            f"Reducing max_tokens from {max_tokens} to {adjusted_max_tokens}"
        )
        max_tokens = adjusted_max_tokens

    if _tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = _tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    try:
        if STREAM_AVAILABLE:
            output = ""
            token_limit_reached = False
            for chunk in generate_stream(
                _model,
                _tokenizer,
                prompt,
                max_tokens=max_tokens,
                stream=True,
                verbose=False,
            ):
                output += chunk
                if len(_tokenizer.encode(output)) >= max_tokens:
                    token_limit_reached = True
                    break

            output = output.strip()
            real_tokens = len(_tokenizer.encode(output))
            finish_reason = "length" if token_limit_reached else "stop"
            return output, finish_reason, real_tokens

        else:
            output = generate_block(
                _model, _tokenizer, prompt, max_tokens=max_tokens, verbose=False
            ).strip()
            real_tokens = len(_tokenizer.encode(output))
            finish_reason = "length" if real_tokens >= max_tokens else "stop"
            return output, finish_reason, real_tokens

    except Exception:
        log_exception_context("run_prompt failed")
        raise


def get_token_count(prompt: str, margin: float = 0.10) -> int:
    try:
        if _tokenizer is not None:
            return len(_tokenizer.encode(prompt))
    except Exception:
        pass

    # fallback stimato
    words = re.findall(r"\b\w+\b", prompt)
    base_count = len(words)
    estimated = base_count * (1 + margin)
    return math.ceil(estimated)


def release_model():
    global _model, _tokenizer, _repo
    _model = None
    _tokenizer = None
    _repo = None
    mx.clear_cache()
    gc.collect()

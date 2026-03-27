import gc
import re
import math
import sys
from pathlib import Path

from llama_cpp import Llama
from utils import print_memory_usage, detect_gpu_cores, get_available_ram_gb, get_models

_model = None
_model_path = None
_ctx_size = None
_is_chat_model = False

# === GPU Layer Logic Settings ===
MIN_RAM_FACTOR = 1.5  # moltiplicatore rispetto a min_ram per permettere più layer
LAYER_WEAK = 4  # se GPU debole, RAM sufficiente
LAYER_MEDIUM = 8  # se GPU media, RAM sufficiente
LAYER_STRONG = 20  # se GPU forte, RAM sufficiente

LAYER_WEAK_LOW = 0  # GPU debole, RAM insufficiente
LAYER_MEDIUM_LOW = 4  # GPU media, RAM insufficiente
LAYER_STRONG_LOW = 12  # GPU forte, RAM insufficiente

GPU_THRESHOLD_WEAK = 10
GPU_THRESHOLD_MEDIUM = 20

DEFAULT_N_GPU_LAYERS = 0  # fallback


def compute_gpu_layers(model_path):
    try:
        model_key = Path(model_path).parts[-2]
        model_info = get_models().get(model_key)

        if not model_info:
            print(
                f"[DEBUG] compute_gpu_layers: model key '{model_key}' not found in repository"
            )
            return DEFAULT_N_GPU_LAYERS

        min_ram = float(model_info.get("min_ram", 0))
        size_gb = float(model_info.get("size_gb", 0))

        gpu_cores_raw = detect_gpu_cores()
        gpu_cores = int(gpu_cores_raw) if gpu_cores_raw.isdigit() else 0
        ram_available = get_available_ram_gb()
        ram_ok = ram_available >= (min_ram * MIN_RAM_FACTOR)

        print(f"[DEBUG] compute_gpu_layers: model={model_key}")
        print(f"[DEBUG] GPU cores={gpu_cores}, RAM available={ram_available:.2f} GB")
        print(f"[DEBUG] min_ram={min_ram}, size_gb={size_gb}, ram_ok={ram_ok}")

        if gpu_cores < GPU_THRESHOLD_WEAK:
            layers = LAYER_WEAK if ram_ok else LAYER_WEAK_LOW
        elif gpu_cores < GPU_THRESHOLD_MEDIUM:
            layers = LAYER_MEDIUM if ram_ok else LAYER_MEDIUM_LOW
        else:
            layers = LAYER_STRONG if ram_ok else LAYER_STRONG_LOW

        print(f"[DEBUG] Selected n_gpu_layers={layers}")
        return layers

    except Exception as e:
        print(f"[ERROR] compute_gpu_layers: fallback to {DEFAULT_N_GPU_LAYERS} ({e})")
        return DEFAULT_N_GPU_LAYERS


def load_model_if_needed(model_path: str, n_ctx: int = 4096):
    global _model, _model_path, _ctx_size, _is_chat_model

    if _model and _model_path == model_path and _ctx_size == n_ctx:
        return  # Already loaded with same context size

    release_model()

    n_gpu_layers = compute_gpu_layers(model_path)
    print(
        f"[GGUF] Loading model from: {model_path} with n_ctx={n_ctx} and n_gpu_layers={n_gpu_layers}"
    )
    print_memory_usage("Before model load")

    _model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=8,
        n_gpu_layers=n_gpu_layers,
        use_mlock=True,
        verbose=False,
    )
    _model_path = model_path
    _ctx_size = n_ctx

    try:
        _is_chat_model = _model.chat_format is not None
        print_memory_usage("After model load")
        print(f"[GGUF] Chat format detected: {_model.chat_format}")
    except Exception:
        _is_chat_model = False
        print("[GGUF] No chat format detected. Using plain prompt.")


def is_model_loaded() -> bool:
    return _model is not None


def run_prompt(
    prompt: str, max_tokens: int, model_path: str, ctx_max: int
) -> tuple[str, str, int]:

    prompt_tokens = get_token_count(prompt)
    ctx_requested = prompt_tokens + max_tokens

    if ctx_requested > ctx_max:
        adjusted_max_tokens = max(ctx_max - prompt_tokens, 16)  # minimi di sicurezza
        print(
            f"[WARN] Requested {ctx_requested} > ctx_max {ctx_max}. "
            f"Reducing max_tokens from {max_tokens} to {adjusted_max_tokens}"
        )
        max_tokens = adjusted_max_tokens

    load_model_if_needed(model_path, ctx_max)

    if _is_chat_model:
        result = _model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
            stream=False,
        )
        content = result["choices"][0]["message"]["content"].strip()
        reason = result["choices"][0].get("finish_reason", "unknown")
        tokens = get_token_count(content)
        return content, reason, tokens

    else:
        formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        stream = _model(
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            stop=["</s>"],
            temperature=0.7,
            echo=False,
            stream=True,
        )

        output = ""
        finish_reason = "length"
        for chunk in stream:
            delta = chunk["choices"][0]["text"]
            output += delta
            sys.stdout.write(delta)
            sys.stdout.flush()
            if chunk["choices"][0].get("finish_reason") == "stop":
                finish_reason = "stop"
                break

        tokens = get_token_count(output)
        return output.strip(), finish_reason, tokens


def get_token_count(prompt: str, margin: float = 0.10) -> int:
    try:
        if _model is not None:
            tokens = _model.tokenize(prompt.encode("utf-8"), add_bos=False)
            return len(tokens)
    except Exception:
        pass

    # fallback stimato
    words = re.findall(r"\b\w+\b", prompt)
    base_count = len(words)
    estimated = base_count * (1 + margin)
    return math.ceil(estimated)


def release_model():
    global _model, _model_path, _ctx_size, _is_chat_model
    _model = None
    _model_path = None
    _ctx_size = None
    _is_chat_model = False
    gc.collect()

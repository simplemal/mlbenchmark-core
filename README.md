# ML Benchmark — Core

Inference engine for the [ML Benchmark](https://mlbenchmark.app) macOS app.

Runs local LLM inference on Apple Silicon across three backends — **MLX**, **GGUF** (llama.cpp), and **MLC** — using the same fixed prompt suite the app uses to produce reproducible, comparable results.

---

## Requirements

- macOS with Apple Silicon (M1 or later)
- Python 3.12 from [python.org](https://www.python.org/downloads/) (not the system Python)
- Xcode Command Line Tools: `xcode-select --install`
- CMake (needed to build llama-cpp-python): `brew install cmake`
- A [Hugging Face](https://huggingface.co) account with access to the gated models

---

## Setup

```bash
git clone https://github.com/your-org/mlbenchmark-core
cd mlbenchmark-core
chmod +x setup.sh && ./setup.sh
```

The script will:
1. Create a `.venv` with Python 3.12
2. Install MLX, llama-cpp-python (Metal), and mlc-llm (nightly)
3. Ask for your Hugging Face token — needed to download models

Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

Then activate the venv:

```bash
source .venv/bin/activate
```

---

## CLI

The `cli.py` tool lets you inspect the environment and run targeted tests without the app.

```
python3 cli.py <command> [args]
```

### `info` — system overview

Prints hardware info, installed packages, and which models are already on disk.

```bash
python3 cli.py info
```

### `jit <tier>` — MLC JIT diagnostics

Step-by-step check of the MLC compilation pipeline. Useful to diagnose failures before running the full benchmark.

```bash
python3 cli.py jit nano
python3 cli.py jit entry
```

Tiers: `nano` `entry` `standard` `advanced` `extreme`

### `test <backend> <tier>` — full inference test

Downloads the model if needed, runs a warm-up, then a single test prompt and reports tokens/s.

```bash
python3 cli.py test mlx nano
python3 cli.py test gguf nano
python3 cli.py test mlc nano
```

Backends: `mlx` `gguf` `mlc`

### `prompt <text>` — free prompt

Runs a single custom prompt against any backend and tier.

```bash
python3 cli.py prompt "What is the capital of France?" --backend mlx --tier nano
python3 cli.py prompt "Explain quantum entanglement." --backend gguf --tier standard --max-tokens 512
```

---

## Models

Models are downloaded automatically on first use from Hugging Face into:

```
~/Library/Application Support/MLBenchmark/models/   ← if the app is installed
./models/                                            ← fallback for standalone use
```

| Tier     | Model                    | ~Size (all backends) |
|----------|--------------------------|----------------------|
| Nano     | Llama 3.2 3B Instruct    | ~6 GB                |
| Entry    | Phi-3.5 Mini Instruct    | ~7 GB                |
| Standard | Gemma 2 9B Instruct      | ~17 GB               |
| Advanced | Qwen 2.5 14B Instruct    | ~26 GB               |
| Extreme  | Qwen 2.5 32B Instruct    | ~58 GB               |

Each tier has three variants: MLX, GGUF, and MLC. You can download only the backend you need.

---

## Architecture

```
cli.py                   ← entry point for CLI commands
app/
  model_runner_mlx.py    ← MLX backend (mlx-lm)
  model_runner_gguf.py   ← GGUF backend (llama-cpp-python)
  model_runner_mlc.py    ← MLC backend (mlc-llm)
  benchmark_runner.py    ← subprocess interface (JSON lines on stdout)
  benchmark_analyzer.py  ← TPS scoring and averages
  globals_state.py       ← shared mutable state across threads
  utils.py               ← hardware detection, config helpers
  config.json            ← prompt suite and score weights
  repository.json        ← model registry (HF repo ids, sizes, quantization)
```

All three runners implement the same interface:

```python
load_model_if_needed(repo: str)
run_prompt(prompt, max_tokens, repo, ctx_max) -> (output, finish_reason, token_count)
release_model()
```

`benchmark_runner.py` is the subprocess entry point used by the ML Benchmark app. It communicates over stdout as JSON lines and accepts `cancel` on stdin.

---

## License

MIT

import math
import json
import csv
import os
import hashlib
import random
import time

from datetime import datetime
from pathlib import Path
from utils import get_score_weigth, get_prompt_categories, BENCHMARK_TIERS, get_results_dir, detect_apple_chip, detect_mac_model, detect_memory_gb, detect_gpu_cores, detect_ne_cores, detect_os_name, detect_os_version, get_models_for_tier

# Capacity points awarded per completed tier (cumulative)
CAPACITY_WEIGHTS = {
    "Entry":    1,
    "Standard": 2,
    "Advanced": 3,
    "Extreme":  5,
}


class BenchmarkAnalyzer:
    def __init__(self, file_path, average_mode="geometric"):
        self.file_path = file_path
        self.average_mode = average_mode
        self.row = self._load_first_row()

    def _load_first_row(self):
        with open(self.file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                return row
        return {}

    def _geometric_mean(self, values):
        values = [v for v in values if v > 0]
        if not values:
            return 0
        product = math.prod(values)
        return product ** (1 / len(values))

    def _arithmetic_mean(self, values):
        values = [v for v in values if v > 0]
        if not values:
            return 0
        return sum(values) / len(values)

    def get_score_type(self):
        try:
            basic = json.loads(self.row.get("prompt_basic", "{}"))
            advanced = json.loads(self.row.get("prompt_advanced", "{}"))
        except json.JSONDecodeError:
            return "unknown"

        has_basic = bool(basic)
        has_advanced = bool(advanced)

        if has_basic and has_advanced:
            return "mixed"
        elif has_advanced:
            return "advanced"
        elif has_basic:
            return "basic"
        else:
            return "unknown"

    def get_full_scores(self):
        df_row = self.row
        scores = {}

        for key in ["prompt_basic", "prompt_advanced"]:
            if key not in df_row or not df_row[key]:
                continue
            try:
                data = json.loads(df_row[key])
            except Exception:
                continue

            for prompt, entries in data.items():
                for token_level, entry in entries.items():
                    if not isinstance(entry, dict):
                        continue

                    tps = entry.get("tps", 0)
                    success = entry.get("success")
                    degenerate = entry.get("degenerate")
                    finish = entry.get("finish_reason")

                    if success is True and not degenerate:
                        pass  # tps già valido
                    elif success is False and finish == "length":
                        if degenerate:
                            tps *= get_score_weigth("truncated_and_degenerate")
                        else:
                            tps *= get_score_weigth("truncated")
                    elif degenerate:
                        tps *= get_score_weigth("truncated_and_degenerate")
                    else:
                        continue  # altri casi non contano

                    if tps > 0:
                        scores.setdefault(prompt, []).append(tps)

        prompt_best = {k: max(v) for k, v in scores.items()}

        group_results = {}
        for group, prompts in get_prompt_categories().items():
            values = [prompt_best[p] for p in prompts if p in prompt_best]
            score = self._geometric_mean(values)
            group_results[group] = int(score * 10)

        if self.average_mode == "geometric":
            global_score = self._geometric_mean(list(group_results.values()))
        else:
            global_score = self._arithmetic_mean(list(group_results.values()))

        result = {
            "total": int(global_score),
        }
        for group, score in group_results.items():
            result[f"{group}"] = score

        return result

    # ------------------------------------------------------------------
    # New tier-based scoring (v1.0 format)
    # ------------------------------------------------------------------

    def get_tier_velocity_scores(self) -> dict:
        """
        Return per-tier velocity score (avg TPS across backends that succeeded).

        Reads the 'tier_results' CSV column (JSON):
          {tier_name: {backend: tps_value, ...}, ...}

        Returns:
          {tier_name: avg_tps, ...}  — only for tiers with at least one result
        """
        try:
            data = json.loads(self.row.get("tier_results", "{}"))
        except (json.JSONDecodeError, AttributeError):
            return {}

        velocity = {}
        tier_order = [t["name"] for t in BENCHMARK_TIERS]
        for tier_name in tier_order:
            backends = data.get(tier_name, {})
            values = [v for v in backends.values() if isinstance(v, (int, float)) and v > 0]
            if values:
                velocity[tier_name] = round(sum(values) / len(values), 2)
        return velocity

    def get_capacity_score(self) -> dict:
        """
        Return the capacity score based on which tiers were completed.

        Returns:
          score           — cumulative capacity points
          max_tier        — name of the highest completed tier (or None)
          completed_tiers — list of completed tier names in order
        """
        try:
            data = json.loads(self.row.get("tier_results", "{}"))
        except (json.JSONDecodeError, AttributeError):
            return {"score": 0, "max_tier": None, "completed_tiers": []}

        tier_order = [t["name"] for t in BENCHMARK_TIERS]
        completed = [t for t in tier_order if data.get(t)]
        score = sum(CAPACITY_WEIGHTS.get(t, 0) for t in completed)
        return {
            "score": score,
            "max_tier": completed[-1] if completed else None,
            "completed_tiers": completed,
        }

    @staticmethod
    def save_result(tier_results: dict, prompt_results: dict = None, benchmark_duration: float = 0) -> Path:
        """
        Save a benchmark result to a CSV file in the results directory.

        tier_results    : compact {tier: {backend: avg_tps}}
        prompt_results  : detailed {tier: {backend: {prompt: {tps, elapsed, real_tokens, finish_reason}}}}
        benchmark_duration: total elapsed seconds for the run

        Returns the path of the saved CSV file.
        """
        import os as _os
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = get_results_dir()
        file_path = results_dir / f"benchmark_{timestamp}.csv"

        # Stable unique ID for this specific benchmark run.
        # sha256(chip | timestamp_ms | 5-digit random) → first 32 hex chars.
        _seed = f"{detect_apple_chip()}|{int(time.time() * 1000)}|{random.randint(10000, 99999)}"
        benchmark_id = hashlib.sha256(_seed.encode()).hexdigest()[:32]

        mac = detect_mac_model()

        from utils import get_version
        app_version = get_version()

        # Compute weighted scores
        scores = BenchmarkAnalyzer.compute_scores(tier_results)

        fieldnames = [
            "timestamp",
            "app_version",
            "os_name",
            "os_version",
            "hardware",
            "mac_name",
            "mac_chip",
            "mac_year",
            "ram_gb",
            "gpu_cores",
            "ne_cores",
            "cpu_cores",
            "benchmark_duration",
            "tier_results",
            "scores",
            "prompt_results",
            "benchmark_id",
        ]

        row = {
            "timestamp": timestamp,
            "app_version": app_version,
            "os_name": detect_os_name(),
            "os_version": detect_os_version(),
            "hardware": detect_apple_chip(),
            "mac_name": mac["name"],
            "mac_chip": mac["chip"],
            "mac_year": mac["year"] or "",
            "ram_gb": detect_memory_gb(),
            "gpu_cores": detect_gpu_cores(),
            "ne_cores": detect_ne_cores(),
            "cpu_cores": _os.cpu_count() or 0,
            "benchmark_duration": benchmark_duration,
            "tier_results": json.dumps(tier_results),
            "scores": json.dumps(scores),
            "prompt_results": json.dumps(prompt_results or {}),
            "benchmark_id": benchmark_id,
        }

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

        return file_path, benchmark_id

    @staticmethod
    def effective_weight(tier_name: str, backend: str) -> float:
        """Compute effective weight: params_b × (quant_bits / 16).
        Reads params_b and quant_bits from repository.json dynamically."""
        models = get_models_for_tier(tier_name)
        for model in models.values():
            if model.get("format") == backend:
                params_b = model.get("params_b", 1)
                quant_bits = model.get("quant_bits", 16)
                return params_b * (quant_bits / 16)
        # Fallback: use any model in the tier for params_b
        for model in models.values():
            params_b = model.get("params_b", 1)
            quant_bits = model.get("quant_bits", 16)
            return params_b * (quant_bits / 16)
        return 1.0

    @staticmethod
    def compute_scores(tier_results: dict) -> dict:
        """Compute weighted scores from tier_results.

        Returns: {tier_name: {"MLX": score, ..., "_avg": avg_score}, ...}
        where score = tps × effective_weight, _avg = mean of backend scores
        """
        scores = {}
        for tier_name, backends in tier_results.items():
            tier_scores = {}
            for backend, tps in backends.items():
                ew = BenchmarkAnalyzer.effective_weight(tier_name, backend)
                tier_scores[backend] = round(tps * ew, 2)
            vals = [v for v in tier_scores.values() if v > 0]
            if vals:
                tier_scores["_avg"] = round(sum(vals) / len(vals), 2)
            if tier_scores:
                scores[tier_name] = tier_scores
        return scores

    @staticmethod
    def compute_tier_results(run_data: dict) -> dict:
        """
        Compute the tier_results dict from raw benchmark run data.

        run_data format (produced by the benchmark loop):
          {tier_name: {backend: {"tps": float, "success": bool, ...}, ...}, ...}

        Returns the compact format stored in the CSV:
          {tier_name: {backend: tps_value, ...}, ...}
        Only successful runs with tps > 0 are included.
        """
        results = {}
        for tier_name, backends in run_data.items():
            tier_data = {}
            for backend, data in backends.items():
                if isinstance(data, dict) and data.get("success") and data.get("tps", 0) > 0:
                    tier_data[backend] = round(data["tps"], 2)
            if tier_data:
                results[tier_name] = tier_data
        return results

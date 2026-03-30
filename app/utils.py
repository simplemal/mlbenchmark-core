import os
import re
import sys
import platform
import subprocess
import traceback
import threading
import math
import time
import psutil
import shutil
import locale
import json

from datetime import datetime

import tkinter as tk
import tkinter.font as tkFont

from tkinter import messagebox, filedialog
from pathlib import Path
from subprocess import Popen, PIPE


def is_frozen_app():
    return getattr(sys, "frozen", False)


def is_bundle_context():
    """True when running inside a .app bundle — either via PyInstaller (sys.frozen)
    or as a Python subprocess launched by the Swift app (script lives inside .app/Contents/).
    Use this (instead of is_frozen_app) for any path that must persist across app updates,
    such as models, results and the home dir."""
    if is_frozen_app():
        return True
    return ".app/Contents/" in str(Path(__file__).resolve())


def log_exception_context(label: str):
    """Logga un'eccezione catturata manualmente (senza crash)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path.home() / "Library" / "Logs" / "MLBenchmark"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"MLBenchmark_{timestamp}_mlx.log"

    with open(log_path, "w") as f:
        f.write(f"{label} - Exception context:\n\n")
        traceback.print_exception(*sys.exc_info(), file=f)

    # opzionale: apri subito il log
    subprocess.run(["open", "-a", "TextEdit", str(log_path)])


def resource_path(filename):
    """
    Return the absolute path to a resource, working both from source and PyInstaller .app
    """
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def shared_path(filename):
    """
    Return the absolute path to a file in the shared/ folder.
    In dev: ../shared/ relative to app/. In PyInstaller bundle: _MEIPASS root.
    """
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shared", filename)


# Function: Load config from config.json
def load_config():
    with open(resource_path("config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_config()


# Function: Load repositoty from repository.json
def load_repository():
    with open(shared_path("repository.json"), "r", encoding="utf-8") as f:
        return json.load(f)


REPOSITORY = load_repository()


def get_models():
    global REPOSITORY
    return REPOSITORY


def get_scroll_speed():
    return 2.5


def is_dark_mode():
    """Return True if macOS is in dark mode using native AppKit API"""
    if platform.system() != "Darwin":
        return False
    try:
        from AppKit import NSApplication
        from Foundation import NSUserDefaults

        defaults = NSUserDefaults.standardUserDefaults()
        style = defaults.stringForKey_("AppleInterfaceStyle")
        return style == "Dark"
    except Exception as e:
        print(f"[DEBUG] Failed to get theme (NSUserDefaults): {e}")
        return False


DARK = is_dark_mode()


def monitor_theme(widget, callback, interval=1500):
    """Call `callback()` if dark mode changed. Uses `widget.after()` every `interval` ms."""
    global DARK
    current = is_dark_mode()
    if current != DARK:
        DARK = current
        callback()
    widget.after(
        interval,
        lambda: monitor_theme(widget, callback, interval),
    )


def get_prompt_warm_up():
    global CONFIG
    PROMPT_WARM_UP = CONFIG.get("PROMPT_WARM_UP", "Hi, how are you?")
    return PROMPT_WARM_UP


def get_prompts(level: str = None):
    global CONFIG
    if level:
        PROMPT = CONFIG.get("PROMPT").get(level)
    else:
        PROMPT = CONFIG.get("PROMPT")
    return PROMPT


def get_prompt_categories():
    global CONFIG
    return CONFIG.get("PROMPT_CATEGORIES")


def get_score_weigth(prompt: str = None):
    SCORE_WEIGHT = CONFIG.get("SCORE_WEIGHT")
    if prompt:
        SCORE_WEIGHT = CONFIG.get("SCORE_WEIGHT").get(prompt, "Not Found")
    return SCORE_WEIGHT


def get_quant_map():
    global CONFIG
    QUANT_MAP = CONFIG.get("QUANT_MAP")
    return QUANT_MAP


def get_app_name():
    global CONFIG
    return CONFIG.get("APP_NAME", "app")


def get_version():
    global CONFIG
    return CONFIG.get("VERSION", "0.0")


def get_version_check_url():
    global CONFIG
    return CONFIG.get("VERSION_CHECK_URL", "")


def check_for_update():
    result = {
        "success": False,
        "error": None,
        "local": {"version": None},
        "remote": {},
        "need_update": False,
    }

    try:
        local_version = get_version()
        result["local"]["version"] = local_version
        local_tuple = tuple(map(int, local_version.split(".")))

        url = get_version_check_url()
        output = subprocess.check_output(["curl", "-s", url], stderr=subprocess.STDOUT)
        data = output.decode("utf-8")

        remote_data = json.loads(data)

        if not remote_data.get("success") or "version" not in remote_data:
            result["error"] = "Invalid response structure"
            return result

        result["remote"] = remote_data
        remote_version = remote_data["version"]
        remote_tuple = tuple(map(int, remote_version.split(".")))

        if remote_tuple > local_tuple:
            result["need_update"] = True

        result["success"] = True

    except subprocess.CalledProcessError as e:
        result["error"] = f"Network error: {e.output.decode('utf-8').strip()}"
    except json.JSONDecodeError:
        result["error"] = "Failed to decode JSON"
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"

    return result


def is_update_available():
    """Check if a new version is available without opening any window."""
    result = check_for_update()
    if not result.get("success", False):
        return False
    return result.get("need_update", False)


def cleanup_incomplete_download(extra_file=None):
    """Delete residual .downloading file and optional specific file if exists."""
    try:
        app_name = get_app_name()
        temp_filename = f"{app_name}_latest.dmg.downloading"
        temp_path = os.path.expanduser(f"~/Downloads/{temp_filename}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception:
        pass

    if extra_file:
        try:
            extra_path = os.path.expanduser(f"~/Downloads/{extra_file}")
            if os.path.exists(extra_path):
                os.remove(extra_path)
        except Exception:
            pass


def get_author():
    global CONFIG
    return CONFIG.get("AUTHOR", "SaggiaMente")


def get_color(key):
    global CONFIG
    UI = CONFIG.get("UI")
    return UI["colors"]["dark" if is_dark_mode() else "light"].get(key, "")


def get_font_name():
    global CONFIG
    for name in CONFIG.get("UI", {}).get("font", []):
        try:
            tkFont.Font(family=name)
            return name
        except tk.TclError:
            continue
    return "TkDefaultFont"  # fallback di sicurezza


def get_font(size=12, style="normal"):
    family = get_font_name()
    style = style.lower()

    # Costruzione della tupla: ("FontName", size, "bold"), ecc.
    if style in ("bold", "italic"):
        return (family, size, style)
    else:
        return (family, size)


def get_card_dimension(dimension: str):
    global CONFIG
    CARD_DIMENSION = CONFIG.get("UI").get("results_card")
    if dimension:
        CARD_DIMENSION = CARD_DIMENSION.get(dimension, 100)
    return CARD_DIMENSION


_APP_ROOT = Path(__file__).resolve().parent.parent


def get_models_dir():
    if is_bundle_context():
        base = Path.home() / "Library/Application Support/MLBenchmark/models"
    else:
        base = _APP_ROOT / "models"
    base.mkdir(parents=True, exist_ok=True)
    return base


# Stable numeric IDs for each tier — independent of model name or tier display name.
_TIER_ID = {"Light": 1, "Speed": 2, "Flash": 3, "Blaze": 4, "Ultra": 5}


def key_to_folder(key: str) -> Path:
    """Convert a repository key to a stable model folder path.

    e.g. 'Light__Qwen2.5-0.5B-Instruct__MLX'  →  Path('Tier1/MLX')

    The folder is independent of the model name: changing the model for a tier
    only requires re-downloading, not hunting for stale directories.
    """
    parts = key.split("__")
    tier_name = parts[0]
    backend   = parts[2] if len(parts) >= 3 else ""
    tier_id   = _TIER_ID.get(tier_name, tier_name)   # fallback: use name as-is
    return Path(f"Tier{tier_id}") / backend


def get_results_dir():
    if is_bundle_context():
        base = Path.home() / "Library/Application Support/MLBenchmark/results"
    else:
        base = _APP_ROOT / "results"
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_home_dir():
    if is_bundle_context():
        base = Path.home() / "Library/Application Support/MLBenchmark/"
    else:
        base = _APP_ROOT
    base.mkdir(parents=True, exist_ok=True)
    return base


def has_permission_file():
    has_permission = (get_home_dir() / ".has_permission").exists()
    if not is_frozen_app:
        print(f"[DEBUG] has_permission: {has_permission}")
    return has_permission


def remove_permission_flag():
    if not is_frozen_app:
        print("[DEBUG] Remove permision file")
    (get_home_dir() / ".has_permission").unlink(missing_ok=True)


def save_permission_flag():
    if not is_frozen_app:
        print("[DEBUG] Saving permision file")
    (get_home_dir() / ".has_permission").touch()


def get_available_ram_gb() -> float:
    """Return the amount of RAM available to the system in GB, including purgeable memory."""
    return psutil.virtual_memory().available / (1024**3)


def has_enough_ram(required_gb: float, safety_margin_gb: float = 2.0) -> bool:
    """Check if the system has enough available RAM for a model, including a safety margin."""
    return get_available_ram_gb() >= (required_gb + safety_margin_gb)


def print_memory_usage(label=""):
    process = psutil.Process()
    mem_bytes = process.memory_info().rss  # Resident Set Size
    mem_gb = mem_bytes / (1024**3)
    print(f"[RAM USAGE] {label}: {mem_gb:.2f} GB (RSS)")


def get_current_ram_usage_gb() -> float:
    """Return current RAM usage of this process (in GB, RSS only)."""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)


# ---------------------------------------------------------------------------
# Benchmark tier definitions and RAM-based tier selection
# ---------------------------------------------------------------------------

def _load_tiers_from_config() -> list:
    cfg = load_config()
    return [
        {
            "name": t["id"],
            "label": t["display"],
            "model_size": t["model_size"],
            "min_ram_gb": t["min_ram_gb"],
            "disk_gb": t["disk_gb"],
            "icon": t["icon"],
        }
        for t in cfg.get("TIERS", [])
    ]


BENCHMARK_TIERS = _load_tiers_from_config()

# Fraction of total RAM assumed free at OS idle — used to compute potential tiers
_OS_IDLE_FRACTION = 0.85


def get_available_tiers(available_gb: float = None) -> list:
    """Return tiers safely runnable with the current available RAM."""
    if available_gb is None:
        available_gb = get_available_ram_gb()
    return [t for t in BENCHMARK_TIERS if available_gb >= t["min_ram_gb"]]


def get_potential_tiers(total_gb: float = None) -> list:
    """Return tiers that could run if RAM were fully freed (total × 0.85)."""
    if total_gb is None:
        total_gb = detect_memory_gb()
    potential_gb = total_gb * _OS_IDLE_FRACTION
    return [t for t in BENCHMARK_TIERS if potential_gb >= t["min_ram_gb"]]


def get_models_for_tier(tier_name: str) -> dict:
    """Return all models in repository.json that belong to the given tier."""
    return {k: v for k, v in REPOSITORY.items() if v.get("tier") == tier_name}


def start_ram_monitor(interval: float = 5.0):
    """
    Start a background thread that monitors available RAM during a benchmark run.
    If RAM drops below the minimum required for the current tier, sets
    globals_state.ram_tier_drop = True and globals_state.cancel_requested = True.

    The thread stops automatically when globals_state.cancel_requested is True.
    Call this just before starting the benchmark loop.
    """
    import threading
    import globals_state as gs

    def _monitor():
        consecutive_low = 0
        while not gs.cancel_requested:
            if gs.current_tier:
                tier_def = next((t for t in BENCHMARK_TIERS if t["name"] == gs.current_tier), None)
                if tier_def:
                    available = get_available_ram_gb()
                    if available < tier_def["min_ram_gb"]:
                        consecutive_low += 1
                        print(f"[RAM MONITOR] RAM at {available:.1f} GB — below {gs.current_tier} minimum "
                              f"({tier_def['min_ram_gb']} GB). Strike {consecutive_low}/2.")
                        if consecutive_low >= 4:
                            print(f"[RAM MONITOR] Confirmed low RAM — stopping benchmark.")
                            gs.ram_tier_drop = True
                            gs.cancel_requested = True
                            return
                    else:
                        if consecutive_low > 0:
                            print(f"[RAM MONITOR] RAM recovered to {available:.1f} GB — resetting strike count.")
                        consecutive_low = 0
            time.sleep(interval)

    t = threading.Thread(target=_monitor, daemon=True)
    t.start()
    return t


def get_benchmark_plan(available_gb: float = None) -> list:
    """
    Return the ordered list of (tier, models_dict) pairs to run,
    from Entry up to the highest tier the current RAM supports.
    Each entry: {"tier": tier_dict, "models": {key: model_dict}}
    """
    if available_gb is None:
        available_gb = get_available_ram_gb()
    runnable = get_available_tiers(available_gb)
    plan = []
    for tier in runnable:
        models = get_models_for_tier(tier["name"])
        if models:
            plan.append({"tier": tier, "models": models})
    return plan


def get_ram_tier_status() -> dict:
    """
    Return a snapshot of RAM state and tier availability.

    Keys:
      total_gb        — total installed RAM
      available_gb    — RAM available right now
      available_tiers — tiers runnable now (list of tier dicts)
      potential_tiers — tiers runnable if RAM were freed (list of tier dicts)
      show_warning    — True if freeing RAM would unlock additional tiers
    """
    total_gb = float(detect_memory_gb())
    available_gb = get_available_ram_gb()
    available_tiers = get_available_tiers(available_gb)
    potential_tiers = get_potential_tiers(total_gb)
    show_warning = len(potential_tiers) > len(available_tiers)
    return {
        "total_gb": total_gb,
        "available_gb": available_gb,
        "available_tiers": available_tiers,
        "potential_tiers": potential_tiers,
        "show_warning": show_warning,
    }


# Function: Get folder size — describes what this function does
def get_folder_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total


# Function: Detect apple chip — describes what this function does
def detect_apple_chip():
    try:
        raw = (
            subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
            .decode()
            .strip()
        )
        # Normalize: remove "Apple " prefix to match Swift's mac_chip format
        if raw.startswith("Apple "):
            raw = raw[6:]
        return raw
    except:
        return platform.processor()


def _load_mac_model_map() -> dict:
    try:
        with open(shared_path("mac_models.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def detect_mac_model() -> dict:
    """Return a dict with keys 'name', 'chip', 'year' for the current Mac model.
    Falls back to {'name': raw_identifier, 'chip': '', 'year': None} if unknown."""
    try:
        identifier = subprocess.check_output(
            ["sysctl", "-n", "hw.model"], text=True
        ).strip()
        entry = _load_mac_model_map().get(identifier)
        if entry:
            return entry
        return {"name": identifier, "chip": "", "year": None}
    except Exception:
        return {"name": "", "chip": "", "year": None}


# Function: Detect memory gb — describes what this function does
def detect_memory_gb():
    return round(psutil.virtual_memory().total / 1024**3)


def format_memory_value_with_unit(key, value):
    if isinstance(value, int):
        if "bytes" in key.lower() or "size" in key.lower() or value > 10**6:
            return f"{value / 1024**3:.2f} GB"
        elif value > 10**3:
            return f"{value / 1024:.2f} KB"
        else:
            return f"{value} B"
    return str(value)


def detect_memory_mac_style_blocks():
    mem = psutil.virtual_memory()

    total = mem.total / 1024**3
    wired = getattr(mem, "wired", 0) / 1024**3
    in_use = getattr(mem, "active", 0) / 1024**3
    cache = getattr(mem, "inactive", 0) / 1024**3
    available = mem.available / 1024**3

    corrected_free = max(0, available - cache)
    other = max(0, total - (wired + in_use + cache + corrected_free))

    return {
        "wired": {"name": "Wired", "value": round(wired, 2)},
        "cache": {"name": "Cache", "value": round(cache, 2)},
        "free": {"name": "Free", "value": round(corrected_free, 2)},
        "other": {"name": "Other", "value": round(other, 2)},
        "total": {"name": "Total", "value": round(total, 2)},
        "used": {"name": "Used", "value": round(mem.used / 1024**3, 2)},
    }


def get_free_disk_space_gb(path=None):
    """Returns the available disk space in GB"""
    if path is None:
        path = get_models_dir()
    usage = shutil.disk_usage(path)
    return round(usage.free / 1024**3, 2)


def check_disk_space_for_models(keys: list) -> dict:
    """
    Check if there is enough disk space to download the given model keys.
    Only counts models that are not already fully downloaded.

    Returns a dict with:
      needed_gb     — total GB needed for missing models
      free_gb       — current free disk space
      enough        — True if free_gb >= needed_gb
      missing_keys  — model keys not yet downloaded
    """
    models = get_models()
    missing_keys = [k for k in keys if not is_model_ready(k)]
    needed_gb = sum(models[k].get("size_gb", 0) for k in missing_keys if k in models)
    free_gb = get_free_disk_space_gb()
    return {
        "needed_gb": round(needed_gb, 1),
        "free_gb": round(free_gb, 1),
        "enough": free_gb >= needed_gb,
        "missing_keys": missing_keys,
    }


def detect_command_line_tools() -> bool:
    try:
        subprocess.run(
            ["xcode-select", "-p"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def install_command_line_tools():
    subprocess.Popen(["xcode-select", "--install"])


# Function: Detect cpu cores — describes what this function does
def detect_cpu_cores():
    try:
        hardware_info = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"]
        ).decode()
        cores_line = next(
            (
                line
                for line in hardware_info.splitlines()
                if "Total Number of Cores:" in line
            ),
            None,
        )
        if cores_line:
            return cores_line.split(":")[-1].strip().split()[0]
        return "?"
    except:
        return "?"


# Function: Detect gpu cores — describes what this function does
def detect_gpu_cores():
    try:
        display_info = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"]
        ).decode()
        for line in display_info.splitlines():
            if (
                "Total Number of Cores:" in line
                and "GPU" in display_info[: display_info.find(line)]
            ):
                return line.split(":")[-1].strip()
        return "?"
    except:
        return "?"


def detect_os_name() -> str:
    """Return the OS product name (e.g. 'macOS')."""
    try:
        return subprocess.check_output(["sw_vers", "-productName"], text=True).strip()
    except Exception:
        return platform.system()


def detect_os_version() -> str:
    """Return the OS version string (e.g. '15.3.1')."""
    try:
        return subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip()
    except Exception:
        return platform.mac_ver()[0] or "unknown"


def detect_ne_cores() -> int:
    """Return the Neural Engine core count via ioreg, falling back to chip-name inference."""
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-c", "H11ANEIn", "-d", "4"],
            capture_output=True, text=True, timeout=5
        ).stdout
        m = re.search(r'"ANEDevicePropertyNumANECores"=(\d+)', out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    # Fallback: chip-name based (accurate for all current Apple Silicon)
    chip = detect_apple_chip().lower()
    return 32 if "ultra" in chip else 16


# Function: Detect mlcore — describes what this function does
def detect_mlcore():
    try:
        mlcore = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"]
        ).decode()
        chip_line = next(
            (line for line in mlcore.splitlines() if "Chip:" in line), None
        )
        if not chip_line:
            return "?"
        chip = chip_line.split(":")[-1].strip()
        if "Ultra" in chip:
            return "32"
        elif "M1" in chip or "M2" in chip or "M3" in chip or "M4" in chip:
            return "16"
        else:
            return "?"
    except:
        return "?"


# Function: Get hardware id — describes what this function does
def get_hardware_id():
    chip = detect_apple_chip()
    ram = detect_memory_gb()
    core_cpu = detect_cpu_cores()
    core_gpu = detect_gpu_cores()
    ml = detect_mlcore()
    safe_chip = chip.replace(" ", "").replace("/", "-")
    return f"{safe_chip}_CPU{core_cpu}core_GPU{core_gpu}core_ML{ml}core_RAM{ram}GB"


def get_machine_uuid():
    try:
        result = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"]
        ).decode()
        for line in result.splitlines():
            if "Hardware UUID" in line:
                return line.split(":")[-1].strip()

        # Fallback al metodo attuale se non trova l'UUID
        return get_hardware_id()
    except Exception:
        return get_hardware_id()


def run_memory_purge():
    """Run memory purge using the C binary"""
    print("[DEBUG] Purge RAM started")

    try:
        if is_frozen_app():
            bin_path = Path(sys.executable).parent / "purge_ram_with_gui_auth"
        else:
            bin_path = Path(__file__).parent / "purge_ram_with_gui_auth"

        print("[DEBUG] is_frozen_app:", is_frozen_app())
        print("[DEBUG] sys.executable:", sys.executable)
        print("[DEBUG] cwd:", os.getcwd())
        print(f"[DEBUG] Looking for purge binary at: {bin_path}")

        if not bin_path.exists():
            raise FileNotFoundError(f"Purge binary not found at: {bin_path}")

        result = subprocess.run([str(bin_path)], capture_output=True, text=True)
        output = result.stdout.strip()
        print("[DEBUG] Purge binary output:", output)

        if output == "0":
            print("User canceled RAM purge")
            return False
        elif output == "1":
            print("RAM purge executed")
            return True
        else:
            print("Unexpected output from purge binary:", output)
            return False

    except Exception as e:
        print(f"[ERROR] Failed to run memory purge: {e}")
        log_exception_context("run_memory_purge")
        raise e


# Function: Open csv — describes what this function does
def open_csv():
    path = get_home_dir() / "benchmark_results.csv"
    if path.exists():
        os.system(f"open '{path}'")
    else:
        messagebox.showwarning("Missing CSV", "You must run at least one benchmark.")


# Function: Export csv — describes what this function does
def export_csv():
    path = get_home_dir() / "benchmark_results.csv"
    if path.exists():
        dest = filedialog.asksaveasfilename(defaultextension=".csv")
        if dest:
            shutil.copy(path, dest)
            messagebox.showinfo("Exported", f"Exported in\n{dest}")


def is_model_ready(key: str):
    path = get_models_dir() / key_to_folder(key)
    model_ready = False
    model_exists = path.exists() and any(path.iterdir())
    if model_exists:
        # check if download is completed
        completed_flag = path / ".completed"
        model_ready = completed_flag.exists()
    return model_ready


def delete_model(key, progress_callback=None):
    from pathlib import Path
    import shutil
    import globals_state as gs

    gs.cancel_requested = False  # reset annullamento

    path = get_models_dir() / key_to_folder(key)
    completed_flag = path / ".completed"

    print(f"[DEBUG] Eliminazione modello: {key}")
    print(f"[DEBUG] Path: {path}")

    try:
        if completed_flag.exists():
            completed_flag.unlink()
            print(f"[DEBUG] Flag .completed rimossa")

        if path.exists():
            shutil.rmtree(path)
            print(f"[DEBUG] Cartella {path} rimossa con successo")
        else:
            print(f"[DEBUG] Nessuna cartella da rimuovere per {path}")

        if progress_callback:
            progress_callback()

        return {"success": True, "message": "Model deleted successfully."}

    except Exception as e:
        print(f"[DEBUG] Errore durante la rimozione: {e}")
        return {"success": False, "message": f"Delete failed: {e}"}


# def download_model(key, progress_callback=None):
#     import globals_state as gs

#     print(f"[DEBUG] sys.executable: {sys.executable}")
#     print(f"[DEBUG] is_frozen_app(): {is_frozen_app()}")

#     download_results = []
#     model = get_models()[key]
#     repo = model["repo"]
#     path = get_models_dir() / key
#     print(f"[DEBUG] path to download: {path}")

#     completed_flag = path / ".completed"
#     if completed_flag.exists():
#         print(f"[DEBUG] Flag di completamento trovata: {completed_flag}")
#         return {"success": True, "message": "Model already exists."}
#     elif path.exists():
#         print(f"[DEBUG] Rimozione cartella incompleta: {path}")
#         try:
#             shutil.rmtree(path)
#             print(f"[DEBUG] Cartella rimossa con successo")
#         except Exception as e:
#             print(f"[DEBUG] Errore nella rimozione della cartella: {e}")

#     # Verifica che la cartella sia stata effettivamente rimossa
#     if path.exists():
#         print(f"[DEBUG] ATTENZIONE: La cartella esiste ancora dopo la rimozione")
#     else:
#         print(f"[DEBUG] La cartella è stata rimossa correttamente")

#     def monitor():
#         while downloading and not gs.cancel_requested:  # Aggiungi questo controllo
#             size_mb = get_folder_size(path) / 1024 / 1024
#             # print(f"Debug: gs.filename = {getattr(gs, 'filename', 'NOT DEFINED')}")
#             downloaded_text = f"Downloaded: {size_mb:.1f} MB " + (
#                 f"of {gs.filename}" if gs.filename else ""
#             )
#             print(downloaded_text)
#             download_results.append(downloaded_text)
#             if progress_callback:
#                 progress_callback(downloaded_text)
#             time.sleep(0.5)

#     global downloading
#     downloading = True
#     thread = threading.Thread(target=monitor)
#     thread.start()

#     if is_frozen_app():
#         # App compilata: il file viene copiato come eseguibile senza estensione
#         script_path = Path(sys.executable).parent / "download_worker"
#     else:
#         # Durante sviluppo: esegue direttamente il file Python
#         script_path = Path(__file__).parent / "download_worker.py"
#     print(f"[DEBUG] script_path: {script_path} ")
#     print(
#         f"[DEBUG] exists: {script_path.exists()}, is_file: {script_path.is_file()}, is_executable: {os.access(script_path, os.X_OK)}"
#     )

#     log_file = get_home_dir() / "download_global.log"

#     try:
#         from huggingface_hub import list_repo_files

#         quant = str(model.get("quantization", "")).upper()

#         # Log prima dell'esecuzione
#         with open(log_file, "a", encoding="utf-8") as f:
#             f.write(f"\n=== {datetime.now().isoformat()} ===\n")
#             f.write("MODEL: " + key + "\n")
#             f.write(f"DOWNLOADING FROM REPO: {repo}\n")

#         # Controlla se l'annullamento è stato richiesto
#         if gs.cancel_requested:
#             downloading = False
#             thread.join()
#             return {"success": False, "message": "Download cancelled."}

#         if model["format"] == "GGUF" and quant in get_quant_map():
#             repo_id = model["repo"]

#             # Usa l'API per ottenere i file invece della richiesta HTTP
#             all_files = list_repo_files(repo_id)

#             include_file = None
#             for q in get_quant_map()[quant]:
#                 for fname in all_files:
#                     if q in fname.upper() and fname.endswith(".gguf"):
#                         include_file = fname
#                         break
#                 if include_file:
#                     break

#             if not include_file:
#                 return {
#                     "success": False,
#                     "message": f"No matching GGUF file found for quantization {quant} in {repo_id}",
#                 }

#             # Scarica solo il file selezionato
#             filename = include_file
#             print(f"[DEBUG] Selected GGUF file: {filename}")

#             # Controlla se l'annullamento è stato richiesto
#             if gs.cancel_requested:
#                 downloading = False
#                 thread.join()
#                 return {"success": False, "message": "Download cancelled by user."}

#             try:
#                 gs.filename = filename
#                 download_results.append(f"Download {filename}")

#                 if is_frozen_app():
#                     # App compilata: esegui direttamente l'eseguibile senza usare sys.executable come interprete
#                     cmd = [
#                         str(script_path),
#                         repo_id,
#                         filename,
#                         str(path),
#                     ]
#                 else:
#                     # Durante sviluppo: esegue il file Python con l'interprete Python
#                     cmd = [
#                         sys.executable,
#                         str(script_path),
#                         repo_id,
#                         filename,
#                         str(path),
#                     ]
#                 env = os.environ.copy()
#                 env["HF_HUB_DISABLE_TELEMETRY"] = "1"
#                 if "HF_TOKEN" in os.environ:
#                     env["HF_TOKEN"] = os.environ["HF_TOKEN"]
#                 proc = Popen(
#                     cmd,
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.PIPE,
#                     text=True,
#                     env=env,
#                 )
#                 print(f"[DEBUG] Started process PID: {proc.pid}")

#                 def stream_output(pipe, label):
#                     for line in iter(pipe.readline, ""):
#                         if line:
#                             print(f"[{label}] {line.strip()}", flush=True)

#                 stdout_thread = threading.Thread(
#                     target=stream_output, args=(proc.stdout, "stdout"), daemon=True
#                 )
#                 stderr_thread = threading.Thread(
#                     target=stream_output, args=(proc.stderr, "stderr"), daemon=True
#                 )
#                 stdout_thread.start()
#                 stderr_thread.start()
#                 # Attendi direttamente la fine del processo
#                 while True:
#                     retcode = proc.poll()
#                     if retcode is not None:
#                         break  # Processo terminato
#                     if gs.cancel_requested:
#                         proc.terminate()
#                         proc.wait()
#                         downloading = False
#                         thread.join()
#                         return {
#                             "success": False,
#                             "message": "Download cancelled by user.",
#                         }
#                     time.sleep(0.3)

#                 # Qui il processo è terminato, ma controlla anche il codice di uscita
#                 if proc.returncode != 0:
#                     downloading = False
#                     thread.join()
#                     return {
#                         "success": False,
#                         "message": f"Process exited with code {proc.returncode}",
#                     }

#             except Exception as e:
#                 download_results.append(f"Error downloading {filename}: {str(e)}")

#             with open(log_file, "a", encoding="utf-8") as f:
#                 f.write("DOWNLOAD RESULTS:\n" + "\n".join(download_results) + "\n")

#         else:
#             # Scarica tutti i file dal repository
#             repo_id = repo
#             try:
#                 all_files = list_repo_files(repo_id)

#                 for filename in all_files:
#                     # Controlla se l'annullamento è stato richiesto
#                     if gs.cancel_requested:
#                         downloading = False
#                         thread.join()
#                         return {
#                             "success": False,
#                             "message": "Download cancelled by user.",
#                         }

#                     try:
#                         gs.filename = filename
#                         download_results.append(f"Download {filename}")

#                         if is_frozen_app():
#                             # App compilata: esegui direttamente l'eseguibile senza usare sys.executable come interprete
#                             cmd = [
#                                 str(script_path),
#                                 repo_id,
#                                 filename,
#                                 str(path),
#                             ]
#                         else:
#                             # Durante sviluppo: esegue il file Python con l'interprete Python
#                             cmd = [
#                                 sys.executable,
#                                 str(script_path),
#                                 repo_id,
#                                 filename,
#                                 str(path),
#                             ]
#                         env = os.environ.copy()
#                         env["HF_HUB_DISABLE_TELEMETRY"] = "1"
#                         if "HF_TOKEN" in os.environ:
#                             env["HF_TOKEN"] = os.environ["HF_TOKEN"]
#                         proc = Popen(
#                             cmd,
#                             stdout=subprocess.PIPE,
#                             stderr=subprocess.PIPE,
#                             text=True,
#                             env=env,
#                         )
#                         print(f"[DEBUG] Started process PID: {proc.pid}")

#                         def stream_output(pipe, label):
#                             for line in iter(pipe.readline, ""):
#                                 if line:
#                                     print(f"[{label}] {line.strip()}")

#                         stdout_thread = threading.Thread(
#                             target=stream_output,
#                             args=(proc.stdout, "stdout"),
#                             daemon=True,
#                         )
#                         stderr_thread = threading.Thread(
#                             target=stream_output,
#                             args=(proc.stderr, "stderr"),
#                             daemon=True,
#                         )
#                         stdout_thread.start()
#                         stderr_thread.start()

#                         # Attendi direttamente la fine del processo
#                         while True:
#                             retcode = proc.poll()
#                             if retcode is not None:
#                                 break  # Processo terminato
#                             if gs.cancel_requested:
#                                 proc.terminate()
#                                 proc.wait()
#                                 downloading = False
#                                 thread.join()
#                                 return {
#                                     "success": False,
#                                     "message": "Download cancelled by user.",
#                                 }
#                             time.sleep(0.3)

#                         # Qui il processo è terminato, ma controlla anche il codice di uscita
#                         if proc.returncode != 0:
#                             downloading = False
#                             thread.join()
#                             return {
#                                 "success": False,
#                                 "message": f"Process exited with code {proc.returncode}",
#                             }

#                     except Exception as e:
#                         download_results.append(
#                             f"Error downloading {filename}: {str(e)}"
#                         )

#                 with open(log_file, "a", encoding="utf-8") as f:
#                     f.write("DOWNLOAD RESULTS:\n" + "\n".join(download_results) + "\n")
#             except Exception as e:
#                 return {
#                     "success": False,
#                     "message": f"Repository not found or error: {repo}. Details: {str(e)}",
#                 }

#         # Controlla se l'annullamento è stato richiesto
#         if gs.cancel_requested:
#             downloading = False
#             thread.join()
#             return {"success": False, "message": "Download cancelled by user."}

#         downloading = False
#         thread.join()

#         if model["format"] == "GGUF" and not any(path.glob("*.gguf")):
#             return {"success": False, "message": f"No GGUF files found in {path}"}

#         completed_flag.touch()

#         clean_huggingface_cache()

#         return {
#             "success": True,
#             "message": "Download completed successfully.",
#         }

#     except Exception as e:
#         downloading = False
#         thread.join()
#         # Log dell'errore
#         with open(log_file, "a", encoding="utf-8") as f:
#             f.write("ERROR:\n" + str(e) + "\n")
#         return {"success": False, "message": str(e)}


def clean_huggingface_cache():
    try:
        if is_frozen_app():
            script_path = Path(sys.executable).parent / "download_worker"
            cmd = [str(script_path), "--clean-cache"]
        else:
            script_path = Path(__file__).parent / "download_worker.py"
            cmd = [sys.executable, str(script_path), "--clean-cache"]
        env = os.environ.copy()
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        if "HF_TOKEN" in os.environ:
            env["HF_TOKEN"] = os.environ["HF_TOKEN"]
        proc = Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        stdout, stderr = proc.communicate()
        if proc.returncode == 0:
            print(f"[DEBUG] Cache clean success:\n{stdout.strip()}")
        else:
            print(f"[DEBUG] Cache clean error:\n{stderr.strip()}")
    except Exception:
        import traceback

        print(f"[DEBUG] Exception during cache clean:\n{traceback.format_exc()}")



def format_timestamp_it(timestamp: str) -> str:
    """Converte un timestamp ISO in data/ora italiana leggibile."""
    try:
        locale.setlocale(locale.LC_TIME, "it_IT.UTF-8")
    except:
        pass  # fallback in caso locale non disponibile

    try:
        dt = datetime.fromisoformat(timestamp)
        # "12 aprile 2025, ore 01:47"
        return dt.strftime("%d %B %Y, ore %H:%M").lower()
    except ValueError:
        return ""


def format_iso_timestamp(timestamp: str) -> str:
    """Restituisce il timestamp ISO in formato 'YYYY-MM-DD HH:MM:SS'."""
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return ""


def get_token_steps():
    basic_tokens = [1024]
    advanced_tokens = [4096]
    return basic_tokens, advanced_tokens


def format_tokens(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def estimate_tokens(
    text: str,
    add_bos: bool = False,
    margin: float = 0.10,
) -> int:
    words = re.findall(r"\b\w+\b", text)
    base_count = len(words)
    estimated = base_count * (1 + margin)
    if add_bos:
        estimated += 1
    return math.ceil(estimated)


def sanitize_text(text: str):
    # Sostituisce gli spazi con underscore e converte in minuscolo
    sanitized_text = text.replace(" ", "_").lower()
    return sanitized_text

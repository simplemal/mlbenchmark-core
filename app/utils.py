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


from pathlib import Path
from subprocess import Popen, PIPE


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


# Stable numeric IDs for each tier — independent of model name or tier display name.
_TIER_ID = {"Light": 1, "Speed": 2, "Flash": 3, "Blaze": 4, "Ultra": 5}


def get_available_ram_gb() -> float:
    """Return the amount of RAM available to the system in GB, including purgeable memory."""
    return psutil.virtual_memory().available / (1024**3)


def print_memory_usage(label=""):
    process = psutil.Process()
    mem_bytes = process.memory_info().rss  # Resident Set Size
    mem_gb = mem_bytes / (1024**3)
    print(f"[RAM USAGE] {label}: {mem_gb:.2f} GB (RSS)")


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

def get_available_tiers(available_gb: float = None) -> list:
    """Return tiers safely runnable with the current available RAM."""
    if available_gb is None:
        available_gb = get_available_ram_gb()
    return [t for t in BENCHMARK_TIERS if available_gb >= t["min_ram_gb"]]


# Function: Get folder size — describes what this function does

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


# Function: Detect memory gb — describes what this function does
def detect_memory_gb():
    return round(psutil.virtual_memory().total / 1024**3)


# Function: Detect cpu cores — describes what this function does

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


# Function: Detect mlcore — describes what this function does

# Function: Get hardware id — describes what this function does

# Function: Open csv — describes what this function does

# Function: Export csv — describes what this function does

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



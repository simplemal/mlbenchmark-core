import os
import sys
import time
import threading
import subprocess
from subprocess import Popen
import shutil
from pathlib import Path
from datetime import datetime

from utils import (
    get_models,
    get_models_dir,
    key_to_folder,
    get_home_dir,
    get_folder_size,
    is_frozen_app,
    get_quant_map,
)
import globals_state as gs

# Global variable for download monitoring
downloading = False


def select_files_to_download(model, all_files):
    """
    Select which files to download based on model format and quantization.

    Args:
        model: Model configuration dictionary
        all_files: List of all files in the repository

    Returns:
        List of files to download
    """
    files_to_download = []

    if model["format"] == "GGUF":
        # Logic for GGUF models - select specific file based on quantization
        # Collects ALL matching .gguf files (handles sharded models like 00001-of-00002)
        quant = str(model.get("quantization", "")).upper()
        if quant in get_quant_map():
            for q in get_quant_map()[quant]:
                for fname in all_files:
                    if q in fname.upper() and fname.endswith(".gguf"):
                        files_to_download.append(fname)
                if files_to_download:
                    break
    elif model["format"] == "MLX":
        # Logic for MLX models - download all files
        files_to_download = all_files
    elif model["format"] == "MLC":
        # Logic for MLC models - download only .mlc files
        files_to_download = all_files
    else:
        # Default case - download all files
        files_to_download = all_files

    return files_to_download


def setup_download_process(script_path, repo_id, filename, path):
    """
    Setup the download process command and environment.

    Args:
        script_path: Path to the download worker script
        repo_id: Repository ID on Hugging Face
        filename: File to download
        path: Path to save the file

    Returns:
        tuple: (cmd, env) - Command and environment for the process
    """
    if is_frozen_app():
        # Compiled app: run executable directly
        cmd = [
            str(script_path),
            repo_id,
            filename,
            str(path),
        ]
    else:
        # Development: run Python script with interpreter
        cmd = [
            sys.executable,
            str(script_path),
            repo_id,
            filename,
            str(path),
        ]

    # Setup environment
    env = os.environ.copy()
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if "HF_TOKEN" in os.environ:
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]

    return cmd, env


def run_download_process(cmd, env):
    """
    Run download process and stream output.

    Args:
        cmd: Command list
        env: Environment dictionary

    Returns:
        subprocess.Popen: Process object
    """
    proc = Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    print(f"[DEBUG] Started process PID: {proc.pid}")

    def stream_output(pipe, label):
        for line in iter(pipe.readline, ""):
            if line:
                print(f"[{label}] {line.strip()}", flush=True)

    stdout_thread = threading.Thread(
        target=stream_output, args=(proc.stdout, "stdout"), daemon=True
    )
    stderr_thread = threading.Thread(
        target=stream_output, args=(proc.stderr, "stderr"), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    return proc


def cleanup_partial_downloads(keys: list):
    """
    Remove partially downloaded model folders (those without a .completed flag).
    Called when the user cancels a benchmark run mid-download.
    """
    models_dir = get_models_dir()
    for key in keys:
        path = models_dir / key_to_folder(key)
        completed_flag = path / ".completed"
        if path.exists() and not completed_flag.exists():
            try:
                shutil.rmtree(path)
                print(f"[CLEANUP] Removed partial download: {key}")
            except Exception as e:
                print(f"[CLEANUP] Error removing partial download {key}: {e}")


def monitor_download_size(path, progress_callback=None, total_bytes=0):
    """
    Monitor download size and update progress.

    Args:
        path: Path to the download directory
        progress_callback: Callback function for progress updates (downloaded_bytes, total_bytes)
        total_bytes: Expected total size in bytes

    Returns:
        function: Monitor function for threading
    """

    def monitor():
        # hf_hub_download writes to HF cache first, then copies to local_dir.
        # We track whichever is larger: the target path or the HF cache entry.
        hf_cache_base = Path.home() / ".cache" / "huggingface" / "hub"

        while downloading and not gs.cancel_requested:
            target_bytes = get_folder_size(path)

            # Also check HF cache for the repo being downloaded
            cache_bytes = 0
            if gs.filename and hf_cache_base.exists():
                for repo_dir in hf_cache_base.iterdir():
                    if repo_dir.is_dir():
                        cache_bytes = max(cache_bytes, get_folder_size(repo_dir))

            downloaded_bytes = max(target_bytes, cache_bytes)
            size_mb = downloaded_bytes / 1024 / 1024
            print(f"Downloaded: {size_mb:.1f} MB" + (f" of {gs.filename}" if gs.filename else ""))

            if progress_callback:
                progress_callback(downloaded_bytes, total_bytes)
            time.sleep(0.5)

    return monitor


def download_model(key, progress_callback=None):
    """
    Download a model from Hugging Face repository.

    Args:
        key: Model key in the models dictionary
        progress_callback: Callback function for progress updates

    Returns:
        dict: Result of the download operation
    """
    print(f"[DEBUG] sys.executable: {sys.executable}")
    print(f"[DEBUG] is_frozen_app(): {is_frozen_app()}")

    download_results = []
    model = get_models()[key]
    repo = model["repo"]
    path = get_models_dir() / key_to_folder(key)
    print(f"[DEBUG] path to download: {path}")

    # Check if model is already downloaded
    completed_flag = path / ".completed"
    if completed_flag.exists():
        print(f"[DEBUG] Completion flag found: {completed_flag}")
        return {"success": True, "message": "Model already exists."}
    elif path.exists():
        print(f"[DEBUG] Removing incomplete folder: {path}")
        try:
            shutil.rmtree(path)
            print(f"[DEBUG] Folder successfully removed")
        except Exception as e:
            print(f"[DEBUG] Error removing folder: {e}")

    # Verify folder was actually removed
    if path.exists():
        print(f"[DEBUG] WARNING: Folder still exists after removal")
    else:
        print(f"[DEBUG] Folder was successfully removed")

    # Start monitoring thread
    global downloading
    downloading = True
    total_bytes = int(model.get("size_gb", 0) * 1024**3)
    monitor_func = monitor_download_size(path, progress_callback, total_bytes)
    thread = threading.Thread(target=monitor_func)
    thread.start()

    # Determine download worker path
    if is_frozen_app():
        # Compiled app: executable without extension
        script_path = Path(sys.executable).parent / "download_worker"
    else:
        # Development: Python file
        script_path = Path(__file__).parent / "download_worker.py"

    print(f"[DEBUG] script_path: {script_path} ")
    print(
        f"[DEBUG] exists: {script_path.exists()}, is_file: {script_path.is_file()}, is_executable: {os.access(script_path, os.X_OK)}"
    )

    log_file = get_home_dir() / "download_global.log"

    try:
        from huggingface_hub import list_repo_files

        # Log before execution
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n=== {datetime.now().isoformat()} ===\n")
            f.write("MODEL: " + key + "\n")
            f.write(f"DOWNLOADING FROM REPO: {repo}\n")

        # Check if cancellation was requested
        if gs.cancel_requested:
            downloading = False
            thread.join()
            return {"success": False, "message": "Download cancelled."}

        # Get all repository files
        try:
            repo_id = repo
            all_files = list_repo_files(repo_id)
        except Exception as e:
            downloading = False
            thread.join()
            return {
                "success": False,
                "message": f"Repository not found or error: {repo}. Details: {str(e)}",
            }

        # Select files to download based on model format
        files_to_download = select_files_to_download(model, all_files)

        # Check if there are files to download
        if not files_to_download:
            downloading = False
            thread.join()
            return {
                "success": False,
                "message": f"No matching {model['format']} files found in {repo_id}",
            }

        # Download all selected files
        for filename in files_to_download:
            # Check if cancellation was requested
            if gs.cancel_requested:
                downloading = False
                thread.join()
                return {
                    "success": False,
                    "message": "Download cancelled by user.",
                }

            try:
                gs.filename = filename
                download_results.append(f"Download {filename}")

                # Setup and start download process
                cmd, env = setup_download_process(script_path, repo_id, filename, path)
                proc = run_download_process(cmd, env)

                # Wait for process completion
                while True:
                    retcode = proc.poll()
                    if retcode is not None:
                        proc.wait()  # reap zombie
                        break  # Process completed
                    if gs.cancel_requested:
                        proc.terminate()
                        proc.wait()
                        downloading = False
                        thread.join()
                        return {
                            "success": False,
                            "message": "Download cancelled by user.",
                        }
                    time.sleep(0.3)

                # Check exit code
                if proc.returncode != 0:
                    downloading = False
                    thread.join()
                    return {
                        "success": False,
                        "message": f"Process exited with code {proc.returncode}",
                    }

            except Exception as e:
                download_results.append(f"Error downloading {filename}: {str(e)}")

        # Log download results
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("DOWNLOAD RESULTS:\n" + "\n".join(download_results) + "\n")

        # Final cancellation check
        if gs.cancel_requested:
            downloading = False
            thread.join()
            return {"success": False, "message": "Download cancelled by user."}

        downloading = False
        thread.join()

        # Verify expected files were downloaded correctly
        if model["format"] == "GGUF" and not any(path.glob("*.gguf")):
            return {"success": False, "message": f"No GGUF files found in {path}"}

        # Mark download as complete
        completed_flag.touch()

        return {
            "success": True,
            "message": "Download completed successfully.",
        }

    except Exception as e:
        downloading = False
        thread.join()
        # Log error
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("ERROR:\n" + str(e) + "\n")
        return {"success": False, "message": str(e)}

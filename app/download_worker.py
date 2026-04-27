import sys
import traceback


def run_download(repo_id: str, filename: str, path: str):
    # Disabilita solo gli specifici warning che abbiamo visto
    import warnings

    warnings.filterwarnings("ignore", message=".*resume_download.*")
    warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")

    from huggingface_hub import hf_hub_download

    print(f"[DOWNLOAD STARTED] {filename} da {repo_id}", flush=True)
    try:
        import os

        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=path,
            resume_download=True,
            local_dir_use_symlinks=False,
        )
        print(f"[DOWNLOAD COMPLETED] {filename}", flush=True)
    except Exception as e:
        print(f"[DOWNLOAD ERROR] {filename}: {str(e)}", flush=True)
        sys.exit(1)


def clean_cache():
    try:
        from huggingface_hub import scan_cache_dir
        import os
        import shutil

        # Misura la dimensione della cache prima della pulizia
        info_before = scan_cache_dir()
        size_before = info_before.size_on_disk_str
        bytes_before = info_before.size_on_disk

        # Trova il percorso della cache in modo dinamico
        cache_path = None
        if info_before.repos:
            sample_repo = list(info_before.repos)[0]
            repo_path = sample_repo.repo_path
            # Risaliamo alla directory huggingface, poi scendiamo in /hub
            cache_path = os.path.join(
                os.path.dirname(os.path.dirname(repo_path)), "hub"
            )

        if cache_path and os.path.exists(cache_path):
            # Pulisci la directory della cache
            for item in os.listdir(cache_path):
                item_path = os.path.join(cache_path, item)
                try:
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    print(
                        f"[CACHE WARNING] Failed to delete {item_path}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )

            # Misura la dimensione della cache dopo la pulizia
            info_after = scan_cache_dir()
            size_after = info_after.size_on_disk_str
            bytes_after = info_after.size_on_disk

            # Calcola quanto spazio è stato liberato
            freed_space = bytes_before - bytes_after

            # Converti in formato leggibile
            if freed_space >= 1024**3:
                freed_space_str = f"{freed_space / 1024**3:.1f}G"
            elif freed_space >= 1024**2:
                freed_space_str = f"{freed_space / 1024**2:.1f}M"
            elif freed_space >= 1024:
                freed_space_str = f"{freed_space / 1024:.1f}K"
            else:
                freed_space_str = f"{freed_space}B"

            print(
                f"[CACHE CLEANED] Hugging Face cache cleared. Freed {freed_space_str} (from {size_before} to {size_after}).",
                flush=True,
            )
        else:
            print(
                "[CACHE INFO] Cache directory not found or already empty.", flush=True
            )

        sys.exit(0)
    except Exception as e:
        print("[CACHE ERROR] Failed to clean cache:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    print("[DEBUG] download_worker.py called with:", sys.argv[1:], flush=True)

    # Comando di pulizia cache
    if len(sys.argv) == 2 and sys.argv[1] == "--clean-cache":
        clean_cache()

    # previous verion that works, but exception
    # if len(sys.argv) < 4
    if len(sys.argv) < 4 or any(arg.startswith("-") for arg in sys.argv[1:4]):
        print("[ERROR] Missing parameters", flush=True)
        sys.exit(1)

    run_download(*sys.argv[1:4])

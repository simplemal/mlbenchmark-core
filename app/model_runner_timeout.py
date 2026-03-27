import threading
import time
import traceback


def model_runner_timeout(func, timeout_seconds, *args, **kwargs):
    """
    Esegue una funzione con un timeout specificato.

    Args:
        func: La funzione da eseguire
        timeout_seconds: Timeout in secondi
        *args, **kwargs: Argomenti da passare alla funzione

    Returns:
        Se la funzione termina entro il timeout, restituisce il risultato della funzione.
        Se si verifica un timeout, restituisce ("", "timeout", 0).
        Se si verifica un'eccezione, restituisce ("", "error", 0).
    """
    result = [None]
    exception = [None]
    exception_tb = [None]
    completed = [False]

    def target():
        try:
            result[0] = func(*args, **kwargs)
            completed[0] = True
        except Exception as e:
            exception[0] = e
            exception_tb[0] = traceback.format_exc()
            completed[0] = True

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()

    # Attendi che il thread termini o che scada il timeout
    start_time = time.time()
    while not completed[0] and (time.time() - start_time) < timeout_seconds:
        time.sleep(0.1)

    if not completed[0]:
        elapsed = time.time() - start_time
        print(f"[TIMEOUT] Operation timed out after {elapsed:.1f}s (limit={timeout_seconds}s)")
        return "", "timeout", 0

    if exception[0]:
        print(f"[ERROR] Operation failed: {type(exception[0]).__name__}: {exception[0]}")
        if exception_tb[0]:
            print(f"[ERROR] Full traceback:\n{exception_tb[0]}")
        return "", "error", 0

    # Operazione completata con successo
    return result[0]

# ============================================================
# ÍTEM (e) - Oversubscription en bs_numpy.py
# ============================================================

import os
import sys
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

RESULTADOS_DIR = Path("resultados")
TXT_PATH = RESULTADOS_DIR / "resultados_e.txt"

P_MAX = os.cpu_count() or 1


# ============================================================
# GUARDAR Y MOSTRAR RESULTADOS
# ============================================================

def log(texto=""):

    RESULTADOS_DIR.mkdir(
        exist_ok=True
    )

    print(texto)

    with open(
        TXT_PATH,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            str(texto) + "\n"
        )


# ============================================================
# WORKER PARA INSPECCIONAR THREADS
# ============================================================

def inspeccionar_worker(worker_id):

    import os
    import numpy as np

    from threadpoolctl import (
        threadpool_info
    )

    # Forzar una operación de álgebra lineal
    # para inicializar BLAS/MKL.
    rng = np.random.default_rng(
        1000 + worker_id
    )

    A = rng.normal(
        size=(200, 200)
    )

    b = rng.normal(
        size=200
    )

    _ = np.linalg.solve(
        A,
        b
    )

    info = threadpool_info()

    return (
        worker_id,
        os.getpid(),
        info
    )


# ============================================================
# MODO 1: INSPECCIÓN
# ============================================================

def inspeccionar(p):

    from joblib import (
        Parallel,
        delayed
    )

    log()
    log("=" * 70)
    log(f"INSPECCIÓN DE THREADS - p = {p}")
    log("=" * 70)

    log(
        f"Cores lógicos disponibles: {P_MAX}"
    )

    # Se ejecuta threadpool_info dentro
    # de cada proceso worker.
    resultados = Parallel(
        n_jobs=p,
        backend="multiprocessing"
    )(
        delayed(
            inspeccionar_worker
        )(i)

        for i in range(p)
    )

    threads_blas = []
    threads_openmp = []

    for (
        worker_id,
        pid,
        pools
    ) in resultados:

        log()
        log(
            f"Worker {worker_id} | PID {pid}"
        )

        for pool in pools:

            api = pool.get(
                "user_api"
            )

            backend = pool.get(
                "internal_api"
            )

            n_threads = pool.get(
                "num_threads"
            )

            log(
                f"  API={api} | "
                f"backend={backend} | "
                f"threads={n_threads}"
            )

            if (
                api == "blas"
                and n_threads is not None
            ):

                threads_blas.append(
                    int(n_threads)
                )

            if (
                api == "openmp"
                and n_threads is not None
            ):

                threads_openmp.append(
                    int(n_threads)
                )

    # --------------------------------------------------------
    # Resumen
    # --------------------------------------------------------

    if threads_blas:

        t_blas = max(
            threads_blas
        )

        potencial = (
            p * t_blas
        )

        log()
        log("RESUMEN")

        log(
            f"Threads BLAS por proceso: "
            f"t = {t_blas}"
        )

        if threads_openmp:

            log(
                f"Threads OpenMP observados: "
                f"{max(threads_openmp)}"
            )

        log(
            f"Paralelismo potencial: "
            f"p*t = {p}*{t_blas} "
            f"= {potencial}"
        )

        log(
            f"Cores lógicos: {P_MAX}"
        )

        if potencial > P_MAX:

            log(
                "Indicio potencial de "
                "oversubscription: SÍ"
            )

        else:

            log(
                "Indicio potencial de "
                "oversubscription: NO"
            )

    log()


# ============================================================
# MODO 2: EJECUCIÓN ORIGINAL
# ============================================================

def ejecutar_original(p):

    from time import (
        perf_counter,
        sleep
    )

    from common import (
        generate_data
    )

    from bs_numpy import (
        bootstrap_numpy
    )

    log()
    log("=" * 70)
    log(f"EJECUCIÓN ORIGINAL - p = {p}")
    log("=" * 70)

    log(
        "Sin límite manual de threads internos."
    )

    log(
        "Mira htop durante esta ejecución."
    )

    # Da tiempo para mirar htop.
    sleep(3)

    X, y, _ = generate_data()

    inicio = perf_counter()

    bootstrap_numpy(
        X,
        y,
        p
    )

    tiempo = (
        perf_counter()
        - inicio
    )

    log(
        f"Tiempo original: "
        f"{tiempo:.2f} s"
    )

    log()


# ============================================================
# MODO 3: EJECUCIÓN CORREGIDA t = 1
# ============================================================

def ejecutar_limitado(p):

    # --------------------------------------------------------
    # IMPORTANTE:
    # Estas variables se fijan ANTES de importar NumPy,
    # common o bs_numpy.
    # --------------------------------------------------------

    os.environ[
        "OMP_NUM_THREADS"
    ] = "1"

    os.environ[
        "OPENBLAS_NUM_THREADS"
    ] = "1"

    os.environ[
        "MKL_NUM_THREADS"
    ] = "1"

    os.environ[
        "NUMEXPR_NUM_THREADS"
    ] = "1"

    os.environ[
        "BLIS_NUM_THREADS"
    ] = "1"

    from time import (
        perf_counter,
        sleep
    )

    from threadpoolctl import (
        threadpool_limits
    )

    from common import (
        generate_data
    )

    from bs_numpy import (
        bootstrap_numpy
    )

    log()
    log("=" * 70)
    log(
        f"EJECUCIÓN LIMITADA t=1 - "
        f"p = {p}"
    )
    log("=" * 70)

    log(
        "Threads internos limitados a t=1."
    )

    log(
        "Mira htop durante esta ejecución."
    )

    sleep(3)

    X, y, _ = generate_data()

    inicio = perf_counter()

    with threadpool_limits(
        limits=1
    ):

        bootstrap_numpy(
            X,
            y,
            p
        )

    tiempo = (
        perf_counter()
        - inicio
    )

    log(
        f"Tiempo con t=1: "
        f"{tiempo:.2f} s"
    )

    log()


# ============================================================
# RESET
# ============================================================

def reset():

    RESULTADOS_DIR.mkdir(
        exist_ok=True
    )

    with open(
        TXT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "ÍTEM (e) - "
            "OVERSUBSCRIPTION EN bs_numpy.py\n"
        )

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            f"Cores lógicos: {P_MAX}\n"
        )

    print(
        f"Archivo reiniciado: "
        f"{TXT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Uso:"
        )

        print(
            "  python experimento_e.py reset"
        )

        print(
            "  python experimento_e.py "
            "inspect <p>"
        )

        print(
            "  python experimento_e.py "
            "original <p>"
        )

        print(
            "  python experimento_e.py "
            "limited <p>"
        )

        sys.exit(1)

    modo = sys.argv[1]

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    if modo == "reset":

        reset()

        sys.exit(0)

    # --------------------------------------------------------
    # Los otros modos necesitan p.
    # --------------------------------------------------------

    if len(sys.argv) != 3:

        print(
            "Debes indicar un valor de p."
        )

        sys.exit(1)

    p = int(
        sys.argv[2]
    )

    if (
        p < 1
        or p > P_MAX
    ):

        print(
            f"p debe estar entre "
            f"1 y {P_MAX}."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Elegir experimento
    # --------------------------------------------------------

    if modo == "inspect":

        inspeccionar(p)

    elif modo == "original":

        ejecutar_original(p)

    elif modo == "limited":

        ejecutar_limitado(p)

    else:

        print(
            "Modo desconocido."
        )

        print(
            "Usa: inspect, original "
            "o limited."
        )
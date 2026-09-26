# ============================================================
# ÍTEM (e) - Oversubscription en bs_numpy.py
# ============================================================
#
# EJECUCIÓN:
#
#     python inspect_oversubscription_e.py
#
# RESULTADOS:
#
#     resultados/resultados_e.txt
#     resultados/resultados_e.csv
#
# Mientras corre, abrir otra terminal y ejecutar:
#
#     htop
#
# ============================================================

import os
import sys
import csv
import json
import signal
import subprocess
from pathlib import Path


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

RESULTADOS_DIR = Path("resultados")

TXT_PATH = RESULTADOS_DIR / "resultados_e.txt"
CSV_PATH = RESULTADOS_DIR / "resultados_e.csv"

# Archivo temporal usado para comunicar
# los resultados de los procesos hijos.
TEMP_PATH = RESULTADOS_DIR / "_resultado_e_temp.json"

P_MAX = os.cpu_count() or 1

# Valores representativos para inspeccionar.
P_INSPECCION = [
    p for p in [1, 2, 4, 8]
    if p <= P_MAX
]

# Casos suficientes para observar el programa:
# - p=1: referencia
# - p=4: caso con oversubscription potencial
P_EJECUCION = [
    p for p in [1, 4]
    if p <= P_MAX
]

# Tiempo máximo permitido para cada ejecución
# original o limitada.
TIMEOUT_SEGUNDOS = 180


# ============================================================
# UTILIDADES
# ============================================================

def escribir_txt(texto=""):

    RESULTADOS_DIR.mkdir(
        exist_ok=True
    )

    print(texto)

    with open(
        TXT_PATH,
        "a",
        encoding="utf-8"
    ) as archivo:

        archivo.write(
            str(texto) + "\n"
        )


def guardar_temporal(datos):

    RESULTADOS_DIR.mkdir(
        exist_ok=True
    )

    with open(
        TEMP_PATH,
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            datos,
            archivo,
            indent=4
        )


def leer_temporal():

    if not TEMP_PATH.exists():

        return None

    with open(
        TEMP_PATH,
        "r",
        encoding="utf-8"
    ) as archivo:

        datos = json.load(
            archivo
        )

    TEMP_PATH.unlink(
        missing_ok=True
    )

    return datos


# ============================================================
# INFORMACIÓN SIMPLE DE MEMORIA EN WSL/LINUX
# ============================================================

def memoria_disponible_mb():

    try:

        with open(
            "/proc/meminfo",
            "r",
            encoding="utf-8"
        ) as archivo:

            for linea in archivo:

                if linea.startswith(
                    "MemAvailable:"
                ):

                    kb = int(
                        linea.split()[1]
                    )

                    return round(
                        kb / 1024,
                        2
                    )

    except Exception:

        pass

    return None


# ============================================================
# WORKER PARA INSPECCIONAR THREADS
# ============================================================

def inspeccionar_worker(worker_id):

    import numpy as np

    from threadpoolctl import (
        threadpool_info
    )

    rng = np.random.default_rng(
        1000 + worker_id
    )

    # Pequeña operación para inicializar
    # las librerías numéricas.
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

    return {
        "worker": worker_id,
        "pid": os.getpid(),
        "pools": info
    }


# ============================================================
# PROCESO HIJO: INSPECCIÓN
# ============================================================

def child_inspect(p):

    from joblib import (
        Parallel,
        delayed
    )

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

    workers = []

    for resultado in resultados:

        worker = {
            "worker": resultado["worker"],
            "pid": resultado["pid"],
            "pools": []
        }

        for pool in resultado["pools"]:

            api = pool.get(
                "user_api"
            )

            backend = pool.get(
                "internal_api"
            )

            n_threads = pool.get(
                "num_threads"
            )

            worker["pools"].append({
                "api": api,
                "backend": backend,
                "threads": n_threads
            })

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

        workers.append(
            worker
        )

    if threads_blas:

        t = max(
            threads_blas
        )

    else:

        t = None

    if t is not None:

        potencial = (
            p * t
        )

        oversubscription = (
            potencial > P_MAX
        )

    else:

        potencial = None
        oversubscription = None

    datos = {
        "tipo": "inspect",
        "p": p,
        "t": t,
        "threads_openmp": (
            max(threads_openmp)
            if threads_openmp
            else None
        ),
        "p_t": potencial,
        "cores_logicos": P_MAX,
        "oversubscription": oversubscription,
        "workers": workers
    }

    guardar_temporal(
        datos
    )


# ============================================================
# PROCESO HIJO: EJECUCIÓN ORIGINAL
# ============================================================

def child_original(p):

    from time import perf_counter

    from common import (
        generate_data
    )

    from bs_numpy import (
        bootstrap_numpy
    )

    memoria_inicial = (
        memoria_disponible_mb()
    )

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

    memoria_final = (
        memoria_disponible_mb()
    )

    datos = {
        "tipo": "original",
        "p": p,
        "t": None,
        "p_t": None,
        "cores_logicos": P_MAX,
        "oversubscription": None,
        "tiempo": round(
            tiempo,
            4
        ),
        "estado": "completado",
        "memoria_inicial_mb": memoria_inicial,
        "memoria_final_mb": memoria_final
    }

    guardar_temporal(
        datos
    )


# ============================================================
# PROCESO HIJO: EJECUCIÓN LIMITADA A t=1
# ============================================================

def child_limited(p):

    # Deben fijarse ANTES de importar NumPy
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["BLIS_NUM_THREADS"] = "1"

    from time import perf_counter

    import numpy as np

    from threadpoolctl import (
        threadpool_limits,
        threadpool_info
    )

    from common import generate_data
    from bs_numpy import bootstrap_numpy

    memoria_inicial = memoria_disponible_mb()

    X, y, _ = generate_data()

    with threadpool_limits(limits=1):

        # Inicializar BLAS/LAPACK
        A = np.random.default_rng(123).normal(
            size=(200, 200)
        )
        b = np.random.default_rng(456).normal(
            size=200
        )

        _ = np.linalg.solve(A, b)

        # Verificar realmente los threads
        info = threadpool_info()

        threads_blas = [
            pool["num_threads"]
            for pool in info
            if pool.get("user_api") == "blas"
        ]

        t_medido = (
            max(threads_blas)
            if threads_blas
            else None
        )

        inicio = perf_counter()

        bootstrap_numpy(
            X,
            y,
            p
        )

        tiempo = perf_counter() - inicio

    memoria_final = memoria_disponible_mb()

    p_t = (
        p * t_medido
        if t_medido is not None
        else None
    )

    datos = {
        "tipo": "limited",
        "p": p,
        "t": t_medido,
        "p_t": p_t,
        "cores_logicos": P_MAX,
        "oversubscription": (
            p_t > P_MAX
            if p_t is not None
            else None
        ),
        "tiempo": round(tiempo, 4),
        "estado": "completado",
        "memoria_inicial_mb": memoria_inicial,
        "memoria_final_mb": memoria_final,
        "threadpool_info": info
    }

    guardar_temporal(datos)


# ============================================================
# EJECUTAR UN PROCESO HIJO CON TIMEOUT
# ============================================================

def ejecutar_hijo(modo, p, timeout=None):

    TEMP_PATH.unlink(
        missing_ok=True
    )

    comando = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        modo,
        str(p)
    ]

    proceso = subprocess.Popen(
        comando,
        start_new_session=True
    )

    try:

        proceso.wait(
            timeout=timeout
        )

    except subprocess.TimeoutExpired:

        # En WSL/Linux se mata todo el grupo,
        # incluyendo workers de joblib.
        try:

            os.killpg(
                os.getpgid(
                    proceso.pid
                ),
                signal.SIGTERM
            )

        except Exception:

            proceso.kill()

        try:

            proceso.wait(
                timeout=5
            )

        except Exception:

            pass

        return {
            "tipo": modo,
            "p": p,
            "t": (
                1
                if modo == "limited"
                else None
            ),
            "p_t": (
                p
                if modo == "limited"
                else None
            ),
            "cores_logicos": P_MAX,
            "oversubscription": None,
            "tiempo": None,
            "estado": (
                f"detenido después de "
                f"{timeout} s"
            ),
            "memoria_inicial_mb": None,
            "memoria_final_mb": None
        }

    datos = leer_temporal()

    if datos is None:

        return {
            "tipo": modo,
            "p": p,
            "t": None,
            "p_t": None,
            "cores_logicos": P_MAX,
            "oversubscription": None,
            "tiempo": None,
            "estado": "error",
            "memoria_inicial_mb": None,
            "memoria_final_mb": None
        }

    return datos


# ============================================================
# GUARDAR CSV
# ============================================================

def guardar_csv(resultados):

    columnas = [
        "tipo",
        "p",
        "t",
        "p_t",
        "cores_logicos",
        "oversubscription",
        "tiempo_s",
        "estado",
        "memoria_inicial_mb",
        "memoria_final_mb"
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as archivo:

        writer = csv.DictWriter(
            archivo,
            fieldnames=columnas
        )

        writer.writeheader()

        for resultado in resultados:

            writer.writerow({
                "tipo": resultado.get(
                    "tipo"
                ),
                "p": resultado.get(
                    "p"
                ),
                "t": resultado.get(
                    "t"
                ),
                "p_t": resultado.get(
                    "p_t"
                ),
                "cores_logicos": resultado.get(
                    "cores_logicos"
                ),
                "oversubscription": resultado.get(
                    "oversubscription"
                ),
                "tiempo_s": resultado.get(
                    "tiempo"
                ),
                "estado": resultado.get(
                    "estado",
                    ""
                ),
                "memoria_inicial_mb": resultado.get(
                    "memoria_inicial_mb"
                ),
                "memoria_final_mb": resultado.get(
                    "memoria_final_mb"
                )
            })


# ============================================================
# ESCRIBIR RESULTADO DE INSPECCIÓN EN TXT
# ============================================================

def escribir_inspeccion(resultado):

    p = resultado["p"]

    escribir_txt()
    escribir_txt(
        "=" * 70
    )
    escribir_txt(
        f"INSPECCIÓN DE THREADS - p = {p}"
    )
    escribir_txt(
        "=" * 70
    )

    escribir_txt(
        f"Cores lógicos disponibles: "
        f"{resultado['cores_logicos']}"
    )

    for worker in resultado[
        "workers"
    ]:

        escribir_txt()

        escribir_txt(
            f"Worker "
            f"{worker['worker']} "
            f"| PID "
            f"{worker['pid']}"
        )

        for pool in worker[
            "pools"
        ]:

            escribir_txt(
                f"  API="
                f"{pool['api']} "
                f"| backend="
                f"{pool['backend']} "
                f"| threads="
                f"{pool['threads']}"
            )

    escribir_txt()
    escribir_txt("RESUMEN")

    escribir_txt(
        f"Threads internos considerados "
        f"por proceso: "
        f"t = {resultado['t']}"
    )

    escribir_txt(
        f"Paralelismo potencial: "
        f"p*t = "
        f"{resultado['p']}*"
        f"{resultado['t']} "
        f"= {resultado['p_t']}"
    )

    escribir_txt(
        f"Cores lógicos: "
        f"{resultado['cores_logicos']}"
    )

    if resultado[
        "oversubscription"
    ]:

        texto = "SÍ"

    else:

        texto = "NO"

    escribir_txt(
        f"Indicio potencial de "
        f"oversubscription: {texto}"
    )


# ============================================================
# ESCRIBIR EJECUCIÓN EN TXT
# ============================================================

def escribir_ejecucion(resultado):

    modo = resultado[
        "tipo"
    ]

    p = resultado[
        "p"
    ]

    escribir_txt()
    escribir_txt(
        "=" * 70
    )

    if modo == "original":

        escribir_txt(
            f"EJECUCIÓN ORIGINAL - p = {p}"
        )

        escribir_txt(
            "=" * 70
        )

        escribir_txt(
            "Sin límite manual de "
            "threads internos."
        )

    else:

        escribir_txt(
            f"EJECUCIÓN LIMITADA "
            f"t=1 - p = {p}"
        )

        escribir_txt(
            "=" * 70
        )

        escribir_txt(
            "Threads internos "
            "limitados a t=1."
        )

    escribir_txt(
        f"Estado: "
        f"{resultado.get('estado')}"
    )

    tiempo = resultado.get(
        "tiempo"
    )

    if tiempo is not None:

        escribir_txt(
            f"Tiempo: "
            f"{tiempo:.2f} s"
        )

    memoria_inicial = resultado.get(
        "memoria_inicial_mb"
    )

    memoria_final = resultado.get(
        "memoria_final_mb"
    )

    if memoria_inicial is not None:

        escribir_txt(
            f"Memoria disponible antes: "
            f"{memoria_inicial:.2f} MB"
        )

    if memoria_final is not None:

        escribir_txt(
            f"Memoria disponible después: "
            f"{memoria_final:.2f} MB"
        )

    if modo == "limited":

        escribir_txt(
            f"p*t = {p}*1 = {p}"
        )

        escribir_txt(
            f"Cores lógicos: {P_MAX}"
        )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    RESULTADOS_DIR.mkdir(
        exist_ok=True
    )

    # Reiniciar archivos anteriores.
    TXT_PATH.write_text(
        "",
        encoding="utf-8"
    )

    TEMP_PATH.unlink(
        missing_ok=True
    )

    escribir_txt(
        "ÍTEM (e) - OVERSUBSCRIPTION EN bs_numpy.py"
    )

    escribir_txt(
        "=" * 70
    )

    escribir_txt(
        f"Cores lógicos disponibles: {P_MAX}"
    )

    escribir_txt(
        f"Valores de p inspeccionados: "
        f"{P_INSPECCION}"
    )

    escribir_txt(
        f"Valores de p ejecutados: "
        f"{P_EJECUCION}"
    )

    escribir_txt(
        f"Timeout por ejecución: "
        f"{TIMEOUT_SEGUNDOS} s"
    )

    escribir_txt()
    escribir_txt(
        "Durante las ejecuciones originales "
        "y limitadas se debe observar htop "
        "en una segunda terminal."
    )

    todos_resultados = []

    # --------------------------------------------------------
    # 1. INSPECCIÓN
    # --------------------------------------------------------

    escribir_txt()
    escribir_txt(
        "#" * 70
    )
    escribir_txt(
        "1. INSPECCIÓN DE THREADS"
    )
    escribir_txt(
        "#" * 70
    )

    for p in P_INSPECCION:

        resultado = ejecutar_hijo(
            "inspect",
            p
        )

        todos_resultados.append(
            resultado
        )

        escribir_inspeccion(
            resultado
        )

    # --------------------------------------------------------
    # 2. EJECUCIÓN ORIGINAL
    # --------------------------------------------------------

    escribir_txt()
    escribir_txt(
        "#" * 70
    )
    escribir_txt(
        "2. EJECUCIÓN ORIGINAL"
    )
    escribir_txt(
        "#" * 70
    )

    for p in P_EJECUCION:

        escribir_txt()
        escribir_txt(
            f"Iniciando ejecución original "
            f"con p={p}. "
            f"Observar htop ahora."
        )

        resultado = ejecutar_hijo(
            "original",
            p,
            timeout=TIMEOUT_SEGUNDOS
        )

        todos_resultados.append(
            resultado
        )

        escribir_ejecucion(
            resultado
        )

    # --------------------------------------------------------
    # 3. EJECUCIÓN CORREGIDA
    # --------------------------------------------------------

    escribir_txt()
    escribir_txt(
        "#" * 70
    )
    escribir_txt(
        "3. EJECUCIÓN LIMITADA A t=1"
    )
    escribir_txt(
        "#" * 70
    )

    for p in P_EJECUCION:

        escribir_txt()
        escribir_txt(
            f"Iniciando ejecución limitada "
            f"con p={p}. "
            f"Observar htop ahora."
        )

        resultado = ejecutar_hijo(
            "limited",
            p,
            timeout=TIMEOUT_SEGUNDOS
        )

        todos_resultados.append(
            resultado
        )

        escribir_ejecucion(
            resultado
        )

    # --------------------------------------------------------
    # 4. CONCLUSIÓN AUTOMÁTICA
    # --------------------------------------------------------

    escribir_txt()
    escribir_txt(
        "#" * 70
    )
    escribir_txt(
        "4. RESUMEN"
    )
    escribir_txt(
        "#" * 70
    )

    inspecciones = [
        r for r in todos_resultados
        if r.get("tipo") == "inspect"
    ]

    casos_over = [
        r["p"]
        for r in inspecciones
        if r.get(
            "oversubscription"
        )
    ]

    casos_no_over = [
        r["p"]
        for r in inspecciones
        if r.get(
            "oversubscription"
        ) is False
    ]

    escribir_txt(
        f"Sin indicio potencial de "
        f"oversubscription: "
        f"p = {casos_no_over}"
    )

    escribir_txt(
        f"Con indicio potencial de "
        f"oversubscription: "
        f"p = {casos_over}"
    )

    escribir_txt(
        "Corrección utilizada: "
        "threadpool_limits(limits=1)."
    )

    escribir_txt(
        "Con t=1, el paralelismo potencial "
        "queda dado por p*t=p."
    )

    # --------------------------------------------------------
    # 5. CSV
    # --------------------------------------------------------

    guardar_csv(
        todos_resultados
    )

    escribir_txt()
    escribir_txt(
        "=" * 70
    )

    escribir_txt(
        "EXPERIMENTO FINALIZADO"
    )

    escribir_txt(
        "=" * 70
    )

    escribir_txt(
        f"TXT guardado en: "
        f"{TXT_PATH}"
    )

    escribir_txt(
        f"CSV guardado en: "
        f"{CSV_PATH}"
    )


# ============================================================
# ENTRADA DEL PROGRAMA
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Modo interno usado por el proceso principal.
    # No se ejecuta manualmente.
    # --------------------------------------------------------

    if (
        len(sys.argv) == 4
        and sys.argv[1] == "--child"
    ):

        modo = sys.argv[2]

        p = int(
            sys.argv[3]
        )

        if modo == "inspect":

            child_inspect(
                p
            )

        elif modo == "original":

            child_original(
                p
            )

        elif modo == "limited":

            child_limited(
                p
            )

        else:

            raise ValueError(
                f"Modo desconocido: {modo}"
            )

    # --------------------------------------------------------
    # Ejecución normal.
    # --------------------------------------------------------

    else:

        main()
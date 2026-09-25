import csv
import os
import sys
from contextlib import redirect_stdout
from time import perf_counter

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_info, threadpool_limits

from common import (
    B,
    generate_data,
    generate_bootstrap_seeds,
)

from bs_numpy import (
    fit_numpy,
    bootstrap_numpy,
)


class Tee:
    """
    Envía la salida simultáneamente a la terminal
    y a un archivo de texto.
    """

    def __init__(self, *outputs):
        self.outputs = outputs

    def write(self, text):
        for output in self.outputs:
            output.write(text)
            output.flush()

    def flush(self):
        for output in self.outputs:
            output.flush()


def get_p_values(p_max):
    """
    Valores de p usados para el ítem (e).

    No se hace todavía el barrido completo
    p = 1, ..., p_max, porque eso corresponde al ítem (f).
    """

    candidates = [
        1,
        2,
        4,
        min(8, p_max)
    ]

    return sorted(
        {
            p
            for p in candidates
            if 1 <= p <= p_max
        }
    )


def get_threadpool_summary():
    """
    Obtiene la información de los threadpools
    usados internamente por NumPy/BLAS.
    """

    # Operación pequeña para asegurar
    # que BLAS esté inicializado
    A = np.ones((64, 64))
    _ = A @ A

    pools = []

    for info in threadpool_info():

        pools.append(
            {
                "user_api": info.get("user_api"),
                "internal_api": info.get("internal_api"),
                "prefix": info.get("prefix"),
                "num_threads": info.get("num_threads"),
                "filepath": info.get("filepath"),
            }
        )

    return pools


def inspect_worker(task_id, thread_limit=None):
    """
    Inspecciona los threadpools dentro
    de un worker de joblib.

    Si thread_limit no es None,
    aplica ese límite antes de medir.
    """

    if thread_limit is None:

        pools = get_threadpool_summary()

    else:

        with threadpool_limits(
            limits=thread_limit
        ):
            pools = get_threadpool_summary()

    return {
        "task": task_id,
        "pid": os.getpid(),
        "thread_limit": thread_limit,
        "pools": pools,
    }


def inspect_threadpools(
    p,
    thread_limit=None
):
    """
    Lanza p workers y observa los threads
    internos disponibles en cada uno.
    """

    return Parallel(
        n_jobs=p,
        backend="multiprocessing"
    )(
        delayed(inspect_worker)(
            task_id,
            thread_limit
        )
        for task_id in range(p)
    )


def max_threads_observed(
    worker_results
):
    """
    Devuelve el mayor num_threads reportado
    por los threadpools observados.
    """

    values = []

    for worker in worker_results:

        for pool in worker["pools"]:

            num_threads = (
                pool["num_threads"]
            )

            if num_threads is not None:

                values.append(
                    int(num_threads)
                )

    if not values:
        return 1

    return max(values)


def bootstrap_fit_limited(
    X,
    y,
    seed,
    t
):
    """
    Ejecuta un resample bootstrap igual
    que bs_numpy.py, pero limita a t
    los threads internos de NumPy/BLAS.
    """

    rng = np.random.default_rng(seed)

    n = X.shape[0]

    # Sortear N índices con reemplazo
    indices = rng.integers(
        low=0,
        high=n,
        size=n
    )

    # Construir X(b) e y(b)
    Xb = X[indices]
    yb = y[indices]

    # Limitar los threads internos
    with threadpool_limits(
        limits=t
    ):

        beta_hat = fit_numpy(
            Xb,
            yb
        )

    return beta_hat


def bootstrap_numpy_limited(
    X,
    y,
    p,
    t
):
    """
    Versión de bs_numpy.py con:

    p procesos de joblib
    t threads internos máximos por proceso
    """

    seeds = generate_bootstrap_seeds()

    betas = Parallel(
        n_jobs=p,
        backend="multiprocessing"
    )(
        delayed(
            bootstrap_fit_limited
        )(
            X,
            y,
            seeds[b],
            t
        )
        for b in range(B)
    )

    return np.array(betas)


def print_worker_info(
    worker_results
):
    """
    Imprime de forma compacta
    la información de los workers.
    """

    seen = set()

    for worker in worker_results:

        pid = worker["pid"]

        if pid in seen:
            continue

        seen.add(pid)

        print(
            f"  PID {pid}:"
        )

        if not worker["pools"]:

            print(
                "    No se detectaron "
                "threadpools BLAS/OpenMP."
            )

            continue

        for pool in worker["pools"]:

            print(
                "    "
                f"API={pool['user_api']} | "
                f"backend={pool['internal_api']} | "
                f"threads={pool['num_threads']}"
            )


def run_analysis():

    print("=" * 72)
    print(
        "ÍTEM (e): "
        "OVERSUBSCRIPTION EN bs_numpy.py"
    )
    print("=" * 72)

    # Número de cores lógicos
    p_max = os.cpu_count() or 1

    p_values = get_p_values(
        p_max
    )

    print(
        f"\nCores lógicos detectados: "
        f"{p_max}"
    )

    print(
        f"Valores de p evaluados: "
        f"{p_values}"
    )

    # --------------------------------------------------
    # Generar dataset
    # --------------------------------------------------

    print(
        "\nGenerando dataset..."
    )

    X, y, _ = generate_data()

    print(
        "Dataset generado."
    )

    print(
        "X:",
        X.shape
    )

    print(
        "y:",
        y.shape
    )

    rows = []

    # --------------------------------------------------
    # Probar distintos valores de p
    # --------------------------------------------------

    for p in p_values:

        print(
            "\n"
            + "-" * 72
        )

        print(
            f"CASO p = {p}"
        )

        print(
            "-" * 72
        )

        # ==================================================
        # 1. CONFIGURACIÓN ORIGINAL
        # ==================================================

        print(
            "\nThreadpools observados "
            "SIN limitar threads:"
        )

        info_default = (
            inspect_threadpools(
                p=p,
                thread_limit=None
            )
        )

        print_worker_info(
            info_default
        )

        t_default = (
            max_threads_observed(
                info_default
            )
        )

        potential_default = (
            p * t_default
        )

        print(
            "\nMáximo de threads internos "
            f"observado por worker: "
            f"{t_default}"
        )

        print(
            "Paralelismo potencial p*t: "
            f"{p}*{t_default} "
            f"= {potential_default}"
        )

        print(
            "Cores lógicos disponibles: "
            f"{p_max}"
        )

        oversubscription = (
            potential_default
            > p_max
        )

        print(
            "Indicio de oversubscription "
            "según p*t > cores lógicos:",
            (
                "Sí"
                if oversubscription
                else "No"
            )
        )

        # ==================================================
        # Ejecutar bs_numpy original
        # ==================================================

        print(
            "\nEjecutando bs_numpy.py "
            "sin límite interno..."
        )

        start = perf_counter()

        betas_default = bootstrap_numpy(
            X,
            y,
            p
        )

        time_default = (
            perf_counter()
            - start
        )

        print(
            "Tiempo sin limitar threads: "
            f"{time_default:.2f} s"
        )

        # ==================================================
        # 2. CORRECCIÓN
        # ==================================================

        t_limited = 1

        print(
            "\nAplicando corrección: "
            f"threadpool_limits("
            f"limits={t_limited})"
        )

        info_limited = (
            inspect_threadpools(
                p=p,
                thread_limit=t_limited
            )
        )

        print(
            "Threadpools observados "
            "CON límite:"
        )

        print_worker_info(
            info_limited
        )

        observed_limited = (
            max_threads_observed(
                info_limited
            )
        )

        potential_limited = (
            p * observed_limited
        )

        print(
            "Paralelismo potencial "
            "corregido p*t: "
            f"{p}*{observed_limited} "
            f"= {potential_limited}"
        )

        # ==================================================
        # Ejecutar versión corregida
        # ==================================================

        print(
            "\nEjecutando bootstrap "
            "con threads internos "
            "limitados..."
        )

        start = perf_counter()

        betas_limited = (
            bootstrap_numpy_limited(
                X,
                y,
                p,
                t_limited
            )
        )

        time_limited = (
            perf_counter()
            - start
        )

        print(
            f"Tiempo con t = "
            f"{t_limited}: "
            f"{time_limited:.2f} s"
        )

        # ==================================================
        # Verificar que limitar threads
        # no cambie los resultados
        # ==================================================

        max_difference = np.max(
            np.abs(
                betas_default
                - betas_limited
            )
        )

        print(
            "Diferencia máxima entre "
            "coeficientes sin/con límite: "
            f"{max_difference:.3e}"
        )

        # ==================================================
        # Comparación temporal
        # ==================================================

        if time_limited > 0:

            improvement = (
                time_default
                / time_limited
            )

        else:

            improvement = np.nan

        print(
            "Razón "
            "tiempo_original / "
            "tiempo_corregido: "
            f"{improvement:.3f}"
        )

        # Guardar resultados
        rows.append(
            {
                "p": p,

                "cores_logicos":
                    p_max,

                "t_original":
                    t_default,

                "p_x_t_original":
                    potential_default,

                "oversubscription":
                    oversubscription,

                "tiempo_original_s":
                    time_default,

                "t_limitado":
                    observed_limited,

                "p_x_t_limitado":
                    potential_limited,

                "tiempo_limitado_s":
                    time_limited,

                "razon_original_limitado":
                    improvement,

                "max_diff_betas":
                    max_difference,
            }
        )

    # --------------------------------------------------
    # Resumen
    # --------------------------------------------------

    print(
        "\n"
        + "=" * 72
    )

    print(
        "RESUMEN"
    )

    print(
        "=" * 72
    )

    print(
        "\np | "
        "t original | "
        "p*t original | "
        "oversub. | "
        "T original [s] | "
        "t limitado | "
        "T limitado [s]"
    )

    for row in rows:

        print(
            f"{row['p']:>2} | "
            f"{row['t_original']:>10} | "
            f"{row['p_x_t_original']:>12} | "
            f"{'Sí' if row['oversubscription'] else 'No':>8} | "
            f"{row['tiempo_original_s']:>14.2f} | "
            f"{row['t_limitado']:>10} | "
            f"{row['tiempo_limitado_s']:>13.2f}"
        )

    return rows


def main():

    os.makedirs(
        "resultados",
        exist_ok=True
    )

    txt_file = (
        "resultados/resultados_e.txt"
    )

    csv_file = (
        "resultados/resultados_e.csv"
    )

    # --------------------------------------------------
    # Guardar salida completa en TXT
    # --------------------------------------------------

    with open(
        txt_file,
        "w",
        encoding="utf-8"
    ) as file:

        tee = Tee(
            sys.stdout,
            file
        )

        with redirect_stdout(tee):

            rows = run_analysis()

    # --------------------------------------------------
    # Guardar tabla resumen en CSV
    # --------------------------------------------------

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "p",
            "cores_logicos",
            "t_original",
            "p_x_t_original",
            "oversubscription",
            "tiempo_original_s",
            "t_limitado",
            "p_x_t_limitado",
            "tiempo_limitado_s",
            "razon_original_limitado",
            "max_diff_betas",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print(
        "\nResultados guardados en:"
    )

    print(
        txt_file
    )

    print(
        csv_file
    )


if __name__ == "__main__":
    main()
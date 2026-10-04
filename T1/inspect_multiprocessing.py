import os
import sys
import time
import multiprocessing as mp
import numpy as np

from contextlib import redirect_stdout
from joblib import Parallel, delayed

from T1.common import generate_data


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


def get_rss_mb():
    """
    Lee la memoria residente (RSS) del proceso actual
    desde /proc/self/status en Linux.
    """

    with open(
        "/proc/self/status",
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if line.startswith("VmRSS:"):

                rss_kb = int(
                    line.split()[1]
                )

                return rss_kb / 1024

    return None


def inspect_worker(X, y, task_id):
    """
    Obtiene información del proceso worker
    y de cómo recibe X e y.
    """

    # Mantener los workers activos brevemente
    # para facilitar la observación de varios procesos.
    time.sleep(1)

    return {
        "task": task_id,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "start_method": mp.get_start_method(),
        "X_type": type(X).__name__,
        "y_type": type(y).__name__,
        "X_memmap": isinstance(X, np.memmap),
        "y_memmap": isinstance(y, np.memmap),
        "X_nbytes_mb": X.nbytes / (1024 ** 2),
        "y_nbytes_mb": y.nbytes / (1024 ** 2),
        "rss_mb": get_rss_mb(),
    }


def run_analysis():

    print("=" * 60)
    print("ANÁLISIS MULTIPROCESSING Y MEMORIA")
    print("=" * 60)

    print("\nGenerando dataset...")

    X, y, beta_true = generate_data()

    print("\nProceso principal")
    print("-" * 40)

    print(
        "PID:",
        os.getpid()
    )

    print(
        "Método de inicio:",
        mp.get_start_method()
    )

    print(
        f"Tamaño X: {X.nbytes / (1024 ** 2):.2f} MiB"
    )

    print(
        f"Tamaño y: {y.nbytes / (1024 ** 2):.2f} MiB"
    )

    print(
        f"RSS proceso principal: {get_rss_mb():.2f} MiB"
    )

    # Número de procesos para el experimento
    p = 4

    print()
    print(
        f"Lanzando joblib con p = {p} procesos..."
    )

    results = Parallel(
        n_jobs=p,
        backend="multiprocessing"
    )(
        delayed(inspect_worker)(
            X,
            y,
            task_id
        )
        for task_id in range(p)
    )

    print("\nInformación de los workers")
    print("=" * 60)

    for result in results:

        print(
            f"\nWorker asociado a tarea {result['task']}"
        )

        print(
            "PID:",
            result["pid"]
        )

        print(
            "PPID:",
            result["ppid"]
        )

        print(
            "Método de inicio:",
            result["start_method"]
        )

        print(
            "Tipo de X:",
            result["X_type"]
        )

        print(
            "Tipo de y:",
            result["y_type"]
        )

        print(
            "X es memmap:",
            result["X_memmap"]
        )

        print(
            "y es memmap:",
            result["y_memmap"]
        )

        print(
            f"Tamaño lógico X: "
            f"{result['X_nbytes_mb']:.2f} MiB"
        )

        print(
            f"Tamaño lógico y: "
            f"{result['y_nbytes_mb']:.2f} MiB"
        )

        print(
            f"RSS del worker: "
            f"{result['rss_mb']:.2f} MiB"
        )

    unique_pids = sorted(
        {
            result["pid"]
            for result in results
        }
    )

    print("\n")
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)

    print(
        "Cantidad de procesos solicitados:",
        p
    )

    print(
        "PIDs de workers observados:",
        unique_pids
    )

    print(
        "Cantidad de workers distintos:",
        len(unique_pids)
    )


def main():

    os.makedirs(
        "resultados",
        exist_ok=True
    )

    output_file = (
        "resultados/resultados_d.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        tee = Tee(
            sys.stdout,
            file
        )

        with redirect_stdout(tee):
            run_analysis()

    print(
        f"\nResultados guardados en: {output_file}"
    )


if __name__ == "__main__":
    main()
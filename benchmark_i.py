import argparse
import csv
import gc
import os
import platform

from time import perf_counter, sleep

from threadpoolctl import threadpool_limits

from common import (
    B,
    K,
    N,
    generate_data,
)
from bs_numpy import bootstrap_numpy


PAUSA_ENTRE_PRUEBAS = 1.0
RESULTADOS_DIR = "resultados"
CSV_PATH = os.path.join(
    RESULTADOS_DIR,
    "resultados_pt.csv"
)
CHECKPOINT_PATH = os.path.join(
    RESULTADOS_DIR,
    "resultados_pt_checkpoint.csv"
)
TXT_PATH = os.path.join(
    RESULTADOS_DIR,
    "resultados_pt.txt"
)


def configuraciones(p_max):

    return [
        {
            "p": p,
            "t": t,
            "tiempo": None,
        }
        for p in range(1, p_max + 1)
        for t in range(1, p_max // p + 1)
    ]


def get_cpu_model():

    try:

        with open(
            "/proc/cpuinfo",
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                if "model name" in line:

                    return line.split(
                        ":",
                        1
                    )[1].strip()

    except OSError:
        pass

    return platform.processor() or "No identificado"


def guardar_checkpoint(resultados):

    with open(
        CHECKPOINT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["p", "t", "p_por_t", "tiempo"]
        )

        writer.writeheader()

        for row in resultados:

            writer.writerow(
                {
                    "p": row["p"],
                    "t": row["t"],
                    "p_por_t": row["p"] * row["t"],
                    "tiempo": ""
                    if row["tiempo"] is None
                    else row["tiempo"],
                }
            )


def cargar_checkpoint(resultados, reanudar):

    if not reanudar or not os.path.exists(CHECKPOINT_PATH):
        return resultados

    with open(
        CHECKPOINT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        filas = csv.DictReader(file)
        valores = {
            (int(fila["p"]), int(fila["t"])): fila["tiempo"]
            for fila in filas
            if fila["tiempo"] != ""
        }

    for row in resultados:

        valor = valores.get(
            (row["p"], row["t"])
        )

        if valor is not None:
            row["tiempo"] = float(valor)

    return resultados


def formato_tiempo(valor):

    if valor is None:
        return "pendiente"

    return f"{valor:.2f}"


def guardar_txt(resultados, p_max, cpu_model):

    with open(
        TXT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("=" * 74 + "\n")
        file.write(
            "EXPERIMENTO NumPy: PROCESOS Y THREADS INTERNOS\n"
        )
        file.write("=" * 74 + "\n\n")

        file.write("INFORMACIÓN DEL COMPUTADOR\n")
        file.write("-" * 74 + "\n")
        file.write(f"Procesador: {cpu_model}\n")
        file.write(
            f"Sistema: {platform.system()} "
            f"{platform.release()}\n"
        )
        file.write(f"p_max: {p_max}\n\n")

        file.write("PARÁMETROS DEL EXPERIMENTO\n")
        file.write("-" * 74 + "\n")
        file.write(f"N = {N:,} observaciones\n")
        file.write(f"k = {K} variables de entrada\n")
        file.write(f"B = {B} resamples bootstrap\n")
        file.write("Se evaluaron todas las parejas con p * t <= p_max.\n\n")

        file.write("TIEMPOS DE EJECUCIÓN\n")
        file.write("-" * 74 + "\n")
        file.write(
            f"{'p':>4} | {'t':>4} | {'p*t':>6} | "
            f"{'NumPy [s]':>16}\n"
        )
        file.write("-" * 74 + "\n")

        for row in resultados:

            file.write(
                f"{row['p']:>4} | "
                f"{row['t']:>4} | "
                f"{row['p'] * row['t']:>6} | "
                f"{formato_tiempo(row['tiempo']):>16}\n"
            )


def guardar_csv(resultados):

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["p", "t", "p_por_t", "tiempo"]
        )

        writer.writeheader()

        for row in resultados:

            writer.writerow(
                {
                    "p": row["p"],
                    "t": row["t"],
                    "p_por_t": row["p"] * row["t"],
                    "tiempo": row["tiempo"],
                }
            )


def medir_tiempo(X, y, p, t):

    gc.collect()
    sleep(PAUSA_ENTRE_PRUEBAS)

    inicio = perf_counter()

    with threadpool_limits(limits=t):

        betas = bootstrap_numpy(
            X,
            y,
            p
        )

    tiempo = perf_counter() - inicio

    del betas
    gc.collect()

    return tiempo


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Mide NumPy variando simultáneamente p y t "
            "con p*t <= p_max."
        )
    )
    parser.add_argument(
        "--p-max",
        type=int,
        default=8,
        help="Límite del producto p*t (por defecto: 8)."
    )
    parser.add_argument(
        "--no-reanudar",
        action="store_true",
        help="Ignora el checkpoint existente."
    )
    args = parser.parse_args()

    if args.p_max < 1:
        parser.error("p_max debe ser al menos 1")

    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    resultados = configuraciones(args.p_max)
    resultados = cargar_checkpoint(
        resultados,
        not args.no_reanudar
    )
    guardar_txt(
        resultados,
        args.p_max,
        get_cpu_model()
    )

    print(f"p_max = {args.p_max}")
    print(f"Configuraciones: {len(resultados)}")
    print("Generando dataset...")
    X, y, _ = generate_data()

    try:

        for indice, row in enumerate(resultados, start=1):

            p = row["p"]
            t = row["t"]

            if row["tiempo"] is not None:

                print(
                    f"[{indice}/{len(resultados)}] "
                    f"p={p}, t={t}: ya medido "
                    f"({row['tiempo']:.2f} s)"
                )
                continue

            print(
                f"[{indice}/{len(resultados)}] "
                f"Ejecutando p={p}, t={t}, p*t={p * t}"
            )

            row["tiempo"] = medir_tiempo(
                X,
                y,
                p,
                t
            )

            print(f"Tiempo = {row['tiempo']:.2f} s")

            guardar_checkpoint(resultados)
            guardar_txt(
                resultados,
                args.p_max,
                get_cpu_model()
            )

    except KeyboardInterrupt:

        print("\nExperimento interrumpido; checkpoint guardado.")

    finally:

        guardar_checkpoint(resultados)
        guardar_csv(resultados)
        guardar_txt(
            resultados,
            args.p_max,
            get_cpu_model()
        )

    print(f"Resultados guardados en: {CSV_PATH}")
    print(f"Resumen guardado en: {TXT_PATH}")


if __name__ == "__main__":
    main()
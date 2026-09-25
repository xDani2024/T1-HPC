import csv
import gc
import os
import platform
from datetime import datetime
from time import perf_counter, sleep

from common import (
    N,
    K,
    B,
    generate_data,
)

from bs_numpy import bootstrap_numpy
from bs_sklearn import bootstrap_sklearn
from bs_auto import bootstrap_auto


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Pequeña pausa entre pruebas para evitar encadenarlas
# inmediatamente.
PAUSA_ENTRE_PRUEBAS = 1.0

# Carpeta de resultados
RESULTADOS_DIR = "resultados"

TXT_PATH = os.path.join(
    RESULTADOS_DIR,
    "resultados_f.txt"
)

CSV_PATH = os.path.join(
    RESULTADOS_DIR,
    "resultados_f.csv"
)


# ============================================================
# INFORMACIÓN DEL COMPUTADOR
# ============================================================

def get_cpu_model():
    """
    Intenta obtener el modelo del procesador.

    Funciona bien en Linux/WSL leyendo /proc/cpuinfo.
    Si no está disponible, utiliza platform.processor().
    """

    try:

        with open(
            "/proc/cpuinfo",
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                if "model name" in line:

                    return (
                        line.split(":", 1)[1]
                        .strip()
                    )

    except OSError:
        pass

    cpu = platform.processor()

    if cpu:
        return cpu

    return "No identificado"


# ============================================================
# MEDICIÓN DE UNA IMPLEMENTACIÓN
# ============================================================

def medir_tiempo(
    nombre,
    funcion,
    X,
    y,
    p
):
    """
    Ejecuta una implementación bootstrap
    con p procesos y mide su tiempo total.
    """

    print(
        f"  {nombre:<18} "
        f"| p = {p:>2} "
        f"| ejecutando...",
        end="",
        flush=True
    )

    inicio = perf_counter()

    betas = funcion(
        X,
        y,
        p
    )

    tiempo = (
        perf_counter()
        - inicio
    )

    print(
        f"\r  {nombre:<18} "
        f"| p = {p:>2} "
        f"| {tiempo:>8.2f} s"
    )

    # Ya no necesitamos las estimaciones para este ítem.
    del betas

    gc.collect()

    return tiempo


# ============================================================
# GUARDAR TXT
# ============================================================

def guardar_txt(
    resultados,
    p_max,
    cpu_model
):

    with open(
        TXT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "=" * 72 + "\n"
        )

        file.write(
            "ÍTEM (f) - TIEMPOS DE EJECUCIÓN "
            "SEGÚN NÚMERO DE PROCESOS\n"
        )

        file.write(
            "=" * 72 + "\n\n"
        )

        # ----------------------------------------------------
        # Información del computador
        # ----------------------------------------------------

        file.write(
            "INFORMACIÓN DEL COMPUTADOR\n"
        )

        file.write(
            "-" * 72 + "\n"
        )

        file.write(
            f"Procesador: {cpu_model}\n"
        )

        file.write(
            f"Sistema: "
            f"{platform.system()} "
            f"{platform.release()}\n"
        )

        file.write(
            f"Número de cores lógicos: "
            f"{p_max}\n"
        )

        file.write("\n")

        # ----------------------------------------------------
        # Parámetros
        # ----------------------------------------------------

        file.write(
            "PARÁMETROS DEL EXPERIMENTO\n"
        )

        file.write(
            "-" * 72 + "\n"
        )

        file.write(
            f"N = {N:,} observaciones\n"
        )

        file.write(
            f"k = {K} variables de entrada\n"
        )

        file.write(
            f"B = {B} resamples bootstrap\n"
        )

        file.write(
            f"Valores de p evaluados: "
            f"1 a {p_max}\n"
        )

        file.write("\n")

        # ----------------------------------------------------
        # Tabla principal
        # ----------------------------------------------------

        file.write(
            "TIEMPOS DE EJECUCIÓN\n"
        )

        file.write(
            "-" * 72 + "\n"
        )

        file.write(
            f"{'p':>4} | "
            f"{'NumPy [s]':>14} | "
            f"{'sklearn [s]':>14} | "
            f"{'Bagging [s]':>14}\n"
        )

        file.write(
            "-" * 72 + "\n"
        )

        for row in resultados:

            file.write(
                f"{row['p']:>4} | "
                f"{row['numpy']:>14.2f} | "
                f"{row['sklearn']:>14.2f} | "
                f"{row['auto']:>14.2f}\n"
            )

        file.write(
            "-" * 72 + "\n\n"
        )

        # ----------------------------------------------------
        # Formato simple p -> t
        # ----------------------------------------------------

        file.write(
            "DETALLE POR IMPLEMENTACIÓN\n"
        )

        file.write(
            "=" * 72 + "\n\n"
        )

        nombres = [
            ("NumPy", "numpy"),
            ("sklearn", "sklearn"),
            (
                "BaggingRegressor",
                "auto"
            ),
        ]

        for nombre, clave in nombres:

            file.write(
                f"{nombre}\n"
            )

            file.write(
                "-" * len(nombre)
                + "\n"
            )

            for row in resultados:

                file.write(
                    f"p = {row['p']:>2} "
                    f"-> "
                    f"t = "
                    f"{row[clave]:.2f} s\n"
                )

            file.write("\n")

        # ----------------------------------------------------
        # Mejor tiempo observado
        # ----------------------------------------------------

        file.write(
            "MEJORES TIEMPOS OBSERVADOS\n"
        )

        file.write(
            "-" * 72 + "\n"
        )

        for nombre, clave in nombres:

            mejor = min(
                resultados,
                key=lambda row:
                    row[clave]
            )

            file.write(
                f"{nombre:<18}: "
                f"{mejor[clave]:.2f} s "
                f"con p = {mejor['p']}\n"
            )


# ============================================================
# GUARDAR CSV
# ============================================================

def guardar_csv(
    resultados
):

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "p",
                "numpy",
                "sklearn",
                "auto",
            ]
        )

        writer.writeheader()

        writer.writerows(
            resultados
        )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    os.makedirs(
        RESULTADOS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Información del computador
    # --------------------------------------------------------

    p_max = os.cpu_count()

    if p_max is None:
        p_max = 1

    cpu_model = get_cpu_model()

    print(
        "=" * 72
    )

    print(
        "ÍTEM (f): BENCHMARK SEGÚN "
        "NÚMERO DE PROCESOS"
    )

    print(
        "=" * 72
    )

    print(
        f"\nProcesador: "
        f"{cpu_model}"
    )

    print(
        f"Cores lógicos detectados: "
        f"{p_max}"
    )

    print(
        f"Se evaluará p = "
        f"1, 2, ..., {p_max}"
    )

    # --------------------------------------------------------
    # Generar los datos UNA sola vez
    # --------------------------------------------------------

    print(
        "\nGenerando dataset..."
    )

    X, y, beta_true = generate_data()

    print(
        f"X: {X.shape}"
    )

    print(
        f"y: {y.shape}"
    )

    print(
        "Dataset generado correctamente."
    )

    # beta_true no se necesita para medir tiempos.
    del beta_true

    # --------------------------------------------------------
    # Ejecutar experimentos
    # --------------------------------------------------------

    resultados = []

    print(
        "\n"
        + "=" * 72
    )

    print(
        "INICIO DE LAS MEDICIONES"
    )

    print(
        "=" * 72
    )

    total_pruebas = (
        p_max * 3
    )

    prueba_actual = 0

    for p in range(
        1,
        p_max + 1
    ):

        print(
            f"\n--- p = {p} / "
            f"{p_max} ---"
        )

        # NumPy
        prueba_actual += 1

        print(
            f"[{prueba_actual}/"
            f"{total_pruebas}]",
            end=" "
        )

        tiempo_numpy = medir_tiempo(
            "NumPy",
            bootstrap_numpy,
            X,
            y,
            p
        )

        sleep(
            PAUSA_ENTRE_PRUEBAS
        )

        # sklearn
        prueba_actual += 1

        print(
            f"[{prueba_actual}/"
            f"{total_pruebas}]",
            end=" "
        )

        tiempo_sklearn = medir_tiempo(
            "sklearn",
            bootstrap_sklearn,
            X,
            y,
            p
        )

        sleep(
            PAUSA_ENTRE_PRUEBAS
        )

        # BaggingRegressor
        prueba_actual += 1

        print(
            f"[{prueba_actual}/"
            f"{total_pruebas}]",
            end=" "
        )

        tiempo_auto = medir_tiempo(
            "BaggingRegressor",
            bootstrap_auto,
            X,
            y,
            p
        )

        sleep(
            PAUSA_ENTRE_PRUEBAS
        )

        resultados.append(
            {
                "p": p,
                "numpy":
                    tiempo_numpy,
                "sklearn":
                    tiempo_sklearn,
                "auto":
                    tiempo_auto,
            }
        )

    # --------------------------------------------------------
    # Guardar resultados
    # --------------------------------------------------------

    guardar_txt(
        resultados,
        p_max,
        cpu_model
    )

    guardar_csv(
        resultados
    )

    # --------------------------------------------------------
    # Resumen por consola
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 72
    )

    print(
        "RESULTADOS"
    )

    print(
        "=" * 72
    )

    print(
        f"\n{'p':>4} | "
        f"{'NumPy [s]':>12} | "
        f"{'sklearn [s]':>12} | "
        f"{'Bagging [s]':>12}"
    )

    print(
        "-" * 52
    )

    for row in resultados:

        print(
            f"{row['p']:>4} | "
            f"{row['numpy']:>12.2f} | "
            f"{row['sklearn']:>12.2f} | "
            f"{row['auto']:>12.2f}"
        )

    print(
        "\nArchivos guardados:"
    )

    print(
        f"  {TXT_PATH}"
    )

    print(
        f"  {CSV_PATH}"
    )

    print(
        "\nÍtem (f) terminado."
    )


if __name__ == "__main__":
    main()
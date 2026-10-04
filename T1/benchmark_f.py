# ============================================================
# IMPORTANTE:
# Estas variables deben definirse ANTES de importar NumPy,
# sklearn o los scripts que los utilizan.
# ============================================================

import os

THREAD_LIMIT = 1

os.environ["OMP_NUM_THREADS"] = str(THREAD_LIMIT)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREAD_LIMIT)
os.environ["MKL_NUM_THREADS"] = str(THREAD_LIMIT)
os.environ["NUMEXPR_NUM_THREADS"] = str(THREAD_LIMIT)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(THREAD_LIMIT)
os.environ["BLIS_NUM_THREADS"] = str(THREAD_LIMIT)


# ============================================================
# IMPORTS
# ============================================================

import csv
import gc
import platform

from time import perf_counter, sleep
from threadpoolctl import threadpool_limits

from T1.common import (
    N,
    K,
    B,
    generate_data,
)

from T1.bs_numpy import bootstrap_numpy
from T1.bs_sklearn import bootstrap_sklearn
from T1.bs_auto import bootstrap_auto


# ============================================================
# CONFIGURACIÓN
# ============================================================

PAUSA_ENTRE_PRUEBAS = 1.0

RESULTADOS_DIR = "resultados"

TXT_PATH = os.path.join(
    RESULTADOS_DIR,
    "resultados_f.txt"
)

CSV_PATH = os.path.join(
    RESULTADOS_DIR,
    "resultados_f.csv"
)

CHECKPOINT_PATH = os.path.join(
    RESULTADOS_DIR,
    "resultados_f_checkpoint.csv"
)


# Si el programa se interrumpe y luego lo vuelves a ejecutar,
# continuará desde donde quedó.
REANUDAR = True


# ============================================================
# INFORMACIÓN DEL COMPUTADOR
# ============================================================

def get_cpu_model():

    try:

        with open(
            "/proc/cpuinfo",
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                if "model name" in line:

                    return (
                        line
                        .split(":", 1)[1]
                        .strip()
                    )

    except OSError:
        pass

    cpu = platform.processor()

    if cpu:
        return cpu

    return "No identificado"


# ============================================================
# CREAR ESTRUCTURA DE RESULTADOS
# ============================================================

def crear_resultados_vacios(p_max):

    resultados = []

    for p in range(1, p_max + 1):

        resultados.append(
            {
                "p": p,
                "numpy": None,
                "sklearn": None,
                "auto": None,
            }
        )

    return resultados


# ============================================================
# CHECKPOINT
# ============================================================

def guardar_checkpoint(resultados):

    with open(
        CHECKPOINT_PATH,
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
                "thread_limit",
            ]
        )

        writer.writeheader()

        for row in resultados:

            writer.writerow(
                {
                    "p": row["p"],

                    "numpy":
                        ""
                        if row["numpy"] is None
                        else row["numpy"],

                    "sklearn":
                        ""
                        if row["sklearn"] is None
                        else row["sklearn"],

                    "auto":
                        ""
                        if row["auto"] is None
                        else row["auto"],

                    "thread_limit":
                        THREAD_LIMIT,
                }
            )


def cargar_checkpoint(resultados):

    if not REANUDAR:

        return resultados

    if not os.path.exists(
        CHECKPOINT_PATH
    ):

        return resultados

    print(
        "\nCheckpoint encontrado."
    )

    print(
        "Se intentará continuar desde "
        "la última medición terminada."
    )

    with open(
        CHECKPOINT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        filas = list(reader)

    # Verificar que corresponde al mismo
    # límite de threads.
    if filas:

        limite_guardado = int(
            filas[0]["thread_limit"]
        )

        if limite_guardado != THREAD_LIMIT:

            print(
                "\nEl checkpoint corresponde "
                "a otra configuración."
            )

            print(
                "Se ignorará y se comenzará "
                "desde cero."
            )

            return resultados

    mapa = {
        row["p"]: row
        for row in resultados
    }

    for fila in filas:

        p = int(fila["p"])

        if p not in mapa:
            continue

        for clave in [
            "numpy",
            "sklearn",
            "auto",
        ]:

            valor = fila[clave]

            if valor != "":

                mapa[p][clave] = float(
                    valor
                )

    return resultados


# ============================================================
# MEDIR UNA IMPLEMENTACIÓN
# ============================================================

def medir_tiempo(
    nombre,
    funcion,
    X,
    y,
    p
):

    print()

    print(
        f"Ejecutando {nombre}"
    )

    print(
        f"p = {p}"
    )

    print(
        f"Threads internos máximos = "
        f"{THREAD_LIMIT}"
    )

    gc.collect()

    sleep(
        PAUSA_ENTRE_PRUEBAS
    )

    inicio = perf_counter()

    # Segundo mecanismo de seguridad:
    # además de las variables de entorno,
    # limitamos los threadpools cargados.
    with threadpool_limits(
        limits=THREAD_LIMIT
    ):

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
        f"Tiempo = {tiempo:.2f} s"
    )

    # Ya no necesitamos las estimaciones
    # bootstrap para el ítem (f).
    del betas

    gc.collect()

    return tiempo


# ============================================================
# FORMATEAR TIEMPOS
# ============================================================

def formato_tiempo(valor):

    if valor is None:
        return "pendiente"

    return f"{valor:.2f}"


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
            "=" * 74
            + "\n"
        )

        file.write(
            "ÍTEM (f) - TIEMPOS DE EJECUCIÓN "
            "SEGÚN NÚMERO DE PROCESOS\n"
        )

        file.write(
            "=" * 74
            + "\n\n"
        )

        # ----------------------------------------------------
        # Computador
        # ----------------------------------------------------

        file.write(
            "INFORMACIÓN DEL COMPUTADOR\n"
        )

        file.write(
            "-" * 74
            + "\n"
        )

        file.write(
            f"Procesador: "
            f"{cpu_model}\n"
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

        file.write(
            f"Threads internos máximos "
            f"por proceso: "
            f"{THREAD_LIMIT}\n"
        )

        file.write("\n")

        # ----------------------------------------------------
        # Parámetros
        # ----------------------------------------------------

        file.write(
            "PARÁMETROS DEL EXPERIMENTO\n"
        )

        file.write(
            "-" * 74
            + "\n"
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
            f"1, 2, ..., {p_max}\n"
        )

        file.write(
            "Los threads internos de "
            "NumPy/BLAS se fijaron en "
            "t = 1 para aislar el efecto "
            "del número de procesos p.\n"
        )

        file.write("\n")

        # ----------------------------------------------------
        # Tabla
        # ----------------------------------------------------

        file.write(
            "TIEMPOS DE EJECUCIÓN\n"
        )

        file.write(
            "-" * 74
            + "\n"
        )

        file.write(
            f"{'p':>4} | "
            f"{'NumPy [s]':>16} | "
            f"{'sklearn [s]':>16} | "
            f"{'Bagging [s]':>16}\n"
        )

        file.write(
            "-" * 74
            + "\n"
        )

        for row in resultados:

            numpy_str = (
                formato_tiempo(
                    row["numpy"]
                )
            )

            sklearn_str = (
                formato_tiempo(
                    row["sklearn"]
                )
            )

            auto_str = (
                formato_tiempo(
                    row["auto"]
                )
            )

            file.write(
                f"{row['p']:>4} | "
                f"{numpy_str:>16} | "
                f"{sklearn_str:>16} | "
                f"{auto_str:>16}\n"
            )

        file.write(
            "-" * 74
            + "\n\n"
        )

        # ----------------------------------------------------
        # Detalle
        # ----------------------------------------------------

        file.write(
            "DETALLE POR IMPLEMENTACIÓN\n"
        )

        file.write(
            "=" * 74
            + "\n\n"
        )

        implementaciones = [
            (
                "NumPy",
                "numpy"
            ),
            (
                "sklearn",
                "sklearn"
            ),
            (
                "BaggingRegressor",
                "auto"
            ),
        ]

        for nombre, clave in implementaciones:

            file.write(
                f"{nombre}\n"
            )

            file.write(
                "-" * len(nombre)
                + "\n"
            )

            for row in resultados:

                if row[clave] is None:

                    file.write(
                        f"p = {row['p']:>2} "
                        f"-> pendiente\n"
                    )

                else:

                    file.write(
                        f"p = {row['p']:>2} "
                        f"-> "
                        f"T(p) = "
                        f"{row[clave]:.2f} s\n"
                    )

            file.write("\n")

        # ----------------------------------------------------
        # Mejores tiempos disponibles
        # ----------------------------------------------------

        file.write(
            "MEJORES TIEMPOS OBSERVADOS\n"
        )

        file.write(
            "-" * 74
            + "\n"
        )

        for nombre, clave in implementaciones:

            validos = [
                row
                for row in resultados
                if row[clave] is not None
            ]

            if not validos:

                file.write(
                    f"{nombre:<18}: "
                    f"sin resultados aún\n"
                )

                continue

            mejor = min(
                validos,
                key=lambda row:
                    row[clave]
            )

            file.write(
                f"{nombre:<18}: "
                f"{mejor[clave]:.2f} s "
                f"con p = "
                f"{mejor['p']}\n"
            )


# ============================================================
# GUARDAR CSV FINAL
# ============================================================

def guardar_csv(resultados):

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
                "thread_limit",
            ]
        )

        writer.writeheader()

        for row in resultados:

            writer.writerow(
                {
                    "p": row["p"],
                    "numpy": row["numpy"],
                    "sklearn": row["sklearn"],
                    "auto": row["auto"],
                    "thread_limit":
                        THREAD_LIMIT,
                }
            )


# ============================================================
# GUARDAR AVANCE
# ============================================================

def guardar_avance(
    resultados,
    p_max,
    cpu_model
):

    guardar_checkpoint(
        resultados
    )

    guardar_txt(
        resultados,
        p_max,
        cpu_model
    )


# ============================================================
# MOSTRAR RESUMEN
# ============================================================

def mostrar_resumen(resultados):

    print()

    print(
        "=" * 74
    )

    print(
        "RESUMEN ACTUAL"
    )

    print(
        "=" * 74
    )

    print(
        f"{'p':>4} | "
        f"{'NumPy [s]':>14} | "
        f"{'sklearn [s]':>14} | "
        f"{'Bagging [s]':>14}"
    )

    print(
        "-" * 68
    )

    for row in resultados:

        numpy_str = (
            "-"
            if row["numpy"] is None
            else f"{row['numpy']:.2f}"
        )

        sklearn_str = (
            "-"
            if row["sklearn"] is None
            else f"{row['sklearn']:.2f}"
        )

        auto_str = (
            "-"
            if row["auto"] is None
            else f"{row['auto']:.2f}"
        )

        print(
            f"{row['p']:>4} | "
            f"{numpy_str:>14} | "
            f"{sklearn_str:>14} | "
            f"{auto_str:>14}"
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
    # Información del sistema
    # --------------------------------------------------------

    p_max = (
        os.cpu_count()
        or 1
    )

    cpu_model = (
        get_cpu_model()
    )

    print(
        "=" * 74
    )

    print(
        "ÍTEM (f): BENCHMARK SEGÚN "
        "NÚMERO DE PROCESOS"
    )

    print(
        "=" * 74
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
        f"Threads internos fijados: "
        f"{THREAD_LIMIT}"
    )

    print(
        f"Se evaluará "
        f"p = 1, 2, ..., {p_max}"
    )

    # --------------------------------------------------------
    # Resultados / posible checkpoint
    # --------------------------------------------------------

    resultados = (
        crear_resultados_vacios(
            p_max
        )
    )

    resultados = (
        cargar_checkpoint(
            resultados
        )
    )

    mostrar_resumen(
        resultados
    )

    # --------------------------------------------------------
    # Dataset común
    # --------------------------------------------------------

    print(
        "\nGenerando dataset..."
    )

    X, y, beta_true = (
        generate_data()
    )

    del beta_true

    print(
        f"X: {X.shape}"
    )

    print(
        f"y: {y.shape}"
    )

    print(
        "Dataset generado correctamente."
    )

    # --------------------------------------------------------
    # Implementaciones
    # --------------------------------------------------------

    implementaciones = [
        (
            "NumPy",
            "numpy",
            bootstrap_numpy
        ),
        (
            "sklearn",
            "sklearn",
            bootstrap_sklearn
        ),
        (
            "BaggingRegressor",
            "auto",
            bootstrap_auto
        ),
    ]

    total_pruebas = (
        p_max
        * len(implementaciones)
    )

    completadas = sum(
        row[clave] is not None
        for row in resultados
        for clave in [
            "numpy",
            "sklearn",
            "auto",
        ]
    )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    try:

        for row in resultados:

            p = row["p"]

            print()

            print(
                "=" * 74
            )

            print(
                f"p = {p} / {p_max}"
            )

            print(
                "=" * 74
            )

            for (
                nombre,
                clave,
                funcion
            ) in implementaciones:

                # Si ya está medido,
                # no repetirlo.
                if row[clave] is not None:

                    print(
                        f"{nombre}: "
                        f"ya medido "
                        f"({row[clave]:.2f} s)"
                    )

                    continue

                print(
                    f"\nPrueba "
                    f"{completadas + 1}"
                    f"/{total_pruebas}"
                )

                tiempo = medir_tiempo(
                    nombre,
                    funcion,
                    X,
                    y,
                    p
                )

                row[clave] = tiempo

                completadas += 1

                # Guardar inmediatamente.
                guardar_avance(
                    resultados,
                    p_max,
                    cpu_model
                )

                print(
                    "Resultado guardado."
                )

            mostrar_resumen(
                resultados
            )

    except KeyboardInterrupt:

        print(
            "\n\nEjecución interrumpida "
            "por el usuario."
        )

        print(
            "Los resultados ya terminados "
            "quedaron guardados."
        )

        print(
            "Puedes volver a ejecutar "
            "benchmark_f.py y continuará "
            "desde el checkpoint."
        )

        guardar_avance(
            resultados,
            p_max,
            cpu_model
        )

        return

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    guardar_txt(
        resultados,
        p_max,
        cpu_model
    )

    guardar_csv(
        resultados
    )

    # Ya no necesitamos el checkpoint
    # si todo terminó correctamente.
    if os.path.exists(
        CHECKPOINT_PATH
    ):

        os.remove(
            CHECKPOINT_PATH
        )

    print()

    print(
        "=" * 74
    )

    print(
        "ÍTEM (f) TERMINADO"
    )

    print(
        "=" * 74
    )

    mostrar_resumen(
        resultados
    )

    print(
        "\nArchivos finales:"
    )

    print(
        f"  {TXT_PATH}"
    )

    print(
        f"  {CSV_PATH}"
    )


if __name__ == "__main__":

    main()
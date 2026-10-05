import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTADOS = Path("resultados_daniela")
P_VALUES = [1, 2, 4, 8]
N_BLOQUE_DYC = 128


# ---------------------------------------------------------
# Lectura Algoritmo 1
# ---------------------------------------------------------
def leer_algoritmo1():
    archivo = RESULTADOS / "parte_b_algoritmo1.txt"

    tiempos = {p: [] for p in P_VALUES}

    patron = re.compile(
        r"N=2048 threads=(\d+) tiempo=([\d.]+) ms checksum=([\d.\-]+)"
    )

    with open(archivo, "r") as f:
        for linea in f:
            match = patron.search(linea)
            if match:
                p = int(match.group(1))
                tiempo = float(match.group(2))

                if p in tiempos:
                    tiempos[p].append(tiempo)

    filas = []

    for p in P_VALUES:
        promedio = np.mean(tiempos[p])
        std = np.std(tiempos[p])

        filas.append({
            "algoritmo": "OpenMP Datos",
            "p": p,
            "tiempo_ms": promedio,
            "std_ms": std
        })

    return pd.DataFrame(filas)


# ---------------------------------------------------------
# Lectura Numba
# ---------------------------------------------------------
def leer_numba():
    archivo = RESULTADOS / "parte_d_numba.txt"

    tiempos = {p: [] for p in P_VALUES}

    patron = re.compile(
        r"N=2048 threads=(\d+) tiempo=([\d.]+) ms checksum=([\d.\-]+)"
    )

    with open(archivo, "r") as f:
        for linea in f:
            match = patron.search(linea)
            if match:
                p = int(match.group(1))
                tiempo = float(match.group(2))

                if p in tiempos:
                    tiempos[p].append(tiempo)

    filas = []

    for p in P_VALUES:
        promedio = np.mean(tiempos[p])
        std = np.std(tiempos[p])

        filas.append({
            "algoritmo": "Numba",
            "p": p,
            "tiempo_ms": promedio,
            "std_ms": std
        })

    return pd.DataFrame(filas)


# ---------------------------------------------------------
# Lectura Divide & Conquer
# ---------------------------------------------------------
def leer_dyc():
    archivo = RESULTADOS / "algoritmo2_experiments.csv"
    df = pd.read_csv(archivo)

    df = df[df["n"] == N_BLOQUE_DYC].copy()

    df = df.rename(columns={
        "avg_time_ms": "tiempo_ms",
        "std_dev_ms": "std_ms"
    })

    df["algoritmo"] = f"OpenMP D&C (n={N_BLOQUE_DYC})"

    return df[["algoritmo", "p", "tiempo_ms", "std_ms"]]


# ---------------------------------------------------------
# Calcular speedup y eficiencia
# ---------------------------------------------------------
def calcular_metricas(df):
    resultados = []

    for algoritmo, grupo in df.groupby("algoritmo"):
        grupo = grupo.sort_values("p").copy()

        t1 = grupo.loc[grupo["p"] == 1, "tiempo_ms"].iloc[0]

        grupo["speedup"] = t1 / grupo["tiempo_ms"]
        grupo["eficiencia"] = grupo["speedup"] / grupo["p"]

        resultados.append(grupo)

    return pd.concat(resultados, ignore_index=True)


# ---------------------------------------------------------
# Gráfico T(p)
# ---------------------------------------------------------
def grafico_tiempo(df):
    plt.figure(figsize=(8, 5))

    for algoritmo, grupo in df.groupby("algoritmo"):
        grupo = grupo.sort_values("p")

        plt.plot(
            grupo["p"],
            grupo["tiempo_ms"],
            marker="o",
            label=algoritmo
        )

    plt.xlabel("Número de threads p")
    plt.ylabel("Tiempo de ejecución (ms)")
    plt.title("Tiempo de ejecución T(p), N = 2048")
    plt.xticks(P_VALUES)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTADOS / "e_tiempo.png",
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# Gráfico S(p)
# ---------------------------------------------------------
def grafico_speedup(df):
    plt.figure(figsize=(8, 5))

    for algoritmo, grupo in df.groupby("algoritmo"):
        grupo = grupo.sort_values("p")

        plt.plot(
            grupo["p"],
            grupo["speedup"],
            marker="o",
            label=algoritmo
        )

    # Speedup ideal
    plt.plot(
        P_VALUES,
        P_VALUES,
        linestyle="--",
        label="Ideal S(p)=p"
    )

    plt.xlabel("Número de threads p")
    plt.ylabel("Speedup S(p)")
    plt.title("Speedup en función del número de threads")
    plt.xticks(P_VALUES)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTADOS / "e_speedup.png",
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# Gráfico E(p)
# ---------------------------------------------------------
def grafico_eficiencia(df):
    plt.figure(figsize=(8, 5))

    for algoritmo, grupo in df.groupby("algoritmo"):
        grupo = grupo.sort_values("p")

        plt.plot(
            grupo["p"],
            grupo["eficiencia"],
            marker="o",
            label=algoritmo
        )

    # Eficiencia ideal = 1
    plt.axhline(
        y=1.0,
        linestyle="--",
        label="Ideal E(p)=1"
    )

    plt.xlabel("Número de threads p")
    plt.ylabel("Eficiencia E(p)")
    plt.title("Eficiencia en función del número de threads")
    plt.xticks(P_VALUES)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTADOS / "e_eficiencia.png",
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# Efecto del tamaño de bloque n en Algoritmo 2
# ---------------------------------------------------------
def grafico_bloques():
    archivo = RESULTADOS / "algoritmo2_experiments.csv"

    df = pd.read_csv(archivo)

    plt.figure(figsize=(8, 5))

    for p in P_VALUES:
        subset = df[df["p"] == p].sort_values("n")

        plt.plot(
            subset["n"],
            subset["avg_time_ms"],
            marker="o",
            label=f"p={p}"
        )

    plt.xscale("log", base=2)
    plt.xlabel("Tamaño de bloque n")
    plt.ylabel("Tiempo promedio (ms)")
    plt.title("Efecto del tamaño de bloque en Divide & Conquer")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTADOS / "e_tamano_bloque.png",
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    df_a1 = leer_algoritmo1()
    df_numba = leer_numba()
    df_dyc = leer_dyc()

    df = pd.concat(
        [df_a1, df_dyc, df_numba],
        ignore_index=True
    )

    df = calcular_metricas(df)

    # Mostrar tabla
    tabla = df[
        [
            "algoritmo",
            "p",
            "tiempo_ms",
            "speedup",
            "eficiencia"
        ]
    ].copy()

    tabla["tiempo_ms"] = tabla["tiempo_ms"].round(2)
    tabla["speedup"] = tabla["speedup"].round(2)
    tabla["eficiencia"] = tabla["eficiencia"].round(2)

    print("\n=== RESULTADOS PARTE (e) ===")
    print(tabla.to_string(index=False))

    # Guardar tabla
    tabla.to_csv(
        RESULTADOS / "parte_e_metricas.csv",
        index=False
    )

    # Gráficos
    grafico_tiempo(df)
    grafico_speedup(df)
    grafico_eficiencia(df)
    grafico_bloques()

    print("\nArchivos generados:")
    print(" - resultados_daniela/parte_e_metricas.csv")
    print(" - resultados_daniela/e_tiempo.png")
    print(" - resultados_daniela/e_speedup.png")
    print(" - resultados_daniela/e_eficiencia.png")
    print(" - resultados_daniela/e_tamano_bloque.png")


if __name__ == "__main__":
    main()
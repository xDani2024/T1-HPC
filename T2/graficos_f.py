import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


RESULTADOS = Path("resultados_daniela")
CSV = RESULTADOS / "weak_scaling_experiments.csv"

P_VALUES = [1, 2, 4, 8]
N_VALUES = [1024, 2048, 4096]


def nombre_corto(nombre):
    if "Algoritmo 1" in nombre:
        return "OpenMP Datos"
    if "Algoritmo 2" in nombre:
        return "OpenMP D&C"
    if "Algoritmo 3" in nombre:
        return "Numba"
    return nombre


def cargar_datos():
    df = pd.read_csv(CSV)

    df["Algorithm_Short"] = df["Algorithm"].apply(nombre_corto)

    return df


# ---------------------------------------------------------
# Gráfico 1:
# Tiempo vs N para cada algoritmo y cantidad de threads
# ---------------------------------------------------------
def grafico_tiempo_vs_n(df):

    for algoritmo in df["Algorithm_Short"].unique():

        subset_alg = df[
            df["Algorithm_Short"] == algoritmo
        ]

        plt.figure(figsize=(8, 5))

        for p in P_VALUES:

            subset = subset_alg[
                subset_alg["p"] == p
            ].sort_values("N")

            plt.plot(
                subset["N"],
                subset["Avg_Time_ms"],
                marker="o",
                label=f"p={p}"
            )

        plt.xlabel("Tamaño de matriz N")
        plt.ylabel("Tiempo promedio (ms)")
        plt.title(
            f"Escalamiento con el tamaño del problema - {algoritmo}"
        )

        plt.xticks(N_VALUES)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        nombre_archivo = (
            algoritmo
            .lower()
            .replace(" ", "_")
            .replace("&", "and")
        )

        plt.savefig(
            RESULTADOS / f"f_tiempo_{nombre_archivo}.png",
            dpi=300
        )

        plt.close()


# ---------------------------------------------------------
# Gráfico 2:
# Eficiencia débil real
#
# Solo casos con N^3 / p igual al caso base:
#
# (1024, 1)
# (2048, 8)
# ---------------------------------------------------------
def grafico_eficiencia_debil(df):

    weak = df[
        df["Weak_Scaling_Match"] == "YES"
    ].copy()

    weak = weak[
        weak["Weak_Efficiency"].notna()
    ]

    plt.figure(figsize=(8, 5))

    for algoritmo in weak["Algorithm_Short"].unique():

        subset = weak[
            weak["Algorithm_Short"] == algoritmo
        ].sort_values("N")

        etiquetas = [
            f"N={int(row.N)}, p={int(row.p)}"
            for _, row in subset.iterrows()
        ]

        plt.plot(
            etiquetas,
            subset["Weak_Efficiency"],
            marker="o",
            label=algoritmo
        )

    plt.axhline(
        y=1.0,
        linestyle="--",
        label="Ideal"
    )

    plt.xlabel("Configuración")
    plt.ylabel("Eficiencia débil")
    plt.title("Eficiencia débil con carga constante por thread")

    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTADOS / "f_eficiencia_debil.png",
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# Tabla resumen weak scaling
# ---------------------------------------------------------
def tabla_weak_scaling(df):

    weak = df[
        df["Weak_Scaling_Match"] == "YES"
    ].copy()

    weak = weak[
        [
            "Algorithm_Short",
            "N",
            "p",
            "Avg_Time_ms",
            "Weak_Efficiency",
            "Checksum_Match"
        ]
    ]

    weak = weak.rename(columns={
        "Algorithm_Short": "Algoritmo",
        "Avg_Time_ms": "Tiempo_ms",
        "Weak_Efficiency": "Eficiencia_Debil"
    })

    weak["Tiempo_ms"] = weak["Tiempo_ms"].round(2)
    weak["Eficiencia_Debil"] = weak[
        "Eficiencia_Debil"
    ].round(4)

    weak.to_csv(
        RESULTADOS / "parte_f_resumen_weak_scaling.csv",
        index=False
    )

    print("\n=== CONFIGURACIONES WEAK SCALING VÁLIDAS ===")
    print(weak.to_string(index=False))


# ---------------------------------------------------------
# Tabla completa para comentar N=4096
# ---------------------------------------------------------
def tabla_completa(df):

    tabla = df[
        [
            "Algorithm_Short",
            "N",
            "p",
            "Avg_Time_ms",
            "Std_Dev_ms",
            "Work_Per_Thread_Ratio"
        ]
    ].copy()

    tabla = tabla.rename(columns={
        "Algorithm_Short": "Algoritmo",
        "Avg_Time_ms": "Tiempo_ms",
        "Std_Dev_ms": "Desv_ms",
        "Work_Per_Thread_Ratio": "Carga_relativa_thread"
    })

    tabla["Tiempo_ms"] = tabla["Tiempo_ms"].round(2)
    tabla["Desv_ms"] = tabla["Desv_ms"].round(2)

    tabla.to_csv(
        RESULTADOS / "parte_f_resultados_completos.csv",
        index=False
    )


def main():

    df = cargar_datos()

    grafico_tiempo_vs_n(df)
    grafico_eficiencia_debil(df)

    tabla_weak_scaling(df)
    tabla_completa(df)

    print("\nArchivos generados:")

    print(
        " - resultados_daniela/"
        "f_tiempo_openmp_datos.png"
    )

    print(
        " - resultados_daniela/"
        "f_tiempo_openmp_dandc.png"
    )

    print(
        " - resultados_daniela/"
        "f_tiempo_numba.png"
    )

    print(
        " - resultados_daniela/"
        "f_eficiencia_debil.png"
    )

    print(
        " - resultados_daniela/"
        "parte_f_resumen_weak_scaling.csv"
    )

    print(
        " - resultados_daniela/"
        "parte_f_resultados_completos.csv"
    )


if __name__ == "__main__":
    main()
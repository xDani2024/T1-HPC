import os
import sys
import numpy as np

from time import perf_counter
from contextlib import redirect_stdout

from T1.common import (
    generate_data,
    confidence_interval
)

from T1.bs_numpy import bootstrap_numpy
from T1.bs_sklearn import bootstrap_sklearn
from T1.bs_auto import bootstrap_auto


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


def analyze_version(name, betas, beta_true):
    """
    Calcula los intervalos de confianza al 95 %
    y verifica cuántos coeficientes verdaderos
    quedan dentro de sus respectivos intervalos.
    """

    lower, upper = confidence_interval(betas)

    inside = (
        (beta_true >= lower)
        & (beta_true <= upper)
    )

    n_inside = np.sum(inside)
    coverage = np.mean(inside)

    print(f"\n{name}")
    print("-" * 40)

    print(
        "Coeficientes dentro del IC:",
        n_inside,
        "/",
        len(beta_true)
    )

    print(
        f"Cobertura: {coverage * 100:.2f}%"
    )

    print("\nPrimeros 5 intervalos:")

    for j in range(5):
        print(
            f"beta[{j}] = {beta_true[j]:.6f} | "
            f"IC = [{lower[j]:.6f}, {upper[j]:.6f}]"
        )

    return lower, upper


def run_analysis():

    print("Generando dataset común...")

    X, y, beta_true = generate_data()

    p = 1

    print(f"\nNúmero de procesos utilizado: p = {p}")

    # NumPy
    print("\nEjecutando bs_numpy...")

    start = perf_counter()

    betas_numpy = bootstrap_numpy(
        X,
        y,
        p
    )

    time_numpy = perf_counter() - start

    print(
        f"Tiempo NumPy: {time_numpy:.2f} s"
    )

    # sklearn
    print("\nEjecutando bs_sklearn...")

    start = perf_counter()

    betas_sklearn = bootstrap_sklearn(
        X,
        y,
        p
    )

    time_sklearn = perf_counter() - start

    print(
        f"Tiempo sklearn: {time_sklearn:.2f} s"
    )

    # BaggingRegressor
    print("\nEjecutando bs_auto...")

    start = perf_counter()

    betas_auto = bootstrap_auto(
        X,
        y,
        p
    )

    time_auto = perf_counter() - start

    print(
        f"Tiempo auto: {time_auto:.2f} s"
    )

    # Intervalos de confianza
    print("\n")
    print("=" * 60)
    print("ANÁLISIS DE CORRECTITUD")
    print("=" * 60)

    lower_numpy, upper_numpy = analyze_version(
        "NumPy",
        betas_numpy,
        beta_true
    )

    lower_sklearn, upper_sklearn = analyze_version(
        "sklearn",
        betas_sklearn,
        beta_true
    )

    lower_auto, upper_auto = analyze_version(
        "BaggingRegressor",
        betas_auto,
        beta_true
    )

    # Comparación NumPy vs sklearn
    print("\n")
    print("=" * 60)
    print("COMPARACIÓN NUMPY VS SKLEARN")
    print("=" * 60)

    max_beta_difference = np.max(
        np.abs(
            betas_numpy
            - betas_sklearn
        )
    )

    max_lower_difference = np.max(
        np.abs(
            lower_numpy
            - lower_sklearn
        )
    )

    max_upper_difference = np.max(
        np.abs(
            upper_numpy
            - upper_sklearn
        )
    )

    print(
        "Máxima diferencia entre betas:",
        max_beta_difference
    )

    print(
        "Máxima diferencia IC inferior:",
        max_lower_difference
    )

    print(
        "Máxima diferencia IC superior:",
        max_upper_difference
    )

    # Comparación NumPy vs BaggingRegressor
    print("\n")
    print("=" * 60)
    print("COMPARACIÓN NUMPY VS BAGGINGREGRESSOR")
    print("=" * 60)

    mean_lower_difference = np.mean(
        np.abs(
            lower_numpy
            - lower_auto
        )
    )

    mean_upper_difference = np.mean(
        np.abs(
            upper_numpy
            - upper_auto
        )
    )

    print(
        "Diferencia media IC inferior:",
        mean_lower_difference
    )

    print(
        "Diferencia media IC superior:",
        mean_upper_difference
    )

    # Comparación sklearn vs BaggingRegressor
    print("\n")
    print("=" * 60)
    print("COMPARACIÓN SKLEARN VS BAGGINGREGRESSOR")
    print("=" * 60)

    mean_lower_difference_sklearn_auto = np.mean(
        np.abs(
            lower_sklearn
            - lower_auto
        )
    )

    mean_upper_difference_sklearn_auto = np.mean(
        np.abs(
            upper_sklearn
            - upper_auto
        )
    )

    print(
        "Diferencia media IC inferior:",
        mean_lower_difference_sklearn_auto
    )

    print(
        "Diferencia media IC superior:",
        mean_upper_difference_sklearn_auto
    )

    # Resumen de tiempos
    print("\n")
    print("=" * 60)
    print("RESUMEN DE TIEMPOS")
    print("=" * 60)

    print(
        f"NumPy:             {time_numpy:.2f} s"
    )

    print(
        f"sklearn:           {time_sklearn:.2f} s"
    )

    print(
        f"BaggingRegressor:  {time_auto:.2f} s"
    )


def main():

    # Crear carpeta resultados si todavía no existe
    os.makedirs(
        "resultados",
        exist_ok=True
    )

    output_file = (
        "resultados/resultados_c.txt"
    )

    # Abrir archivo de resultados
    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        # Enviar cada print a terminal + archivo
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
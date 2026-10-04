#Parte B
import os
import sys
import numpy as np

from time import perf_counter
from contextlib import redirect_stdout
from joblib import Parallel, delayed

from T1.common import (
    B,
    generate_data,
    generate_bootstrap_seeds
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


def fit_inverse(X, y):
    """
    Versión inicial:
    calcula explícitamente la inversa de X.T @ X.
    """

    XtX = X.T @ X
    Xty = X.T @ y

    beta_hat = np.linalg.inv(XtX) @ Xty

    return beta_hat


def fit_solve(X, y):
    """
    Versión final:
    resuelve directamente el sistema lineal.
    """

    XtX = X.T @ X
    Xty = X.T @ y

    beta_hat = np.linalg.solve(
        XtX,
        Xty
    )

    return beta_hat


def bootstrap_fit_inverse(X, y, seed):

    rng = np.random.default_rng(seed)

    n = X.shape[0]

    indices = rng.integers(
        low=0,
        high=n,
        size=n
    )

    Xb = X[indices]
    yb = y[indices]

    return fit_inverse(
        Xb,
        yb
    )


def bootstrap_fit_solve(X, y, seed):

    rng = np.random.default_rng(seed)

    n = X.shape[0]

    indices = rng.integers(
        low=0,
        high=n,
        size=n
    )

    Xb = X[indices]
    yb = y[indices]

    return fit_solve(
        Xb,
        yb
    )


def run_bootstrap_inverse(X, y, p):

    seeds = generate_bootstrap_seeds()

    betas = Parallel(
        n_jobs=p,
        backend="multiprocessing"
    )(
        delayed(bootstrap_fit_inverse)(
            X,
            y,
            seeds[b]
        )
        for b in range(B)
    )

    return np.array(betas)


def run_bootstrap_solve(X, y, p):

    seeds = generate_bootstrap_seeds()

    betas = Parallel(
        n_jobs=p,
        backend="multiprocessing"
    )(
        delayed(bootstrap_fit_solve)(
            X,
            y,
            seeds[b]
        )
        for b in range(B)
    )

    return np.array(betas)


def run_analysis():

    print("=" * 60)
    print("COMPARACIÓN INV VS SOLVE")
    print("=" * 60)

    print("\nGenerando dataset...")

    X, y, beta_true = generate_data()

    p = 1

    print(
        f"Número de procesos utilizado: p = {p}"
    )

    # np.linalg.inv
    print("\nEjecutando versión con np.linalg.inv...")

    start = perf_counter()

    betas_inverse = run_bootstrap_inverse(
        X,
        y,
        p
    )

    time_inverse = perf_counter() - start

    print(
        f"Tiempo inv: {time_inverse:.2f} s"
    )

    # np.linalg.solve
    print("\nEjecutando versión con np.linalg.solve...")

    start = perf_counter()

    betas_solve = run_bootstrap_solve(
        X,
        y,
        p
    )

    time_solve = perf_counter() - start

    print(
        f"Tiempo solve: {time_solve:.2f} s"
    )

    # Correctitud
    max_difference = np.max(
        np.abs(
            betas_inverse
            - betas_solve
        )
    )

    print("\n")
    print("=" * 60)
    print("COMPARACIÓN DE RESULTADOS")
    print("=" * 60)

    print(
        "Máxima diferencia entre resultados:",
        max_difference
    )

    # Diferencia de rendimiento
    time_difference = (
        time_inverse
        - time_solve
    )

    percentage_change = (
        time_difference
        / time_inverse
    ) * 100

    print("\n")
    print("=" * 60)
    print("COMPARACIÓN DE TIEMPOS")
    print("=" * 60)

    print(
        f"np.linalg.inv:   {time_inverse:.2f} s"
    )

    print(
        f"np.linalg.solve: {time_solve:.2f} s"
    )

    print(
        f"Diferencia:      {time_difference:.2f} s"
    )

    print(
        f"Variación respecto de inv: "
        f"{percentage_change:.2f}%"
    )


def main():

    os.makedirs(
        "resultados",
        exist_ok=True
    )

    output_file = (
        "resultados/"
        "resultados_b_optimizacion.txt"
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
        f"\nResultados guardados en: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()
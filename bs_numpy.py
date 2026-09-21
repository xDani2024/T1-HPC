import numpy as np

from joblib import Parallel, delayed
from time import perf_counter

from common import (
    B,
    generate_data,
    generate_bootstrap_seeds,
)


def fit_numpy(X, y):
    """
    Calcula beta_hat resolviendo:
    (X.T @ X) beta_hat = X.T @ y
    """

    XtX = X.T @ X
    Xty = X.T @ y

    beta_hat = np.linalg.solve(
        XtX,
        Xty
    )

    return beta_hat


def bootstrap_fit(X, y, seed):
    """
    Ejecuta un único resample bootstrap y calcula beta_hat.
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

    # Calcular beta_hat para el resample
    beta_hat = fit_numpy(Xb, yb)

    return beta_hat


def bootstrap_numpy(X, y, p):
    """
    Ejecuta los B resamples utilizando p procesos.
    """

    seeds = generate_bootstrap_seeds()

    betas = Parallel(
        n_jobs=p,
        backend="multiprocessing"
    )(
        delayed(bootstrap_fit)(
            X,
            y,
            seeds[b]
        )
        for b in range(B)
    )

    return np.array(betas)


def main():
    print("Generando datos...")

    X, y, beta_true = generate_data()

    print("Datos generados.")
    print("X:", X.shape)
    print("y:", y.shape)

    # Paso 1: calcular beta_hat sobre el dataset completo
    print("\nCalculando beta_hat sobre el dataset completo...")

    beta_full = fit_numpy(X, y)

    print("Dimensión beta_hat:", beta_full.shape)

    print("\nPrimeros 5 coeficientes estimados con el dataset completo:")
    print(beta_full[:5])

    # Paso 2: bootstrap
    p = 1

    print()
    print(f"Ejecutando bootstrap con p = {p}...")

    start = perf_counter()

    betas = bootstrap_numpy(
        X,
        y,
        p
    )

    elapsed = perf_counter() - start

    print()
    print("Bootstrap terminado.")
    print("Dimensión de betas:", betas.shape)
    print(f"Tiempo: {elapsed:.2f} segundos")

    print()
    print("Primeros 5 coeficientes verdaderos:")
    print(beta_true[:5])

    print()
    print("Primeros 5 coeficientes estimados con dataset completo:")
    print(beta_full[:5])

    print()
    print("Primeros 5 coeficientes del primer bootstrap:")
    print(betas[0, :5])


if __name__ == "__main__":
    main()
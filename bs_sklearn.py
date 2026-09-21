import numpy as np

from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from time import perf_counter

from common import (
    B,
    generate_data,
    generate_bootstrap_seeds,
)


def fit_sklearn(X, y):

    model = LinearRegression(
        fit_intercept=False
    )

    model.fit(X, y)

    return model.coef_


def bootstrap_fit(X, y, seed):

    rng = np.random.default_rng(seed)

    n = X.shape[0]

    indices = rng.integers(
        low=0,
        high=n,
        size=n
    )

    Xb = X[indices]
    yb = y[indices]

    beta_hat = fit_sklearn(
        Xb,
        yb
    )

    return beta_hat


def bootstrap_sklearn(X, y, p):

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

    print("\nCalculando beta_hat sobre el dataset completo...")

    beta_full = fit_sklearn(
        X,
        y
    )

    print(
        "Dimensión beta_hat:",
        beta_full.shape
    )

    print(
        "\nPrimeros 5 coeficientes estimados con el dataset completo:"
    )
    print(beta_full[:5])

    p = 1

    print()
    print(
        f"Ejecutando bootstrap con p = {p}..."
    )

    start = perf_counter()

    betas = bootstrap_sklearn(
        X,
        y,
        p
    )

    elapsed = (
        perf_counter()
        - start
    )

    print()
    print("Bootstrap terminado.")
    print(
        "Dimensión de betas:",
        betas.shape
    )

    print(
        f"Tiempo: {elapsed:.2f} segundos"
    )

    print()
    print(
        "Primeros 5 coeficientes verdaderos:"
    )
    print(beta_true[:5])

    print()
    print(
        "Primeros 5 coeficientes estimados con dataset completo:"
    )
    print(beta_full[:5])

    print()
    print(
        "Primeros 5 coeficientes del primer bootstrap:"
    )
    print(betas[0, :5])


if __name__ == "__main__":
    main()
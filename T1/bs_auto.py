import numpy as np

from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from time import perf_counter

from T1.common import (
    B,
    SEED_BOOTSTRAP,
    generate_data,
)


def bootstrap_auto(X, y, p):

    base_model = LinearRegression(
        fit_intercept=False
    )

    model = BaggingRegressor(
        estimator=base_model,
        n_estimators=B,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        n_jobs=p,
        random_state=SEED_BOOTSTRAP
    )

    model.fit(X, y)

    betas = np.array([
        estimator.coef_
        for estimator in model.estimators_
    ])

    return betas


def main():

    print("Generando datos...")

    X, y, beta_true = generate_data()

    print("Datos generados.")
    print("X:", X.shape)
    print("y:", y.shape)

    print("\nCalculando beta_hat sobre el dataset completo...")

    model_full = LinearRegression(
        fit_intercept=False
    )

    model_full.fit(X, y)

    beta_full = model_full.coef_

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
        f"Ejecutando bootstrap automático con p = {p}..."
    )

    start = perf_counter()

    betas = bootstrap_auto(
        X,
        y,
        p
    )

    elapsed = perf_counter() - start

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
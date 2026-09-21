import numpy as np


# Parámetros definidos por el enunciado
N = 100_000  # observaciones
K = 300      # variables de entrada
B = 48       # resamples

# Semillas fijas
SEED_DATA = 3533
SEED_BOOTSTRAP = 1234


def generate_data():

    rng = np.random.default_rng(SEED_DATA)

    # (i) k + 1 coeficientes verdaderos beta*
    beta_true = rng.normal(
        loc=0.0,
        scale=1.0,
        size=K + 1
    )

    # (ii) Matriz de datos N x K
    X_data = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(N, K)
    )

    # Agregar columna de unos al inicio
    X = np.column_stack(
        (
            np.ones(N),
            X_data
        )
    )

    # (iii) Ruido N(0,1)
    noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=N
    )

    # y = X beta* + ruido
    y = X @ beta_true + noise

    return X, y, beta_true


def generate_bootstrap_seeds():
    seed_sequence = np.random.SeedSequence(SEED_BOOTSTRAP)
    return seed_sequence.spawn(B)

def confidence_interval(betas):
    """
    Calcula el intervalo de confianza bootstrap al 95 %
    para cada coeficiente.
    """

    lower = np.percentile(
        betas,
        2.5,
        axis=0
    )

    upper = np.percentile(
        betas,
        97.5,
        axis=0
    )

    return lower, upper

if __name__ == "__main__":

    X, y, beta_true = generate_data()

    # Comprobaciones
    print("Comprobaciones")

    print("Dimensión X:", X.shape)
    print("Dimensión y:", y.shape)
    print("Dimensión beta*:", beta_true.shape)

    print("\nPrimeras 3 filas de X:")
    print(X[:3, :5])

    print("\nPrimeros 5 valores de y:")
    print(y[:5])

    print("\nPrimeros 5 coeficientes beta*:")
    print(beta_true[:5])
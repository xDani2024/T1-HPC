import os
import sys
import numpy as np

from contextlib import redirect_stdout

from common import generate_data
from bs_numpy import bootstrap_numpy
from bs_sklearn import bootstrap_sklearn
from bs_auto import bootstrap_auto


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


def compare_runs(name, run1, run2):
    """
    Compara dos ejecuciones de una misma versión
    para verificar reproducibilidad.
    """

    max_difference = np.max(
        np.abs(run1 - run2)
    )

    equivalent = np.allclose(
        run1,
        run2
    )

    print(f"\n{name}")
    print("-" * 40)

    print(
        "Resultados equivalentes:",
        equivalent
    )

    print(
        "Máxima diferencia:",
        max_difference
    )


def run_analysis():

    print("=" * 60)
    print("PRUEBA DE REPRODUCIBILIDAD")
    print("=" * 60)

    # Reproducibilidad de los datos sintéticos
    print("\nGenerando dataset dos veces...")

    X1, y1, beta_true1 = generate_data()
    X2, y2, beta_true2 = generate_data()

    print(
        "X reproducible:",
        np.array_equal(X1, X2)
    )

    print(
        "y reproducible:",
        np.array_equal(y1, y2)
    )

    print(
        "beta* reproducible:",
        np.array_equal(
            beta_true1,
            beta_true2
        )
    )

    # Para esta prueba utilizamos un solo proceso
    p = 1

    print(
        f"\nNúmero de procesos utilizado: p = {p}"
    )

    # NumPy
    print("\nEjecutando NumPy - corrida 1...")

    numpy_run1 = bootstrap_numpy(
        X1,
        y1,
        p
    )

    print("Ejecutando NumPy - corrida 2...")

    numpy_run2 = bootstrap_numpy(
        X1,
        y1,
        p
    )

    compare_runs(
        "NumPy",
        numpy_run1,
        numpy_run2
    )

    # sklearn
    print("\nEjecutando sklearn - corrida 1...")

    sklearn_run1 = bootstrap_sklearn(
        X1,
        y1,
        p
    )

    print("Ejecutando sklearn - corrida 2...")

    sklearn_run2 = bootstrap_sklearn(
        X1,
        y1,
        p
    )

    compare_runs(
        "sklearn",
        sklearn_run1,
        sklearn_run2
    )

    # BaggingRegressor
    print(
        "\nEjecutando BaggingRegressor - corrida 1..."
    )

    auto_run1 = bootstrap_auto(
        X1,
        y1,
        p
    )

    print(
        "Ejecutando BaggingRegressor - corrida 2..."
    )

    auto_run2 = bootstrap_auto(
        X1,
        y1,
        p
    )

    compare_runs(
        "BaggingRegressor",
        auto_run1,
        auto_run2
    )

    # Resumen
    print("\n")
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)

    print(
        "Dataset reproducible:",
        np.array_equal(X1, X2)
        and np.array_equal(y1, y2)
        and np.array_equal(beta_true1, beta_true2)
    )

    print(
        "NumPy reproducible:",
        np.allclose(
            numpy_run1,
            numpy_run2
        )
    )

    print(
        "sklearn reproducible:",
        np.allclose(
            sklearn_run1,
            sklearn_run2
        )
    )

    print(
        "BaggingRegressor reproducible:",
        np.allclose(
            auto_run1,
            auto_run2
        )
    )


def main():

    # Crear carpeta resultados si todavía no existe
    os.makedirs(
        "resultados",
        exist_ok=True
    )

    output_file = (
        "resultados/reproducibilidad.txt"
    )

    # Mostrar en terminal y guardar en archivo
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
        f"\nResultados guardados en: {output_file}"
    )


if __name__ == "__main__":
    main()
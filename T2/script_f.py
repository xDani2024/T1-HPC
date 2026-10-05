import csv
import os
import subprocess
import sys
import time
import numpy as np


def run_cpp_program(executable, N, n, p):
    """Ejecuta un ejecutable C++ y retorna tiempo en ms y checksum."""
    if n is not None:
        cmd = [f"./{executable}", str(N), str(n), str(p)]
    else:
        cmd = [f"./{executable}", str(N), str(p)]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )

    output = result.stdout.strip()

    values = {
        key: value
        for token in output.split()
        if "=" in token
        for key, value in [token.split("=", 1)]
    }

    return float(values["tiempo"]), float(values["checksum"])


def run_python_program(N, p):
    """Ejecuta Algoritmo 3 y retorna tiempo en ms y checksum."""
    cmd = [sys.executable, "algoritmo3.py", str(N), str(p)]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )

    output = result.stdout.strip()

    values = {
        key: value
        for token in output.split()
        if "=" in token
        for key, value in [token.split("=", 1)]
    }

    return float(values["tiempo"]), float(values["checksum"])


def main():

    # -----------------------------------------------------
    # Parámetros
    # -----------------------------------------------------

    N_values = [1024, 2048, 4096]
    p_values = [1, 2, 4, 8]

    # Tamaño elegido para Divide & Conquer
    n_optimal = 128

    runs_per_config = 3

    csv_filename = "resultados_daniela/weak_scaling_experiments.csv"

    os.makedirs("resultados_daniela", exist_ok=True)

    headers = [
        "Algorithm",
        "N",
        "p",
        "Run_1_ms",
        "Run_2_ms",
        "Run_3_ms",
        "Avg_Time_ms",
        "Std_Dev_ms",
        "Work_Per_Thread_Ratio",
        "Weak_Scaling_Match",
        "Weak_Efficiency",
        "Checksum_Match",
    ]

    print(
        "=== EXPERIMENTOS DE ESCALABILIDAD DÉBIL ==="
    )

    # -----------------------------------------------------
    # Compilación
    # -----------------------------------------------------

    print("\n[+] Compilando C++...")

    os.system(
        "g++ -O3 -march=native -fopenmp "
        "algoritmo1.cpp -o algoritmo1"
    )

    os.system(
        "g++ -O3 -march=native -fopenmp "
        "algoritmo2.cpp -o algoritmo2"
    )

    # -----------------------------------------------------
    # Algoritmos
    # -----------------------------------------------------

    algorithms = [
        (
            "Algoritmo 1 (C++ Data Parallel)",
            "cpp1"
        ),
        (
            "Algoritmo 2 (C++ Divide & Conquer)",
            "cpp2"
        ),
        (
            "Algoritmo 3 (Python Numba)",
            "python"
        ),
    ]

    # Referencia:
    #
    # trabajo base por thread =
    # 1024^3 / 1
    #
    BASE_N = 1024
    BASE_P = 1

    base_work_per_thread = (BASE_N ** 3) / BASE_P

    # -----------------------------------------------------
    # Crear CSV desde cero
    # -----------------------------------------------------

    with open(
        csv_filename,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)
        writer.writerow(headers)

    # -----------------------------------------------------
    # Experimentos
    # -----------------------------------------------------

    for alg_label, alg_type in algorithms:

        print(
            f"\n---> Evaluando: {alg_label}"
        )

        base_time = None

        for N in N_values:

            for p in p_values:

                times = []
                checksums = []

                print(
                    f"   N={N}, p={p} "
                    f"({runs_per_config} repeticiones)...",
                    end="",
                    flush=True,
                )

                for _ in range(runs_per_config):

                    if alg_type == "cpp1":

                        t, chk = run_cpp_program(
                            "algoritmo1",
                            N,
                            None,
                            p
                        )

                    elif alg_type == "cpp2":

                        t, chk = run_cpp_program(
                            "algoritmo2",
                            N,
                            n_optimal,
                            p
                        )

                    else:

                        t, chk = run_python_program(
                            N,
                            p
                        )

                    times.append(t)
                    checksums.append(chk)

                    time.sleep(0.2)

                avg_t = np.mean(times)
                std_t = np.std(times)

                chk_consistent = np.allclose(
                    checksums,
                    checksums[0],
                    atol=1e-4
                )

                # -----------------------------------------
                # Tiempo base T1(1024)
                # -----------------------------------------

                if N == BASE_N and p == BASE_P:

                    base_time = avg_t

                # -----------------------------------------
                # Trabajo por thread
                # -----------------------------------------

                current_work_per_thread = (N ** 3) / p

                work_ratio = (
                    current_work_per_thread
                    / base_work_per_thread
                )

                # -----------------------------------------
                # ¿Es una configuración weak scaling real?
                # -----------------------------------------

                weak_match = np.isclose(
                    work_ratio,
                    1.0,
                    rtol=1e-10
                )

                # -----------------------------------------
                # Eficiencia débil
                #
                # SOLO se calcula cuando N^3/p
                # coincide con el caso base.
                # -----------------------------------------

                if weak_match and base_time is not None:

                    weak_efficiency = (
                        base_time / avg_t
                    )

                    weak_eff_str = (
                        f"{weak_efficiency:.4f}"
                    )

                else:

                    weak_efficiency = None
                    weak_eff_str = ""

                # -----------------------------------------

                if weak_efficiency is not None:

                    print(
                        f" Promedio: {avg_t:.2f} ms"
                        f" | Eweak: "
                        f"{weak_efficiency:.2%}"
                    )

                else:

                    print(
                        f" Promedio: {avg_t:.2f} ms"
                        f" | carga/thread = "
                        f"{work_ratio:.2f}x base"
                    )

                # -----------------------------------------
                # CSV
                # -----------------------------------------

                with open(
                    csv_filename,
                    mode="a",
                    newline="",
                    encoding="utf-8"
                ) as f:

                    writer = csv.writer(f)

                    writer.writerow([
                        alg_label,
                        N,
                        p,
                        f"{times[0]:.2f}",
                        f"{times[1]:.2f}",
                        f"{times[2]:.2f}",
                        f"{avg_t:.2f}",
                        f"{std_t:.2f}",
                        f"{work_ratio:.4f}",
                        "YES" if weak_match else "NO",
                        weak_eff_str,
                        "YES" if chk_consistent else "NO",
                    ])

    print(
        "\n[✓] Experimentos completados."
    )

    print(
        f"[✓] Resultados: {csv_filename}"
    )

    print(
        "\nConfiguraciones con carga por thread "
        "equivalente al caso base:"
    )

    print(
        "  N=1024, p=1"
    )

    print(
        "  N=2048, p=8"
    )

    print(
        "\nPara N=4096 serían necesarios "
        "p=64 threads para conservar "
        "exactamente N^3/p."
    )


if __name__ == "__main__":
    main()
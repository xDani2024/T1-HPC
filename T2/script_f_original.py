import csv
import os
import subprocess
import sys
import time
import numpy as np


def run_cpp_program(executable, N, n, p):
    """Ejecuta un ejecutable C++ y retorna el tiempo en ms y el checksum."""
    if n is not None:
        cmd = [f"./{executable}", str(N), str(n), str(p)]
    else:
        cmd = [f"./{executable}", str(N), str(p)]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output = result.stdout.strip()

    # El algoritmo 2 incluye el campo adicional n=..., por lo que no se
    # deben usar posiciones fijas para localizar tiempo y checksum.
    values = {
        key: value
        for token in output.split()
        if "=" in token
        for key, value in [token.split("=", 1)]
    }
    return float(values["tiempo"]), float(values["checksum"])


def run_python_program(N, p):
    """Ejecuta el script de Python (Algoritmo 3) y retorna tiempo en ms y checksum."""
    cmd = [sys.executable, "algoritmo3.py", str(N), str(p)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output = result.stdout.strip()

    values = {
        key: value
        for token in output.split()
        if "=" in token
        for key, value in [token.split("=", 1)]
    }
    return float(values["tiempo"]), float(values["checksum"])


def main():
    # Parámetros del experimento
    N_values = [1024, 2048, 4096]
    p_values = [1, 2, 4, 8]
    n_optimal = 64  # Ajusta este valor al n óptimo que encontraste en el inciso (a)
    runs_per_config = 3

    csv_filename = "weak_scaling_experiments.csv"

    # Encabezados del archivo CSV
    headers = [
        "Algorithm",
        "N",
        "p",
        "Run_1_ms",
        "Run_2_ms",
        "Run_3_ms",
        "Avg_Time_ms",
        "Std_Dev_ms",
        "Weak_Efficiency",
        "Checksum_Match",
    ]

    print("=== INICIANDO EXPERIMENTOS DE ESCALABILIDAD DÉBIL (INCISO F) ===")

    # 1. Compilar los ejecutables de C++
    print("\n[+] Compilando ejecutables C++...")
    # os.system("g++-16 -O3 -mcpu=apple-m1 -fopenmp algoritmo1.cpp -o algoritmo1")
    # os.system("g++-16 -O3 -mcpu=apple-m1 -fopenmp algoritmo2.cpp -o algoritmo2")

    os.system("g++ -O3 -march=native -fopenmp algoritmo1.cpp -o algoritmo1")
    os.system("g++ -O3 -march=native -fopenmp algoritmo2.cpp -o algoritmo2")

    completed_configs = set()
    base_times = {}

    if os.path.exists(csv_filename) and os.path.getsize(csv_filename) > 0:
        with open(csv_filename, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                algorithm = row["Algorithm"]
                N = int(row["N"])
                p = int(row["p"])
                completed_configs.add((algorithm, N, p))
                if N == 1024 and p == 1:
                    base_times[algorithm] = float(row["Avg_Time_ms"])
    else:
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            f.flush()
            os.fsync(f.fileno())

    # Definir los algoritmos a evaluar
    algorithms = [
        ("Algoritmo 1 (C++ Data Parallel)", "cpp1"),
        ("Algoritmo 2 (C++ Divide & Conquer)", "cpp2"),
        ("Algoritmo 3 (Python Numba)", "python"),
    ]

    for alg_label, alg_type in algorithms:
        print(f"\n---> Evaluando: {alg_label}")

        # Diccionario para guardar el tiempo base T(1, 1024) por algoritmo para el cálculo de eficiencia débil
        base_time_map = {}
        if alg_label in base_times:
            base_time_map[1024] = base_times[alg_label]

        for N in N_values:
            for p in p_values:
                if (alg_label, N, p) in completed_configs:
                    print(f"   N={N}, p={p} ya existe en el CSV; se omite.")
                    continue

                times = []
                checksums = []

                print(
                    f"   Ejecutando N={N}, p={p} ({runs_per_config} repeticiones)...",
                    end="",
                    flush=True,
                )

                for _ in range(runs_per_config):
                    if alg_type == "cpp1":
                        t, chk = run_cpp_program("algoritmo1", N, None, p)
                    elif alg_type == "cpp2":
                        t, chk = run_cpp_program("algoritmo2", N, n_optimal, p)
                    elif alg_type == "python":
                        t, chk = run_python_program(N, p)

                    times.append(t)
                    checksums.append(chk)
                    time.sleep(0.2)  # Pequeña pausa para estabilizar la CPU

                avg_t = np.mean(times)
                std_t = np.std(times)

                # Verificar consistencia de checksums entre las 3 corridas
                chk_consistent = np.allclose(checksums, checksums[0], atol=1e-4)

                # Guardar el tiempo base T(p=1, N=1024)
                if p == 1 and N == 1024:
                    base_time_map[1024] = avg_t

                # Calcular la Eficiencia Débil respecto al caso base proporcional:
                # La carga relativa por hilo escala con N^3 / p.
                # Para la pareja estándar (p=1, N=1024) vs (p=8, N=2048), la carga por hilo es idéntica (1024^3/1 == 2048^3/8).
                if 1024 in base_time_map:
                    # Relación teórica de carga de trabajo por hilo
                    work_ratio = ((N / 1024) ** 3) / p
                    # Eficiencia débil ajustada a la carga por hilo
                    weak_efficiency = (
                        base_time_map[1024] * work_ratio
                    ) / avg_t
                else:
                    weak_efficiency = 1.0

                print(
                    f" Promedio: {avg_t:.2f} ms | Efic. Débil: {weak_efficiency:.2%}"
                )

                # Escribir cada configuración inmediatamente para conservar el progreso.
                with open(csv_filename, mode="a", newline="", encoding="utf-8") as f:
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
                        f"{weak_efficiency:.4f}",
                        "YES" if chk_consistent else "NO",
                    ])
                    f.flush()
                    os.fsync(f.fileno())
                completed_configs.add((alg_label, N, p))

    print(f"\n[✓] ¡Experimentos completados exitosamente!")
    print(f"[✓] Resultados exportados a: {csv_filename}")


if __name__ == "__main__":
    main()
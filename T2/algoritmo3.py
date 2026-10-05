# algoritmo3.py - Multiplicación de matrices con Numba (Paralelismo de datos).
# Uso: python3 algoritmo3.py [N p]   (N por defecto 2048, p por defecto 1 thread)

import sys
import time
import numpy as np
from numba import njit, prange, set_num_threads


# Generador determinista de valores en [-1, 1] (reproducible)
@njit
def det_value(i, j, offset):
    return np.sin(i * 0.1 + j * 0.3 + offset)


# Inicialización de matriz
@njit
def init_matrix(N, offset):
    M = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            M[i, j] = det_value(i, j, offset)
    return M


# Algoritmo 1 implementado en Numba con paralelismo de datos
@njit(parallel=True, fastmath=True)
def multiply_data_parallel_numba(A, B, C, N):
    # prange paraleliza automáticamente el loop exterior 'i'
    for i in prange(N):
        for k in range(N):
            a = A[i, k]
            for j in range(N):
                C[i, j] += a * B[k, j]


def main():
    # 1. Leer parámetros desde la terminal
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    # 2. Configurar el número de hilos en Numba
    set_num_threads(p)

    # 3. Inicializar matrices
    A = init_matrix(N, 1.0)
    B = init_matrix(N, 2.0)
    C = np.zeros((N, N), dtype=np.float64)

    # 4. Calentamiento (Warm-up): Fuerza la compilación JIT
    # Usamos una matriz pequeña fija para compilar rápido antes de medir
    A_dummy = np.ones((16, 16), dtype=np.float64)
    B_dummy = np.ones((16, 16), dtype=np.float64)
    C_dummy = np.zeros((16, 16), dtype=np.float64)
    multiply_data_parallel_numba(A_dummy, B_dummy, C_dummy, 16)

    # Reset de C antes de la medición real
    C.fill(0.0)

    # 5. Medición del tiempo de ejecución (solo cómputo puro)
    t0 = time.perf_counter()
    multiply_data_parallel_numba(A, B, C, N)
    t1 = time.perf_counter()

    ms = (t1 - t0) * 1000.0

    # 6. Checksum para verificación
    checksum = np.sum(C)

    print(f"N={N} threads={p} tiempo={ms:.2f} ms checksum={checksum:.6f}")


if __name__ == "__main__":
    main()
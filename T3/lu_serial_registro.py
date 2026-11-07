# Version con registro automatico basada en lu_serial.py (IIC3533 - Tarea 3).
# Mantiene las operaciones y la temporizacion original del algoritmo.
# Ejemplo (apartado a): python lu_serial_registro.py --todos --repeticiones 3

import argparse
import csv
import datetime
import platform
import time
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, solve_triangular


def matrix_entries(rows, cols, N):
    """Entradas A[rows, cols] definidas por sus indices globales."""
    I = np.asarray(rows, dtype=np.float64)[:, None]
    J = np.asarray(cols, dtype=np.float64)[None, :]
    A = np.sin(1e-3 * I * J + 0.3 * I + 0.7 * J)
    A[np.asarray(rows)[:, None] == np.asarray(cols)[None, :]] = N
    return A


def lu_diag_block(Akk):
    """LU del bloque diagonal, guardada in-place (sin pivoteo)."""
    lu, piv = lu_factor(Akk, overwrite_a=True, check_finite=False)
    assert np.all(piv == np.arange(len(piv))), "Hay pivoteo."
    Akk[:] = lu


def lu_blocked(A, b):
    """Factorizacion LU por bloques (right-looking), in-place."""
    N = A.shape[0]
    for k in range(0, N, b):
        e = min(k + b, N)
        lu_diag_block(A[k:e, k:e])
        Lkk = A[k:e, k:e]
        A[k:e, e:] = solve_triangular(
            Lkk, A[k:e, e:], lower=True, unit_diagonal=True, check_finite=False
        )
        A[e:, k:e] = solve_triangular(
            Lkk, A[e:, k:e].T, trans="T", lower=False, check_finite=False
        ).T
        A[e:, e:] -= A[e:, k:e] @ A[k:e, e:]


def residual(A, LU):
    """Error relativo ||A - LU||_F / ||A||_F."""
    L = np.tril(LU, -1) + np.eye(LU.shape[0])
    U = np.triu(LU)
    return np.linalg.norm(A - L @ U) / np.linalg.norm(A)


def ejecutar(N, b, check):
    idx = np.arange(N)
    A = matrix_entries(idx, idx, N)
    A0 = A.copy() if check else None
    t0 = time.perf_counter()
    lu_blocked(A, b)
    t1 = time.perf_counter()
    tiempo = t1 - t0
    gflops = (2 * N**3 / 3) / tiempo / 1e9
    checksum = A.sum()
    residuo = residual(A0, A) if check else None
    return tiempo, gflops, checksum, residuo


def main():
    parser = argparse.ArgumentParser(
        description="Factorizacion LU serial con registro automatico en TXT tabulado."
    )
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--b", type=int, default=256)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--todos", action="store_true",
                        help="probar b = 4096, 2048, 1024, 512, 256, 128, 64 (los que quepan en N)")
    parser.add_argument("--repeticiones", type=int, default=1,
                        help="veces que se ejecuta cada b (por defecto 1)")
    parser.add_argument("--salida", default="resultados_lu_serial.txt",
                        help="ruta del TXT con resultados (por defecto junto al script)")
    args = parser.parse_args()
    if args.N <= 0 or args.b <= 0 or args.repeticiones <= 0:
        parser.error("N, b y repeticiones deben ser positivos")

    tamanos = ([b for b in (4096, 2048, 1024, 512, 256, 128, 64) if b <= args.N]
               if args.todos else [args.b])
    if not tamanos:
        parser.error("N debe ser al menos 64 para usar --todos")

    salida = Path(args.salida)
    if not salida.is_absolute():
        salida = Path(__file__).resolve().parent / salida
    salida.parent.mkdir(parents=True, exist_ok=True)
    necesita_cabecera = not salida.exists() or salida.stat().st_size == 0

    columnas = ["fecha", "equipo", "N", "b", "repeticion", "tiempo_s",
                "gflops", "checksum", "residuo_relativo"]
    with salida.open("a", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo, delimiter="\t")
        if necesita_cabecera:
            escritor.writerow(columnas)
            archivo.flush()
        for numero in range(1, args.repeticiones + 1):
            for b in tamanos:
                tiempo, gflops, checksum, residuo = ejecutar(args.N, b, args.check)
                escritor.writerow([
                    datetime.datetime.now().isoformat(timespec="seconds"),
                    platform.node(), args.N, b, numero,
                    f"{tiempo:.6f}", f"{gflops:.4f}",
                    f"{checksum:.6e}",
                    f"{residuo:.6e}" if residuo is not None else "",
                ])
                archivo.flush()  # Se guarda cada corrida incluso si las siguientes fallan.
                print(f"N={args.N} b={b} rep={numero}/{args.repeticiones} "
                      f"tiempo={tiempo:.6f} s GFLOP/s={gflops:.4f} "
                      f"checksum={checksum:.6e}", flush=True)
                if residuo is not None:
                    print(f"  residuo relativo={residuo:.3e}", flush=True)
    print(f"\nResultados guardados en: {salida}")
    print("Formato: columnas separadas por tabuladores; se puede abrir en Excel.")


if __name__ == "__main__":
    main()

"""LU distribuida bloque-ciclica 2D con mpi4py.

Uso:
  mpirun -np 4 python lu_distributed.py --N 128 --b 16 --pr 2 --pc 2 --check
  mpirun -np 4 python lu_distributed.py --N 4096 --b 512 --pr 2 --pc 2 --output resultados_lu_distributed.csv

Los bloques estan ordenados por indice global y almacenados en un unico ndarray
local por proceso. En un paso k, el panel activo es un subarreglo contiguo.
"""

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from mpi4py import MPI
from scipy.linalg import lu_factor, solve_triangular

# Misma matriz determinista de lu_serial.py.
from lu_serial import matrix_entries, lu_blocked, residual


def indice_primero_mayor(indices_bloque, k):
    """Posicion local del primer bloque cuyo indice global es mayor que k."""
    return int(np.searchsorted(indices_bloque, k + 1))


def construir_matriz_local(N, b, myrow, mycol, pr, pc):
    """Devuelve matriz local contigua e indices globales de bloques asignados."""
    nb = N // b
    filas = np.arange(myrow, nb, pr, dtype=np.int64)
    columnas = np.arange(mycol, nb, pc, dtype=np.int64)
    A = np.empty((len(filas) * b, len(columnas) * b), dtype=np.float64)
    indices_col = (columnas[:, None] * b + np.arange(b)[None, :]).ravel()

    # Generacion por filas de bloques: evita materializar la matriz global.
    for i_local, i_global in enumerate(filas):
        indices_fila = i_global * b + np.arange(b)
        A[i_local * b:(i_local + 1) * b, :] = matrix_entries(
            indices_fila, indices_col, N
        )
    return A, filas, columnas


def factorizar(A, N, b, filas, columnas, pr, pc, myrow, mycol,
               row_comm, col_comm):
    """Algoritmo de cuatro etapas: diagonal, paneles, difusion, actualizacion."""
    # Desglose: diag/panel/update (computo) y bcast_diag/bcast_panel (comunicacion).
    det = dict(diag=0.0, panel=0.0, update=0.0, bcast_diag=0.0, bcast_panel=0.0,
               bytes_bcast=0)
    nb = N // b

    for k in range(nb):
        r, c = k % pr, k % pc
        # Indices locales del bloque k, cuando este rank posee su fila/columna.
        ik = (k - myrow) // pr if myrow == r else None
        jk = (k - mycol) // pc if mycol == c else None
        i0 = indice_primero_mayor(filas, k) * b
        j0 = indice_primero_mayor(columnas, k) * b

        # 1) Factorizar bloque diagonal en el proceso (r,c).
        diag = np.empty((b, b), dtype=np.float64)
        if myrow == r and mycol == c:
            t0 = MPI.Wtime()
            akk = A[ik*b:(ik+1)*b, jk*b:(jk+1)*b]
            lu, piv = lu_factor(akk, overwrite_a=True, check_finite=False)
            if not np.array_equal(piv, np.arange(b)):
                raise RuntimeError(f"Se produjo pivoteo en el bloque {k}")
            akk[:] = lu
            diag[:] = lu
            det["diag"] += MPI.Wtime() - t0

        # 2) Comunicar bloque LU diagonal a fila r y columna c.
        # row_comm tiene rank local = mycol; col_comm tiene rank local = myrow.
        if myrow == r:
            t0 = MPI.Wtime()
            row_comm.Bcast(diag, root=c)
            det["bcast_diag"] += MPI.Wtime() - t0
            det["bytes_bcast"] += diag.nbytes
        if mycol == c:
            t0 = MPI.Wtime()
            col_comm.Bcast(diag, root=r)
            det["bcast_diag"] += MPI.Wtime() - t0
            det["bytes_bcast"] += diag.nbytes

        # 3) Calcular paneles de U y L con solve_triangular.
        # Todos los ranks disponen de arreglos del tamano requerido por su fila/col.
        L_panel = np.empty((A.shape[0] - i0, b), dtype=np.float64)
        U_panel = np.empty((b, A.shape[1] - j0), dtype=np.float64)

        if myrow == r and U_panel.shape[1] > 0:
            t0 = MPI.Wtime()
            panel = A[ik*b:(ik+1)*b, j0:]
            panel[:] = solve_triangular(
                diag, panel, lower=True, unit_diagonal=True, check_finite=False
            )
            U_panel[:] = panel
            det["panel"] += MPI.Wtime() - t0

        if mycol == c and L_panel.shape[0] > 0:
            t0 = MPI.Wtime()
            panel = A[i0:, jk*b:(jk+1)*b]
            panel[:] = solve_triangular(
                diag, panel.T, trans="T", lower=False, check_finite=False
            ).T
            L_panel[:] = panel
            det["panel"] += MPI.Wtime() - t0

        # 4) Difundir panel L por cada fila; panel U por cada columna.
        # Las dimensiones son iguales en los ranks de cada comunicador.
        if L_panel.size:
            t0 = MPI.Wtime()
            row_comm.Bcast(L_panel, root=c)
            det["bcast_panel"] += MPI.Wtime() - t0
            det["bytes_bcast"] += L_panel.nbytes
        if U_panel.size:
            t0 = MPI.Wtime()
            col_comm.Bcast(U_panel, root=r)
            det["bcast_panel"] += MPI.Wtime() - t0
            det["bytes_bcast"] += U_panel.nbytes

        # 5) Actualizar toda la submatriz local activa con un unico producto.
        if L_panel.shape[0] > 0 and U_panel.shape[1] > 0:
            t0 = MPI.Wtime()
            A[i0:, j0:] -= L_panel @ U_panel
            det["update"] += MPI.Wtime() - t0

    t_compute = det["diag"] + det["panel"] + det["update"]
    t_communication = det["bcast_diag"] + det["bcast_panel"]
    return t_compute, t_communication, det


def verificar(N, b, A, filas, columnas, comm):
    """Reconstruye LU distribuida solo en modo de prueba y compara con LU serial."""
    bloques = comm.gather((filas, columnas, A), root=0)
    if comm.rank != 0:
        return None

    LU = np.empty((N, N), dtype=np.float64)
    for f, c, local in bloques:
        for i, fi in enumerate(f):
            for j, cj in enumerate(c):
                LU[fi*b:(fi+1)*b, cj*b:(cj+1)*b] = local[
                    i*b:(i+1)*b, j*b:(j+1)*b
                ]

    idx = np.arange(N)
    original = matrix_entries(idx, idx, N)
    serial = original.copy()
    lu_blocked(serial, b)
    err_serial = np.linalg.norm(LU - serial) / np.linalg.norm(serial)
    err_res = residual(original, LU)
    return err_serial, err_res


def guardar_resultados(destino, filas):
    ruta = Path(destino)
    campos = list(filas[0].keys())
    existe = ruta.exists() and ruta.stat().st_size > 0
    with ruta.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        if not existe:
            writer.writeheader()
        writer.writerows(filas)


def main():
    parser = argparse.ArgumentParser(description="LU distribuida bloque-ciclica con mpi4py")
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--b", type=int, default=16)
    parser.add_argument("--pr", type=int, default=2)
    parser.add_argument("--pc", type=int, default=2)
    parser.add_argument("--check", action="store_true", help="compara con LU serial (N<=512)")
    parser.add_argument("--output", type=str, default=None, help="CSV de tiempos por proceso")
    parser.add_argument("--parte", type=str, default="", help="etiqueta (c, d, e, f...) para el CSV")
    parser.add_argument("--config", type=str, default="", help="configuracion de la tabla (0-5)")
    parser.add_argument("--rep", type=int, default=0, help="numero de repeticion")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    if (args.N <= 0 or args.b <= 0 or args.pr <= 0 or args.pc <= 0 or
            args.N % args.b != 0 or args.pr * args.pc != size or
            (args.check and args.N > 512)):
        if rank == 0:
            parser.error("Requiere N,b,pr,pc positivos; N divisible por b; "
                         "pr*pc=numero de procesos; --check solo con N<=512")
        comm.Abort(2)

    N, b, pr, pc = args.N, args.b, args.pr, args.pc
    myrow, mycol = rank // pc, rank % pc
    row_comm = comm.Split(color=myrow, key=mycol)
    col_comm = comm.Split(color=mycol, key=myrow)

    A, filas, columnas = construir_matriz_local(N, b, myrow, mycol, pr, pc)
    # Sincronizar fuera del intervalo cronometrado.
    comm.Barrier()
    t0 = MPI.Wtime()
    t_comp, t_comm, det = factorizar(
        A, N, b, filas, columnas, pr, pc, myrow, mycol, row_comm, col_comm
    )
    t_total = MPI.Wtime() - t0

    datos = comm.gather((rank, myrow, mycol, t_comp, t_comm, t_total, det,
                         MPI.Get_processor_name()), root=0)
    errors = verificar(N, b, A, filas, columnas, comm) if args.check else None

    if rank == 0:
        global_t = max(t[5] for t in datos)
        print(f"N={N} b={b} grid={pr}x{pc} p={size} tiempo_global={global_t:.6f} s")
        print("rank (fila,col)   computo(s)    comunicacion(s)    total(s)")
        for rank_i, r_i, c_i, comp_i, comm_i, total_i, _, host_i in datos:
            print(f"{rank_i:4d} ({r_i},{c_i})   {comp_i:12.6f}   {comm_i:14.6f}   {total_i:10.6f}   {host_i}")
        if args.check:
            err_serial, err_res = errors
            print(f"error_relativo_vs_serial={err_serial:.3e} residuo_relativo={err_res:.3e}")
            if not np.isfinite(err_serial) or not np.isfinite(err_res) or err_serial > 1e-10 or err_res > 1e-10:
                raise SystemExit("ERROR: factorizacion distribuida no paso la prueba")
            print("CHECK: OK")
        if args.output:
            timestamp = datetime.now().isoformat(timespec="seconds")
            hilos = os.environ.get("OMP_NUM_THREADS", "")
            nodos = len({d[7] for d in datos})
            registros = [dict(
                fecha=timestamp, N=N, b=b, pr=pr, pc=pc, procesos=size,
                rank=rk, fila=rr, columna=cc,
                computo_s=f"{tc:.9f}", comunicacion_s=f"{tm:.9f}",
                total_s=f"{tt:.9f}", tiempo_global_s=f"{global_t:.9f}",
                # Columnas agregadas (no alteran las anteriores).
                T=hilos, nodos=nodos, host=host, parte=args.parte,
                config=args.config, rep=args.rep,
                diag_s=f"{d['diag']:.9f}", panel_s=f"{d['panel']:.9f}",
                update_s=f"{d['update']:.9f}",
                bcast_diag_s=f"{d['bcast_diag']:.9f}",
                bcast_panel_s=f"{d['bcast_panel']:.9f}",
                bytes_bcast=d["bytes_bcast"],
                jobid=os.environ.get("SLURM_JOB_ID", ""),
            ) for rk, rr, cc, tc, tm, tt, d, host in datos]
            guardar_resultados(args.output, registros)
            print(f"Resultados guardados en {args.output}")

    row_comm.Free()
    col_comm.Free()


if __name__ == "__main__":
    main()

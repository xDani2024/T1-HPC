"""Genera un .sbatch por cada (parte, configuracion, grid, T) de la Tarea 3.

Uso: python slurm/generar_sbatch.py   (desde la carpeta T3)
Los archivos quedan en slurm/jobs/ y se entregan en el .zip.
"""
from pathlib import Path

# (parte, config, pr, pc, T, nodos, procesos_por_nodo)
JOBS = [("c", 2, 2, 2, 2, 1, 4)]
JOBS += [("d", 0, 1, 1, T, 1, 1) for T in (1, 2, 4, 8)]
JOBS += [("d", 1, pr, pc, T, 1, 2) for (pr, pc) in ((1, 2), (2, 1)) for T in (1, 2, 4)]
JOBS += [("d", 2, 2, 2, T, 1, 4) for T in (1, 2)]
JOBS += [("d", 3, pr, pc, 1, 1, 8) for (pr, pc) in ((2, 4), (4, 2))]
JOBS += [("e", 4, 2, 4, 1, 2, 4), ("e", 5, 4, 4, 1, 2, 8)]
JOBS += [("f", 2, 2, 2, 1, 1, 4)]   # (f) con T=1; T=2 ya sale de la parte (c)

PLANTILLA = """#!/bin/bash
#SBATCH --job-name=lu_{nombre}
#SBATCH --qos=hpc-iic3533
#SBATCH --partition=hpc-iic3533
#SBATCH --account=default-account
#SBATCH --comment="IIC3533 Tarea 3: LU distribuida MPI, parte {parte}, config {config}, grid {pr}x{pc}, T={T}"
#SBATCH --nodes={nodos}
#SBATCH --ntasks={p}
#SBATCH --ntasks-per-node={ppn}
#SBATCH --cpus-per-task={T}
#SBATCH --mem=3500M
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
# Parte ({parte}) - configuracion {config}: grid {pr}x{pc}, T={T}, {nodos} nodo(s), {ppn} proc/nodo, {cpus} CPUs.
# Los valores de b se pasan con: sbatch --export=ALL,B_LIST="512 256" este_archivo
PARTE={parte}; CONFIG={config}; PR={pr}; PC={pc}
{extra}source "$SLURM_SUBMIT_DIR/slurm/comun.sh"
"""


def nombre_job(parte, config, pr, pc, T, nodos):
    return f"{parte}_cfg{config}_{pr}x{pc}_T{T}_n{nodos}"


def main():
    out = Path(__file__).resolve().parent / "jobs"
    out.mkdir(exist_ok=True)
    for parte, config, pr, pc, T, nodos, ppn in JOBS:
        nombre = nombre_job(parte, config, pr, pc, T, nodos)
        extra = 'B_LIST=${B_LIST:-"4096 2048 1024 512 256 128 64"}\n' if parte == "c" else ""
        extra += 'B_LIST=${B_LIST:-"4096 512"}  # b = N/2 y b = N/16\n' if parte == "f" else ""
        txt = PLANTILLA.format(nombre=nombre, parte=parte, config=config, pr=pr, pc=pc,
                               T=T, nodos=nodos, p=pr * pc, ppn=ppn, cpus=pr * pc * T,
                               extra=extra)
        (out / f"{nombre}.sbatch").write_text(txt)
        print(out / f"{nombre}.sbatch")


if __name__ == "__main__":
    main()

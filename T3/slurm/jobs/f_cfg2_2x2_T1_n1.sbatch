#!/bin/bash
#SBATCH --job-name=lu_f_cfg2_2x2_T1_n1
#SBATCH --qos=hpc-iic3533
#SBATCH --partition=hpc-iic3533
#SBATCH --account=default-account
#SBATCH --comment="IIC3533 Tarea 3: LU distribuida MPI, parte f, config 2, grid 2x2, T=1"
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=3500M
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
# Parte (f) - configuracion 2: grid 2x2, T=1, 1 nodo(s), 4 proc/nodo, 4 CPUs.
# Los valores de b se pasan con: sbatch --export=ALL,B_LIST="512 256" este_archivo
PARTE=f; CONFIG=2; PR=2; PC=2
B_LIST=${B_LIST:-"4096 512"}  # b = N/2 y b = N/16
source "$SLURM_SUBMIT_DIR/slurm/comun.sh"

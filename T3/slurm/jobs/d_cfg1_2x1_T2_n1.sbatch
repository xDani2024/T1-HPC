#!/bin/bash
#SBATCH --job-name=lu_d_cfg1_2x1_T2_n1
#SBATCH --qos=hpc-iic3533
#SBATCH --partition=hpc-iic3533
#SBATCH --account=default-account
#SBATCH --comment="IIC3533 Tarea 3: LU distribuida MPI, parte d, config 1, grid 2x1, T=2"
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=3500M
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
# Parte (d) - configuracion 1: grid 2x1, T=2, 1 nodo(s), 2 proc/nodo, 4 CPUs.
# Los valores de b se pasan con: sbatch --export=ALL,B_LIST="512 256" este_archivo
PARTE=d; CONFIG=1; PR=2; PC=1
source "$SLURM_SUBMIT_DIR/slurm/comun.sh"

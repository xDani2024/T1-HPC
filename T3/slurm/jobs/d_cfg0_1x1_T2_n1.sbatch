#!/bin/bash
#SBATCH --job-name=lu_d_cfg0_1x1_T2_n1
#SBATCH --qos=hpc-iic3533
#SBATCH --partition=hpc-iic3533
#SBATCH --account=default-account
#SBATCH --comment="IIC3533 Tarea 3: LU distribuida MPI, parte d, config 0, grid 1x1, T=2"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3500M
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
# Parte (d) - configuracion 0: grid 1x1, T=2, 1 nodo(s), 1 proc/nodo, 2 CPUs.
# Los valores de b se pasan con: sbatch --export=ALL,B_LIST="512 256" este_archivo
PARTE=d; CONFIG=0; PR=1; PC=1
source "$SLURM_SUBMIT_DIR/slurm/comun.sh"

#!/bin/bash
#SBATCH --job-name=lu_d_cfg3_4x2_T1_n1
#SBATCH --qos=hpc-iic3533
#SBATCH --partition=hpc-iic3533
#SBATCH --account=default-account
#SBATCH --comment="IIC3533 Tarea 3: LU distribuida MPI, parte d, config 3, grid 4x2, T=1"
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=3500M
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
# Parte (d) - configuracion 3: grid 4x2, T=1, 1 nodo(s), 8 proc/nodo, 8 CPUs.
# Los valores de b se pasan con: sbatch --export=ALL,B_LIST="512 256" este_archivo
PARTE=d; CONFIG=3; PR=4; PC=2
source "$SLURM_SUBMIT_DIR/slurm/comun.sh"

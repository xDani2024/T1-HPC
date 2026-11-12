#!/bin/bash
#SBATCH --job-name=lu_e_cfg4_2x4_T1_n2
#SBATCH --qos=hpc-iic3533
#SBATCH --partition=hpc-iic3533
#SBATCH --account=default-account
#SBATCH --comment="IIC3533 Tarea 3: LU distribuida MPI, parte e, config 4, grid 2x4, T=1"
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=3500M
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
# Parte (e) - configuracion 4: grid 2x4, T=1, 2 nodo(s), 4 proc/nodo, 8 CPUs.
# Los valores de b se pasan con: sbatch --export=ALL,B_LIST="512 256" este_archivo
PARTE=e; CONFIG=4; PR=2; PC=4
source "$SLURM_SUBMIT_DIR/slurm/comun.sh"

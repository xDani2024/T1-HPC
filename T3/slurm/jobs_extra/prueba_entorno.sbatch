#!/bin/bash
#SBATCH --job-name=lu_prueba_entorno
#SBATCH --qos=hpc-iic3533
#SBATCH --partition=hpc-iic3533
#SBATCH --account=default-account
#SBATCH --comment="IIC3533 Tarea 3: prueba de entorno y correctitud de LU distribuida MPI"
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=3500M
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
# Valida entorno + correctitud en MPI real (evidencia para la parte b):
# compara la LU distribuida con la serial en todos los grids usados, en 1 y 2 nodos.
cd "$SLURM_SUBMIT_DIR"
CONDA_BASE=${CONDA_BASE:-$HOME/miniconda3}; CONDA_ENV=${CONDA_ENV:-tarea3lu}
source "$CONDA_BASE/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
MPI_EXTRA=${MPI_EXTRA:-}
mkdir -p resultados_cluster

echo "== nodos: $SLURM_JOB_NODELIST"
mpirun --version | head -1
python -c 'from mpi4py import MPI; print(MPI.Get_library_version().splitlines()[0])'
echo "== hostnames por rank (16 procesos, deben repartirse 8+8):"
mpirun -n 16 --bind-to none $MPI_EXTRA python -c \
  'from mpi4py import MPI; c=MPI.COMM_WORLD; print(c.rank, MPI.Get_processor_name(), flush=True)' | sort -n

for cfg in "1 1 1" "2 1 2" "2 2 1" "4 2 2" "8 2 4" "8 4 2" "16 4 4"; do
  set -- $cfg
  for Nb in "512 64" "512 32" "256 128"; do
    set -- $1 $2 $3 $Nb
    echo "--- CHECK p=$1 grid=$2x$3 N=$4 b=$5"
    mpirun -n $1 --bind-to none $MPI_EXTRA python lu_distributed.py \
      --N $4 --b $5 --pr $2 --pc $3 --check \
      --output resultados_cluster/check_correctitud.csv --parte check
  done
done
echo "== fin prueba"

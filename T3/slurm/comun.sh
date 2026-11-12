#!/bin/bash
# Cuerpo comun de todos los jobs de la Tarea 3. Lo cargan los .sbatch con `source`.
# Variables que define cada .sbatch antes de cargarlo: PARTE, CONFIG, PR, PC.
# Variables opcionales (se pueden pasar con sbatch --export=ALL,VAR=...):
#   B_LIST   lista de b separados por espacio   (por defecto: "512")
#   N        tamano de la matriz                (por defecto: 8192 = 2^13)
#   REPS     repeticiones por b                 (por defecto: 3)
#   CONDA_BASE, CONDA_ENV   instalacion y entorno conda (por defecto ~/miniconda3, tarea3lu)
#   MPI_EXTRA  flags extra para mpirun (p.ej. "--mca btl_tcp_if_include eno1")
set -uo pipefail

N=${N:-8192}
REPS=${REPS:-3}
B_LIST=${B_LIST:-512}
CONDA_BASE=${CONDA_BASE:-$HOME/miniconda3}
CONDA_ENV=${CONDA_ENV:-tarea3lu}
MPI_EXTRA=${MPI_EXTRA:-}
LIMITE_S=${LIMITE_S:-780}   # no lanzar corridas nuevas despues de 13 min (el job muere a los 15)

cd "${SLURM_SUBMIT_DIR:-.}"   # carpeta T3 (donde estan lu_distributed.py y lu_serial.py)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Threads de BLAS = CPUs por proceso (evita oversubscription).
T=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T

mkdir -p resultados_cluster
SALIDA="resultados_cluster/${PARTE}_cfg${CONFIG}_${PR}x${PC}_T${T}_n${SLURM_JOB_NUM_NODES:-1}_${SLURM_JOB_ID:-local}.csv"

echo "== job ${SLURM_JOB_ID:-} parte=$PARTE config=$CONFIG grid=${PR}x${PC} T=$T"
echo "   nodos=${SLURM_JOB_NODELIST:-?} ntasks=${SLURM_NTASKS:-?} tasks/nodo=${SLURM_TASKS_PER_NODE:-?}"
echo "   N=$N b=[$B_LIST] reps=$REPS  -> $SALIDA"
python -c 'import numpy; numpy.show_config()' 2>/dev/null | grep -iE "openblas|mkl" | head -2

# --bind-to none: cada proceso usa sus T threads dentro de las CPUs asignadas por SLURM
# (con el binding por defecto de OpenMPI los T threads quedarian en un solo core).
for rep in $(seq 1 "$REPS"); do
  for b in $B_LIST; do
    if (( N % b != 0 )); then echo "b=$b no divide N=$N, se omite"; continue; fi
    if (( SECONDS > LIMITE_S )); then echo "Limite de tiempo cercano: se detiene en rep=$rep b=$b"; break 2; fi
    echo "--- rep=$rep b=$b (t=${SECONDS}s)"
    mpirun -n "$SLURM_NTASKS" --bind-to none $MPI_EXTRA \
      python lu_distributed.py --N "$N" --b "$b" --pr "$PR" --pc "$PC" \
      --output "$SALIDA" --parte "$PARTE" --config "$CONFIG" --rep "$rep"
  done
done
echo "== fin job (${SECONDS}s)"

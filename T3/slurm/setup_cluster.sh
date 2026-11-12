#!/bin/bash
# Prepara Miniconda + entorno en el cluster (correr UNA vez en el nodo de login: bash slurm/setup_cluster.sh)
set -euo pipefail
CONDA_BASE=${CONDA_BASE:-$HOME/miniconda3}
CONDA_ENV=${CONDA_ENV:-tarea3lu}

if [ ! -x "$CONDA_BASE/bin/conda" ]; then
  echo "Instalando Miniconda en $CONDA_BASE ..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_BASE"
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^$CONDA_ENV\s"; then
  conda create -y -n "$CONDA_ENV" -c conda-forge --override-channels \
    python=3.11 numpy scipy mpi4py openmpi
fi
conda activate "$CONDA_ENV"
python -c 'import numpy, scipy; from mpi4py import MPI; print("OK numpy", numpy.__version__, "scipy", scipy.__version__); print(MPI.Get_library_version().splitlines()[0])'
mpirun --version | head -1

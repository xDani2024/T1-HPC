#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Un hilo BLAS por proceso MPI: evita oversubscription en las pruebas locales.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if command -v mpirun >/dev/null 2>&1; then
    MPI=mpirun
elif command -v mpiexec >/dev/null 2>&1; then
    MPI=mpiexec
else
    echo "No se encontro mpirun/mpiexec. Instale MPI en su entorno Conda." >&2
    exit 1
fi

python -c 'import numpy, scipy; from mpi4py import MPI; print("Dependencias Python OK; MPI:", MPI.Get_library_version().splitlines()[0])'

echo '--- Prueba 1: 1x1 ---'
"$MPI" -n 1 python lu_distributed.py --N 128 --b 32 --pr 1 --pc 1 --check

echo '--- Prueba 2: 1x2 ---'
"$MPI" -n 2 python lu_distributed.py --N 128 --b 32 --pr 1 --pc 2 --check

echo '--- Prueba 3: 2x1 ---'
"$MPI" -n 2 python lu_distributed.py --N 128 --b 32 --pr 2 --pc 1 --check

echo '--- Prueba 4: 2x2 ---'
"$MPI" -n 4 python lu_distributed.py --N 256 --b 64 --pr 2 --pc 2 --check --output resultados_b_local.csv

echo 'Pruebas terminadas. Revise los mensajes CHECK: OK y resultados_b_local.csv.'

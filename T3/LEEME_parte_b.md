# TAREA 3 · IIC3533 · APARTADO (b) · Pruebas locales en Ubuntu/WSL

## ARCHIVOS
- lu_serial.py: referencia original del curso, necesaria para la matriz y la comprobación.
- lu_distributed.py: LU por bloques en grid MPI 2D, distribución bloque-cíclica.
- probar_lu_local.sh: ejecuta cuatro pruebas de correctitud con 1, 2 y 4 procesos.

### 1. DESCOMPRIMIR EN UNA MISMA CARPETA
   El archivo lu_distributed.py importa funciones de lu_serial.py. Ambos deben
   permanecer en la misma carpeta al ejecutar el programa.

### 2. PREPARAR ENTORNO CONDA (opcion recomendada para mantener MPI y mpi4py compatibles)
   conda create -n tarea3lu -c conda-forge python=3.11 numpy scipy mpi4py openmpi -y
   conda activate tarea3lu
   python -c 'from mpi4py import MPI; import scipy; print(MPI.Get_library_version())'
   mpirun --version

### 3. PROBAR TODO EN WSL
   cd /ruta/a/la/carpeta
   bash probar_lu_local.sh

### 4. EJECUTAR UNA PRUEBA MANUAL
   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
   mpirun -n 4 python lu_distributed.py --N 256 --b 64 --pr 2 --pc 2 --check --output resultados_b_local.csv

### 5. EJECUTAR SIN CHECK Y REGISTRAR RENDIMIENTO LOCAL
   mpirun -n 4 python lu_distributed.py --N 1024 --b 128 --pr 2 --pc 2 --output resultados_b_local.csv

## SALIDA
- Cada rank reporta tiempo de cómputo, comunicación y tiempo local total.
- tiempo_global = max(tiempo local de cada rank) para esa ejecucion.
- --output guarda un CSV con una fila por proceso, agregando nuevas corridas.
- --check compara con la LU serial y calcula el residuo relativo (solo N <= 512).
- No use --check para obtener las mediciones finales de rendimiento.
- Repetir 3 veces en la misma configuración para una mediana más estable.

## COMUNICACIONES (k=indice de bloque, r=k%pr, c=k%pc)
- Factor diagonal: Bcast en row_comm de fila r, root=c.
- Factor diagonal: Bcast en col_comm de columna c, root=r.
- Panel L: Bcast en cada row_comm, root=c.
- Panel U: Bcast en cada col_comm, root=r.
- La submatriz local se actualiza con A[i0:,j0:] -= L_panel @ U_panel.

## IMPORTANTE
Este código fue contrastado con el algoritmo serial usando un entorno de prueba
que simula las comunicaciones en distintas topologías. Las pruebas reales con
OpenMPI/mpi4py se deben ejecutar y documentar en Ubuntu/WSL. Las mediciones
obtenidas por una simulacion no son representativas del rendimiento de MPI.

Declarar el uso de IA en el informe, según las instrucciones del curso.

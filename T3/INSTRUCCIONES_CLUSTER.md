# Tarea 3 – Cómo correr todo en el clúster (partes c, d, e, f)

Todo queda automatizado: un solo comando envía los 20 jobs en orden, respetando el
límite del curso (máx. 4 en cola, 1 ejecutándose), y elige `b_c` solo a partir de los
resultados de (c).

## 1. Subir la carpeta T3 y preparar el entorno (una vez)

```bash
# en tu computador
scp -r T3 kraken:~/            # o: ssh kraken && git clone https://github.com/xDani2024/T1-HPC.git
ssh kraken
cd ~/T3                        # (o ~/T1-HPC/T3)
bash slurm/setup_cluster.sh    # instala Miniconda (si falta) + entorno tarea3lu (numpy, scipy, mpi4py, openmpi)
```

Si ya tienen conda en otra ruta o con otro nombre de entorno:
`export CONDA_BASE=/ruta/miniconda3 CONDA_ENV=mi_env` antes de los pasos siguientes.

## 2. Lanzar todo (≈ 1–1,5 h en total, sin supervisión)

```bash
mkdir -p logs
tmux new -s t3          # o screen; si se cae el ssh, el orquestador sigue corriendo
bash slurm/enviar_todo.sh 2>&1 | tee logs/orquestador.log
# (sin tmux:  nohup bash slurm/enviar_todo.sh > logs/orquestador.log 2>&1 & )
```

Orden que sigue el orquestador:

| Etapa | Job(s) | Qué hace |
|---|---|---|
| prueba | `jobs_extra/prueba_entorno.sbatch` (2 nodos, 16 procesos) | Verifica MPI en 1 y 2 nodos y corre `--check` vs LU serial en los 7 grids → evidencia de correctitud (b). Si falla, se detiene. |
| c | `c_cfg2_2x2_T2_n1` | 2×2, T=2, b = 4096…64, 3 reps. Calcula `b_c` (mediana mínima) → `resultados_cluster/b_c.txt` |
| d | 14 jobs `d_cfg{0..3}_*` | Configuraciones 0–3 con todos sus T, b ∈ {b_a, b_c} |
| e | `e_cfg4_2x4_T1_n2`, `e_cfg5_4x4_T1_n2` | 2 nodos |
| f | `f_cfg2_2x2_T1_n1` | 2×2, T=1, b = 4096 y 512 (el caso T=2 ya sale de (c)) |

`b_a` = 512 por defecto (mejor mediana en `resultados_lu_serial.txt`). Para cambiarlo: `B_A=256 bash slurm/enviar_todo.sh`.
Para repetir solo algunas etapas: `ETAPAS="d e" bash slurm/enviar_todo.sh`.

Seguimiento: `squeue -u $USER`, `tail -f logs/orquestador.log`, `ls resultados_cluster/`.

## 3. Traer resultados y generar tablas/gráficos

```bash
# en tu computador
scp -r kraken:~/T3/resultados_cluster kraken:~/T3/logs T3/
cd T3 && python3 analisis/analizar.py      # requiere pandas, matplotlib, tabulate
```

Salida en `analisis/salida/`: `tablas.md` (todas las tablas listas para el informe), CSVs y
PNGs: `a_serial_tiempo_vs_b`, `c_tiempo_vs_b`, `g_desglose_componentes_vs_b`, `d_speedup`,
`d_eficiencia`, `e_1nodo_vs_2nodos`, `f_por_proceso`.

## Si algo falla

- **La prueba de 2 nodos se cuelga o falla** (típico: OpenMPI elige una interfaz de red equivocada).
  Ver interfaces con `ip -br addr` en ventress/yodaxico y relanzar con, p. ej.:
  `MPI_EXTRA="--mca btl_tcp_if_include <interfaz>" bash slurm/enviar_todo.sh`
  (si es OpenMPI 5 agregar también `--mca oob_tcp_if_include <interfaz>`).
- **Out of memory**: cada job pide `--mem=3500M` (límite 4 GB). El peor caso (1×1, N=8192) usa ~1,2 GB.
- **Un job llega a 15 min**: `comun.sh` no inicia corridas nuevas después de 13 min; los CSV se
  escriben corrida a corrida, así que no se pierde lo ya medido.
- **Correr un job suelto a mano**: `sbatch --export=ALL,B_LIST="512 256" slurm/jobs/d_cfg2_2x2_T1_n1.sbatch`

## Qué se mide (columnas nuevas en el CSV)

Las columnas originales de `lu_distributed.py` no cambiaron. Se agregaron al final:
`T, nodos, host, parte, config, rep, diag_s, panel_s, update_s, bcast_diag_s, bcast_panel_s, bytes_bcast, jobid`.
`computo_s = diag_s + panel_s + update_s` y `comunicacion_s = bcast_diag_s + bcast_panel_s`.
El tiempo de `Bcast` incluye la espera por el proceso raíz (desbalance), lo que conviene comentar en (f) y (g).
`host` permite mostrar en (e) qué ranks quedaron en cada nodo (con 2×4 en 2 nodos, cada fila del grid queda en un nodo:
`row_comm` es intra-nodo y `col_comm` inter-nodo).

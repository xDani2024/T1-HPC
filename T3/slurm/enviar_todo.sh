#!/bin/bash
# Orquestador: envia todos los jobs de la Tarea 3 respetando los limites del curso
# (max 4 jobs en cola, 1 en ejecucion). Correr desde la carpeta T3 en el nodo de login,
# idealmente dentro de tmux/screen o con nohup:
#     nohup bash slurm/enviar_todo.sh > logs/orquestador.log 2>&1 &
#
# Etapas (por defecto todas):  ETAPAS="prueba c d e f" bash slurm/enviar_todo.sh
#   prueba : valida entorno y correctitud con MPI real (1 y 2 nodos)
#   c      : 2x2, T=2, b = 4096..64           -> elige b_c automaticamente (mediana minima)
#   d,e,f  : usan B_LIST = "B_A b_c"
# B_A: b elegido en (a). Por defecto 512 (mediana minima en resultados_lu_serial.txt).
set -uo pipefail
cd "$(dirname "$0")/.."            # carpeta T3
ETAPAS=${ETAPAS:-"prueba c d e f"}
B_A=${B_A:-512}
MAX_COLA=${MAX_COLA:-4}
mkdir -p logs resultados_cluster

python3 slurm/generar_sbatch.py > /dev/null

en_cola() { squeue -u "${USER:-$(whoami)}" -h | wc -l; }

esperar_cupo() {
  while [ "$(en_cola)" -ge "$MAX_COLA" ]; do sleep 20; done
}

esperar_job() {   # espera a que termine el job $1
  while squeue -j "$1" -h 2>/dev/null | grep -q .; do sleep 20; done
  sacct -j "$1" --format=JobID,JobName%30,State,Elapsed -n 2>/dev/null | head -1
}

enviar() {        # enviar <archivo.sbatch> [B_LIST]
  esperar_cupo
  local id
  if [ -n "${2:-}" ]; then
    id=$(sbatch --parsable --export=ALL,B_LIST="$2" "$1")
  else
    id=$(sbatch --parsable "$1")
  fi
  if [ -z "$id" ]; then
    echo "$(date +%T) ERROR: sbatch rechazo $1. Se detiene el orquestador." >&2
    kill -TERM $$; exit 1
  fi
  echo "$(date +%T) enviado $1 (B_LIST='${2:-default}') -> job $id" >&2
  echo "$id"
}

tiene() { [[ " $ETAPAS " == *" $1 "* ]]; }

# ---------- prueba de entorno ----------
if tiene prueba; then
  id=$(enviar slurm/jobs_extra/prueba_entorno.sbatch)
  esperar_job "$id"
  ok=$(grep -c "CHECK: OK" logs/lu_prueba_entorno_"$id".out 2>/dev/null || true)
  echo "Prueba de entorno: $ok/21 CHECK OK (ver logs/lu_prueba_entorno_$id.out)"
  if [ "${ok:-0}" -lt 15 ]; then
    echo "ATENCION: la prueba fallo (al menos en 1 nodo). Revise el log antes de seguir."; exit 1
  fi
  if [ "$ok" -lt 21 ]; then
    echo "ATENCION: fallan pruebas de 2 nodos (grid 4x4). Se sigue; revise MPI_EXTRA para la parte e."
  fi
fi

# ---------- parte c ----------
if tiene c; then
  id=$(enviar slurm/jobs/c_cfg2_2x2_T2_n1.sbatch)
  esperar_job "$id"
fi

B_C=$(python3 - <<'EOF'
import csv, glob, statistics, collections
t = collections.defaultdict(lambda: collections.defaultdict(float))
for f in glob.glob("resultados_cluster/c_*.csv"):
    for r in csv.DictReader(open(f)):
        t[int(r["b"])][(f, r["fecha"], r["rep"])] = float(r["tiempo_global_s"])
med = {b: statistics.median(v.values()) for b, v in t.items()}
print(min(med, key=med.get) if med else "")
EOF
)
if [ -z "$B_C" ]; then echo "No hay resultados de (c); usando solo B_A=$B_A"; B_C=$B_A; fi
echo "$B_C" > resultados_cluster/b_c.txt
B_LIST=$(echo "$B_A $B_C" | tr ' ' '\n' | sort -nu | tr '\n' ' ')
echo "b_a=$B_A  b_c=$B_C  -> B_LIST='$B_LIST'"

# ---------- partes d, e, f ----------
ids=()
for parte in d e f; do
  tiene "$parte" || continue
  for job in slurm/jobs/${parte}_*.sbatch; do
    if [ "$parte" = f ]; then ids+=("$(enviar "$job")")      # f usa b = N/2 y N/16
    else ids+=("$(enviar "$job" "$B_LIST")"); fi
  done
done
for id in "${ids[@]}"; do esperar_job "$id"; done
echo "$(date +%T) Listo. CSVs en resultados_cluster/ ; logs en logs/"
echo "Siguiente paso: python3 analisis/analizar.py"

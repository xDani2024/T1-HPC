#!/bin/bash
set -e
echo "== $(date) borrando instalacion incompleta"
rm -rf ~/miniconda3 /tmp/miniconda.sh
echo "== $(date) descargando Miniforge"
curl -fsSL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
echo "== $(date) instalando Miniforge en ~/miniconda3"
bash /tmp/miniforge.sh -b -p ~/miniconda3
echo "== $(date) creando entorno tarea3lu"
cd ~/T3 && bash slurm/setup_cluster.sh
echo "== $(date) INSTALACION_TERMINADA"

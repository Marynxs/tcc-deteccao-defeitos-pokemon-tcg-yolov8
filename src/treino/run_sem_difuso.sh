#!/bin/bash
# Configuracao 2 a 1280 nas 5 dobras, sem as 9 cartas de desgaste difuso no TREINO
# (016, 047, 072, 073 a 077, 108; 1.563 das 4.808 caixas). Validacao interna e dobra retida
# iguais as da campanha: so o treino muda. Comparacao depois, por carta, nas 197 outras cartas.
# Comeca assim que a GPU fica livre (treino atual e SAHI). Retoma se relancado.
#   nohup bash src/treino/run_sem_difuso.sh >> Dataset_YOLO/runs_206_rev/sem_difuso.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
R=Dataset_YOLO/runs_206_rev
L="$PWD/Dataset_YOLO/folds_206_rev/sem_difuso"
while pgrep -f "[r]un_640_dobras.sh|[s]ahi_apos_treino.sh" >/dev/null; do sleep 60; done
for f in 0 1 2 3 4; do
  nome=cfg2_fold${f}_1280_semdifuso
  [ -f "$R/$nome/resumo.json" ] && { echo "== $(date '+%F %T') $nome ja concluida"; continue; }
  echo "== $(date '+%F %T') sem difuso: dobra $f ($nome)"
  extra=""
  [ -f "$R/$nome/weights/last.pt" ] && extra="--resume"
  $PY src/treino/train_fold.py --config 2 --fold "$f" --pool pool_206_rev --cache ram \
      --train-list "$L/fold${f}_treino.txt" --name "$nome" $extra
done
echo "== $(date '+%F %T') sem difuso: concluida; backup"
[ -f src/ferramentas/backup_drive.sh ] && { bash src/ferramentas/backup_drive.sh || echo "== backup falhou"; }
echo "== $(date '+%F %T') FIM SEM DIFUSO"

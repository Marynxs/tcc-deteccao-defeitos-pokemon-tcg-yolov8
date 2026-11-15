#!/bin/bash
# Curva de aprendizado (25/50/75% das cartas de treino) nas dobras que faltavam (0, 1 e 3),
# para a curva sair nas cinco dobras e permitir a comparacao carta a carta.
# Retoma de onde parou se relancado (resumo.json = concluida; last.pt = continua).
#   nohup bash src/treino/run_curva_dobras.sh >> Dataset_YOLO/runs_206_rev/campanha.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
for f in 0 1 3; do
    echo "== $(date '+%F %T') curva de aprendizado, cfg2 dobra $f"
    $PY src/treino/learning_curve.py --fold $f --config 2 --pool pool_206_rev
done
echo "== $(date '+%F %T') curva nas cinco dobras: concluida; backup"
bash src/ferramentas/backup_drive.sh

#!/bin/bash
# Noite de 19/09/2026: duas provas sobre a cfg2, dobra 4, em sequencia.
#   1. curva de aprendizado (25/50/75% das cartas de treino)  -> falta de dados?
#   2. cfg2 a 640 (mesmo lote 4, mesma dobra)                  -> efeito da resolucao de entrada
#   3. yolov8m no lugar do yolov8s                             -> capacidade do modelo?
# Retoma de onde parou se relancado (resumo.json = concluida; last.pt = continua).
#   nohup bash src/treino/run_noite_curva_e_m.sh >> Dataset_YOLO/runs_206_rev/campanha.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
R=Dataset_YOLO/runs_206_rev

echo "== $(date '+%F %T') noite: curva de aprendizado, cfg2 dobra 4"
$PY src/treino/learning_curve.py --fold 4 --config 2 --pool pool_206_rev

nome=cfg2_fold4_640
echo "== $(date '+%F %T') noite: cfg2 a 640, dobra 4 ($nome)"
extra=""
[ -f "$R/$nome/weights/last.pt" ] && [ ! -f "$R/$nome/resumo.json" ] && extra="--resume"
$PY src/treino/train_fold.py --config 2 --fold 4 --pool pool_206_rev --cache ram \
    --imgsz 640 --name "$nome" $extra

nome=cfg2_fold4_1280_m
echo "== $(date '+%F %T') noite: yolov8m, cfg2 dobra 4 ($nome)"
extra=""
[ -f "$R/$nome/weights/last.pt" ] && [ ! -f "$R/$nome/resumo.json" ] && extra="--resume"
$PY src/treino/train_fold.py --config 2 --fold 4 --pool pool_206_rev --cache ram \
    --model yolov8m.pt --name "$nome" $extra

echo "== $(date '+%F %T') noite: concluida; backup"
bash src/ferramentas/backup_drive.sh
echo "== $(date '+%F %T') FIM DA NOITE"

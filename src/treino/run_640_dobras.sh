#!/bin/bash
# Configuracao 2 a 640 pixels nas dobras que faltam (0 a 3; a dobra 4 rodou em 19/09/2026).
# Com as cinco dobras, a comparacao 1280 x 640 passa a ter as 206 cartas como pares, como o fatorial.
# Mesmos hiperparametros, mesmo lote 4 e mesmas dobras das rodadas a 1280; so muda --imgsz.
# Retoma de onde parou se relancado (resumo.json = concluida; last.pt = continua).
#   nohup bash src/treino/run_640_dobras.sh >> Dataset_YOLO/runs_206_rev/campanha.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
R=Dataset_YOLO/runs_206_rev

for f in 0 1 2 3; do
  nome=cfg2_fold${f}_640
  [ -f "$R/$nome/resumo.json" ] && { echo "== $(date '+%F %T') 640: $nome ja concluida"; continue; }
  echo "== $(date '+%F %T') 640: cfg2 dobra $f ($nome)"
  extra=""
  [ -f "$R/$nome/weights/last.pt" ] && extra="--resume"
  $PY src/treino/train_fold.py --config 2 --fold "$f" --pool pool_206_rev --cache ram \
      --imgsz 640 --name "$nome" $extra
done

echo "== $(date '+%F %T') 640: concluida; backup"
[ -f src/ferramentas/backup_drive.sh ] && { bash src/ferramentas/backup_drive.sh || echo "== $(date '+%F %T') 640: backup falhou, seguindo mesmo assim"; }
echo "== $(date '+%F %T') FIM 640"

# Em seguida, o controle de luminosidade: Configuracao 3 com a luminancia (L*) repetida nos tres
# canais, sem realce, nas cinco dobras (decidido com o autor em 12/11/2026). Separa a perda da cor
# do efeito do realce nas comparacoes das Configuracoes 3 e 4 com as 1 e 2.
ds=Dataset_YOLO/dataset_206_rev_realce_l
[ -d "$ds/images" ] || $PY src/dados/enhance_relief.py --variante l --pool pool_206_rev
for f in 0 1 2 3 4; do
  nome=cfg3_fold${f}_1280_lum
  [ -f "$R/$nome/resumo.json" ] && { echo "== $(date '+%F %T') lum: $nome ja concluida"; continue; }
  echo "== $(date '+%F %T') lum: cfg3 dobra $f ($nome)"
  extra=""
  [ -f "$R/$nome/weights/last.pt" ] && extra="--resume"
  $PY src/treino/train_fold.py --config 3 --fold "$f" --realce l --pool pool_206_rev --cache ram \
      --name "$nome" $extra
done
echo "== $(date '+%F %T') lum: concluida; backup"
[ -f src/ferramentas/backup_drive.sh ] && { bash src/ferramentas/backup_drive.sh || echo "== backup falhou"; }
echo "== $(date '+%F %T') FIM LUM"

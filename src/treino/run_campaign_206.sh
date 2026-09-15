#!/usr/bin/env bash
# Campanha completa sobre as 206 cartas, na ordem de dependencia:
#   1. gera as 4 variantes de realce sobre dataset_206 (a, b, d, e)
#   2. teste de realce nas 5 dobras: config 3 com cada variante
#   3. fatorial 2x2 com a variante vencedora (escolhida pelo autor a partir do
#      passo 2, dobra retida, media das 5 dobras)
# Cache em RAM: 266 imagens a 1280 cabem em ~1 GB dos 30 GB e nao gravam .npy.
#
#   bash src/treino/run_campaign_206.sh realce      # passos 1 e 2
#   bash src/treino/run_campaign_206.sh fatorial b  # passo 3 com a variante b
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python

etapa="${1:-}"
case "$etapa" in
  realce)
    for v in a b d e; do
      [ -d "Dataset_YOLO/dataset_206_realce_$v/images" ] || \
        $PY src/dados/enhance_relief.py --variante "$v" --pool pool_206
    done
    for v in a b d e; do
      for f in 0 1 2 3 4; do
        $PY src/treino/train_fold.py --config 3 --fold "$f" --realce "$v" \
          --pool pool_206 --cache ram --name "cfg3_fold${f}_1280_realce_$v"
      done
    done
    ;;
  fatorial)
    v="${2:?informe a variante vencedora: a, b, d ou e}"
    sed -i "s/^REALCE_PADRAO = .*/REALCE_PADRAO = \"$v\"/" src/treino/train_fold.py
    $PY src/treino/run_campaign.py --pool pool_206 --cache ram
    ;;
  *)
    echo "uso: $0 realce | fatorial <variante>"; exit 1;;
esac

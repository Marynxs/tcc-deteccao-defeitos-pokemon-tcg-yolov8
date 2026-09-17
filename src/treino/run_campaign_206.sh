#!/usr/bin/env bash
# Campanha completa sobre as 206 cartas com a anotacao REVISADA (pool_206_rev), na ordem:
#   1+2. teste de realce nas 5 dobras: para cada variante (a, b, d, e) gera o dataset de
#        realce, treina a config 3 nas 5 dobras e APAGA o dataset (2,3 GB cada; o disco
#        nao comporta os quatro ao mesmo tempo). Regenerar custa minutos.
#   3.   fatorial 2x2 com a variante vencedora (escolhida pelo autor a partir do passo 2,
#        dobra retida, media das 5 dobras); regenera o dataset da vencedora se faltar.
# Cache em RAM: as imagens a 1280 cabem em ~1 GB dos 30 GB e nao gravam .npy.
#
#   bash src/treino/run_campaign_206.sh realce      # passos 1 e 2
#   bash src/treino/run_campaign_206.sh fatorial b  # passo 3 com a variante b
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
POOL=pool_206_rev
SUF=${POOL#pool}          # _206_rev

etapa="${1:-}"
case "$etapa" in
  realce)
    for v in a b d e; do
      ds="Dataset_YOLO/dataset${SUF}_realce_$v"
      [ -d "$ds/images" ] || $PY src/dados/enhance_relief.py --variante "$v" --pool "$POOL"
      for f in 0 1 2 3 4; do
        $PY src/treino/train_fold.py --config 3 --fold "$f" --realce "$v" \
          --pool "$POOL" --cache ram --name "cfg3_fold${f}_1280_realce_$v"
      done
      rm -rf "$ds"
    done
    ;;
  fatorial)
    v="${2:?informe a variante vencedora: a, b, d ou e}"
    ds="Dataset_YOLO/dataset${SUF}_realce_$v"
    [ -d "$ds/images" ] || $PY src/dados/enhance_relief.py --variante "$v" --pool "$POOL"
    sed -i "s/^REALCE_PADRAO = .*/REALCE_PADRAO = \"$v\"/" src/treino/train_fold.py
    $PY src/treino/run_campaign.py --pool "$POOL" --cache ram
    ;;
  *)
    echo "uso: $0 realce | fatorial <variante>"; exit 1;;
esac

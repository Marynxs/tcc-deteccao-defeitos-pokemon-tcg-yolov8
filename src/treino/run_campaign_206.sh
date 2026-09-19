#!/usr/bin/env bash
# Campanha completa sobre as 206 cartas com a anotacao REVISADA (pool_206_rev), na ordem:
#   1+2. teste de realce como SELECAO DE HIPERPARAMETRO (decisao do autor, 17/09/2026):
#        para cada variante (a, b, d, e) gera o dataset de realce, treina a config 3 na
#        DOBRA 0 e APAGA o dataset (2,3 GB cada). A escolha e pela VALIDACAO INTERNA
#        (f1_val_interna / mAP interno no resumo.json), nao pela dobra retida, para nao
#        contaminar o fatorial (Cawley2010). 4 rodadas, ~4 h.
#   3.   fatorial 2x2 nas 5 dobras com a variante escolhida; regenera o dataset se faltar.
#   4.   extensao P2: a config vencedora do fatorial, mesmo tudo, com a cabeca P2
#        (yolov8s-p2.yaml + pesos do yolov8s.pt), nas 5 dobras. Medido: 11,3 GB de VRAM
#        com lote 4 a 1280 e ~2x o tempo por epoca. Comparacao pareada por carta contra
#        a mesma config sem P2.
# Cache em RAM: as imagens a 1280 cabem em ~1 GB dos 30 GB e nao gravam .npy.
#
#   bash src/treino/run_campaign_206.sh realce      # passos 1 e 2
#   bash src/treino/run_campaign_206.sh fatorial b  # passo 3 com a variante b
#   bash src/treino/run_campaign_206.sh p2 2         # passo 4 com a config 2 vencedora
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
      # rodada ja concluida: nao regenera o dataset (relancamento apos queda)
      [ -f "Dataset_YOLO/runs${SUF}/cfg3_fold0_1280_realce_$v/resumo.json" ] && continue
      [ -d "$ds/images" ] || $PY src/dados/enhance_relief.py --variante "$v" --pool "$POOL"
      $PY src/treino/train_fold.py --config 3 --fold 0 --realce "$v" \
        --pool "$POOL" --cache ram --name "cfg3_fold0_1280_realce_$v"
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
  p2)
    c="${2:?informe a config vencedora do fatorial: 1, 2, 3 ou 4}"
    # A cabeca P2 parte sem pre-treino e atinge o pico bem mais tarde. A regra de
    # parada e a mesma do fatorial (100 epocas sem melhora na validacao interna);
    # o teto de epocas e so seguranca. Se uma dobra bater no teto com o pico nas
    # ultimas 100 epocas, a regra nao foi cumprida: estende o teto em 100 e retoma
    # do last.pt (mesmo otimizador, mesmo best.pt), ate a folga existir ou 600.
    for f in 0 1 2 3 4; do
      nome="cfg${c}_fold${f}_1280_p2"; dir="Dataset_YOLO/runs${SUF}/$nome"; teto=300
      while :; do
        $PY src/treino/train_fold.py --config "$c" --fold "$f" --pool "$POOL" --cache ram \
          --model yolov8s-p2.yaml --name "$nome" --epochs "$teto" \
          $([ -f "$dir/weights/last.pt" ] && echo --resume)
        [ -f "$dir/resumo.json" ] || break              # falhou: deixa o run_pos/autor decidir
        # folga = epocas depois do pico pelo MESMO criterio da biblioteca (fitness,
        # 0,9*mAP50-95 + 0,1*mAP50), que e o que grava best.pt e conta a paciencia;
        # o pico do mAP50 puro cai em outra epoca e nao e o que decide a parada
        # Criterio da propria biblioteca: ao parar por paciencia ela imprime "Training
        # stopped early ... Best results observed at epoch N" (o fitness dela tem
        # suavizacao que nao se reproduz pelo results.csv). Se a rodada chegou ao teto,
        # essa mensagem nao existe para ela e a paciencia NAO foi cumprida: estende.
        parou_por_paciencia=$(awk -v n="$nome" '$0 ~ ("name=" n) {f=1; c=0} f && /Training stopped early/ {c=1} END {print c+0}' "Dataset_YOLO/runs${SUF}/campanha.log")
        ep=$($PY -c "import json;print(json.load(open('$dir/resumo.json'))['epocas_rodadas'])")
        if [ "$ep" -ge "$teto" ] && [ "$parou_por_paciencia" -eq 0 ] && [ "$teto" -lt 600 ]; then
          teto=$((teto+100))
          echo "== $(date '+%F %T') $nome: bateu no teto de $ep epocas sem parada por paciencia; estendendo para $teto e retomando"
          rm -f "$dir/resumo.json"
        else
          echo "== $(date '+%F %T') $nome: concluida em $ep epocas (parada por paciencia da biblioteca: $([ "$parou_por_paciencia" -gt 0 ] && echo sim || echo nao))"
          break
        fi
      done
    done
    ;;
  *)
    echo "uso: $0 realce | fatorial <variante> | p2 <config>"; exit 1;;
esac

#!/usr/bin/env bash
# Encadeia o que vem depois do fatorial 2x2 (decisao do autor, 17/09/2026): espera o
# processo do fatorial terminar, escolhe a config vencedora pela media do mAP50 nas
# 5 dobras retidas, faz backup no Drive, e ja lanca a extensao P2 nela, sem esperar
# confirmacao. Roda em paralelo ao fatorial (que ja esta em andamento); nao lanca nada
# do fatorial em si.
#
#   nohup bash src/treino/run_pos_fatorial.sh <PID_do_bash_do_fatorial> >> Dataset_YOLO/runs_206_rev/campanha.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
PID="${1:?informe o PID do processo 'run_campaign_206.sh fatorial'}"

echo "== $(date '+%F %T') run_pos_fatorial: aguardando o fatorial terminar (PID $PID)"
while kill -0 "$PID" 2>/dev/null; do sleep 30; done

# confere que as 20 rodadas do fatorial de fato existem antes de escolher (senao o
# processo pode ter morrido por queda de energia, nao por conclusao)
n=$(ls Dataset_YOLO/runs_206_rev/cfg[1-4]_fold[0-4]_1280/resumo.json 2>/dev/null | wc -l)
if [ "$n" -lt 20 ]; then
  echo "== $(date '+%F %T') run_pos_fatorial: fatorial NAO concluido ($n/20 rodadas). Encerrando sem acao. Relance o fatorial e depois este script."
  exit 1
fi

vencedora=$($PY - <<'PYEOF'
import json, glob, statistics as st
melhor = None
for c in (1, 2, 3, 4):
    m = [json.load(open(p))["map50"] for p in sorted(glob.glob(f"Dataset_YOLO/runs_206_rev/cfg{c}_fold*_1280/resumo.json"))]
    media = st.mean(m)
    print(f"# cfg{c}: mAP50 por dobra {[round(x,3) for x in m]} media {media:.4f}", flush=True)
    if melhor is None or media > melhor[1]:
        melhor = (c, media)
print(melhor[0])
PYEOF
)
echo "$vencedora" | grep '^#'
cfg=$(echo "$vencedora" | tail -1)
echo "== $(date '+%F %T') run_pos_fatorial: config vencedora = cfg$cfg" | tee -a Dataset_YOLO/runs_206_rev/config_vencedora.txt

echo "== $(date '+%F %T') run_pos_fatorial: backup no Drive"
bash src/ferramentas/backup_drive.sh || echo "== $(date '+%F %T') run_pos_fatorial: backup falhou, seguindo mesmo assim"

echo "== $(date '+%F %T') run_pos_fatorial: iniciando P2 na cfg$cfg (5 dobras)"
bash src/treino/run_campaign_206.sh p2 "$cfg"
echo "== $(date '+%F %T') run_pos_fatorial: P2 concluida"

echo "== $(date '+%F %T') run_pos_fatorial: backup final no Drive"
bash src/ferramentas/backup_drive.sh || echo "== $(date '+%F %T') run_pos_fatorial: backup final falhou"
echo "== $(date '+%F %T') FIM DA CAMPANHA COMPLETA (fatorial + P2)"

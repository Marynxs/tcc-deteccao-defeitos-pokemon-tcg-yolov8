#!/usr/bin/env bash
# Piloto automatico da campanha das 206 (decisao do autor em 17/09/2026): roda o teste de
# realce (4 variantes x dobra 0), escolhe a variante pela VALIDACAO INTERNA (f1_val_interna,
# desempate por map50_val_interna) e emenda o fatorial 2x2 nas 5 dobras sem esperar ninguem.
# A P2 NAO entra aqui: depende da config vencedora do fatorial, que o autor confere.
# Log em Dataset_YOLO/runs_206_rev/campanha.log; cada rodada concluida grava resumo.json,
# entao relancar retoma de onde parou.
#
#   nohup bash src/treino/run_campaign_206_auto.sh > Dataset_YOLO/runs_206_rev/campanha.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
RUNS=Dataset_YOLO/runs_206_rev
mkdir -p "$RUNS"
echo "== $(date '+%F %T') inicio: realce (4 variantes x dobra 0)"
bash src/treino/run_campaign_206.sh realce
v=$($PY - <<'PYEOF'
import json, pathlib
runs = pathlib.Path("Dataset_YOLO/runs_206_rev")
cand = []
for v in "abde":
    r = runs / f"cfg3_fold0_1280_realce_{v}" / "resumo.json"
    if r.exists():
        d = json.loads(r.read_text())
        cand.append((d.get("f1_val_interna", 0), d.get("map50_val_interna", 0), v, d.get("map50")))
cand.sort(reverse=True)
for f1, m, v, ret in cand:
    print(f"# variante {v}: f1_val_interna {f1:.4f}  map50_val_interna {m:.4f}  (dobra retida map50 {ret}, so informativo)", flush=True)
print(cand[0][2])
PYEOF
)
escolha=$(echo "$v" | tail -1)
echo "$v" | grep '^#' || true
echo "== $(date '+%F %T') variante escolhida pela validacao interna: $escolha"
echo "$escolha" > "$RUNS/variante_escolhida.txt"
echo "== $(date '+%F %T') inicio: fatorial 2x2 x 5 dobras com realce $escolha"
bash src/treino/run_campaign_206.sh fatorial "$escolha"
echo "== $(date '+%F %T') FIM da campanha (realce + fatorial). P2 aguarda a config vencedora."

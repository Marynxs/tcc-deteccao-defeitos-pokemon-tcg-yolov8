"""
Servidor local que mostra o andamento do treino no navegador, em tempo real.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "Dataset_YOLO" / "runs"

# Quantos segundos sem o results.csv mudar antes de considerar o treino parado.
# Uma epoca custa cerca de 12 s a 1280; 90 s cobre com folga a epoca mais lenta
# somada a validacao, sem demorar demais para perceber uma queda.
IDLE_LIMIT = 90


def latest_run() -> Path | None:
    """Devolve a rodada com o results.csv modificado mais recentemente."""
    candidates = [d for d in RUNS.glob("*/results.csv")] if RUNS.is_dir() else []
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime).parent


def is_training(run: Path) -> bool:
    """Procura em /proc um processo cuja linha de comando cite esta rodada."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        if "train_fold.py" in cmd and run.name in cmd:
            return True
    return False


def planned_epochs(run: Path) -> tuple[int, int]:
    """Le epocas e paciencia do args.yaml que a Ultralytics grava na rodada."""
    epochs, patience = 0, 0
    args_file = run / "args.yaml"
    if args_file.is_file():
        texto = args_file.read_text(encoding="utf-8", errors="replace")
        for chave, alvo in (("epochs", "epochs"), ("patience", "patience")):
            m = re.search(rf"^{chave}:\s*(\d+)", texto, re.M)
            if m:
                if alvo == "epochs":
                    epochs = int(m.group(1))
                else:
                    patience = int(m.group(1))
    return epochs, patience


def read_history(run: Path) -> list[dict]:
    csv_file = run / "results.csv"
    if not csv_file.is_file():
        return []
    linhas = []
    with csv_file.open(encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            try:
                linhas.append({
                    "epoca": int(float(row["epoch"])),
                    "segundos": float(row["time"]),
                    "box": float(row["train/box_loss"]),
                    "cls": float(row["train/cls_loss"]),
                    "dfl": float(row["train/dfl_loss"]),
                    "map50": float(row["metrics/mAP50(B)"]),
                    "map5095": float(row["metrics/mAP50-95(B)"]),
                    "precisao": float(row["metrics/precision(B)"]),
                    "recall": float(row["metrics/recall(B)"]),
                })
            except (KeyError, ValueError):
                continue
    return linhas


def build_status() -> dict:
    run = latest_run()
    if run is None:
        return {"estado": "sem_rodada"}

    hist = read_history(run)
    epochs, patience = planned_epochs(run)
    resumo_file = run / "resumo.json"
    concluida = resumo_file.is_file()
    idade = time.time() - (run / "results.csv").stat().st_mtime
    viva = is_training(run)

    if concluida:
        estado = "concluida"
    elif viva:
        estado = "treinando"
    elif idade > IDLE_LIMIT:
        estado = "parada"
    else:
        estado = "treinando"

    atual = hist[-1] if hist else None
    melhor = max(hist, key=lambda d: d["map50"]) if hist else None

    feitas = atual["epoca"] if atual else 0
    decorrido = atual["segundos"] if atual else 0.0
    por_epoca = decorrido / feitas if feitas else 0.0

    # A parada antecipada pode encerrar antes do total planejado. O limite real
    # e o que vier primeiro: o total de epocas, ou a melhor epoca somada a
    # paciencia. Mostrar so o total planejado daria um "falta" sempre pessimista.
    limite_paciencia = (melhor["epoca"] + patience) if (melhor and patience) else epochs
    alvo = min(epochs, limite_paciencia) if epochs else limite_paciencia
    alvo = max(alvo, feitas)

    restantes = max(alvo - feitas, 0)
    return {
        "estado": estado,
        "rodada": run.name,
        "epochs_planejadas": epochs,
        "paciencia": patience,
        "alvo": alvo,
        "feitas": feitas,
        "restantes": restantes,
        "pct": (feitas / alvo * 100) if alvo else 0.0,
        "decorrido": decorrido,
        "por_epoca": por_epoca,
        "restante_seg": restantes * por_epoca,
        "inicio": (run / "results.csv").stat().st_mtime - decorrido,
        "atual": atual,
        "melhor": melhor,
        "serie": [{"e": d["epoca"], "m": d["map50"]} for d in hist],
        "resumo": json.loads(resumo_file.read_text(encoding="utf-8")) if concluida else None,
    }


PAGE = r"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Treino YOLOv8</title>
<style>
:root{
  --fundo:#0f1117; --painel:#171a24; --borda:#252a38;
  --texto:#e9ecf5; --fraco:#878ea8;
  --ambar:#ffb020; --verde:#3ddc97; --vermelho:#ff5f6d;
}
*{box-sizing:border-box}
body{margin:0;background:var(--fundo);color:var(--texto);
  font:14px/1.5 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif;
  display:flex;justify-content:center;padding:32px 16px}
.wrap{width:100%;max-width:760px}
h1{margin:0 0 2px;font-size:19px;font-weight:650;letter-spacing:-.2px}
.sub{color:var(--fraco);font-size:12.5px;margin-bottom:22px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.card{background:var(--painel);border:1px solid var(--borda);
  border-radius:12px;padding:20px;margin-bottom:14px}
.topo{display:flex;justify-content:space-between;align-items:baseline;
  gap:12px;margin-bottom:14px}
.pct{font:600 34px/1 ui-monospace,SFMono-Regular,Menlo,monospace;
  font-variant-numeric:tabular-nums}
.chip{font-size:11px;font-weight:600;text-transform:uppercase;
  letter-spacing:.09em;padding:4px 10px;border-radius:999px;
  border:1px solid currentColor}
.treinando{color:var(--ambar)} .concluida{color:var(--verde)}
.parada{color:var(--vermelho)} .sem_rodada{color:var(--fraco)}
.barra{height:22px;background:#0b0d13;border:1px solid var(--borda);
  border-radius:6px;overflow:hidden;position:relative}
.preenche{height:100%;width:0;transition:width .6s ease;
  background:repeating-linear-gradient(115deg,var(--ambar) 0 14px,#e29a12 14px 28px)}
.concluida .preenche,.preenche.ok{background:repeating-linear-gradient(
  115deg,var(--verde) 0 14px,#31b47e 14px 28px)}
.grade{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));
  gap:1px;background:var(--borda);border:1px solid var(--borda);
  border-radius:10px;overflow:hidden}
.celula{background:var(--painel);padding:13px 15px}
.rot{color:var(--fraco);font-size:10.5px;text-transform:uppercase;
  letter-spacing:.08em;margin-bottom:5px}
.val{font:600 19px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace;
  font-variant-numeric:tabular-nums}
.val small{font-size:12px;font-weight:400;color:var(--fraco)}
.tit{font-size:11px;color:var(--fraco);text-transform:uppercase;
  letter-spacing:.08em;margin:0 0 12px}
svg{width:100%;height:110px;display:block}
.aviso{color:var(--fraco);text-align:center;padding:36px 0}
</style></head><body><div class="wrap">
<h1>Detecção de defeitos, treino YOLOv8</h1>
<div class="sub" id="rodada">carregando...</div>
<div class="card" id="painel">
  <div class="topo">
    <div><div class="rot">progresso</div><div class="pct" id="pct">--</div></div>
    <div class="chip sem_rodada" id="estado">aguardando</div>
  </div>
  <div class="barra"><div class="preenche" id="preenche"></div></div>
  <div class="rot" style="margin-top:10px" id="epocas">--</div>
</div>
<div class="grade" id="grade"></div>
<div class="card" style="margin-top:14px">
  <p class="tit">mAP@50 por época</p><svg id="graf" viewBox="0 0 700 110"
   preserveAspectRatio="none"></svg>
</div>
</div>
<script>
const q=i=>document.getElementById(i);
const hms=s=>{if(!isFinite(s)||s<0)return"--";s=Math.round(s);
  const h=Math.floor(s/3600),m=Math.floor(s%3600/60),g=s%60;
  return h?`${h}h ${String(m).padStart(2,"0")}min`:m?`${m}min ${String(g).padStart(2,"0")}s`:`${g}s`};
const hora=t=>new Date(t*1000).toLocaleTimeString("pt-BR");
const rots={treinando:"treinando",concluida:"concluída",parada:"parada",sem_rodada:"sem rodada"};

function celula(rot,val,extra){return `<div class="celula"><div class="rot">${rot}</div>
  <div class="val">${val}${extra?` <small>${extra}</small>`:""}</div></div>`}

function grafico(serie){
  const svg=q("graf");
  if(!serie.length){svg.innerHTML="";return}
  const W=700,H=110,P=6;
  const mx=Math.max(...serie.map(p=>p.m),1e-6);
  const n=Math.max(serie.length-1,1);
  const pts=serie.map((p,i)=>[P+i/n*(W-2*P),H-P-(p.m/mx)*(H-2*P)]);
  const linha=pts.map(([x,y])=>`${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const area=`${P},${H-P} ${linha} ${(W-P).toFixed(1)},${H-P}`;
  const [bx,by]=pts[serie.reduce((b,p,i)=>serie[b].m>=p.m?b:i,0)];
  svg.innerHTML=`<polygon points="${area}" fill="rgba(255,176,32,.14)"/>
    <polyline points="${linha}" fill="none" stroke="#ffb020" stroke-width="2"
      stroke-linejoin="round" vector-effect="non-scaling-stroke"/>
    <circle cx="${bx.toFixed(1)}" cy="${by.toFixed(1)}" r="3.5" fill="#3ddc97"/>`;
}

async function tick(){
  let d; try{d=await (await fetch("/api/status",{cache:"no-store"})).json()}catch(e){return}
  if(d.estado==="sem_rodada"){q("rodada").textContent="nenhuma rodada encontrada";
    q("grade").innerHTML='<div class="aviso">Aguardando o treino começar.</div>';return}

  q("rodada").textContent=d.rodada;
  q("pct").textContent=d.pct.toFixed(1)+"%";
  q("preenche").style.width=Math.min(d.pct,100)+"%";
  q("preenche").classList.toggle("ok",d.estado==="concluida");
  const c=q("estado"); c.className="chip "+d.estado; c.textContent=rots[d.estado]||d.estado;
  q("epocas").textContent=`época ${d.feitas} de ${d.alvo}`+
    (d.epochs_planejadas&&d.alvo<d.epochs_planejadas
      ? ` (limite da paciência; teto planejado ${d.epochs_planejadas})`:"");

  const a=d.atual||{},b=d.melhor||{};
  q("grade").innerHTML=[
    celula("começou às",hora(d.inicio)),
    celula("já rodou",hms(d.decorrido)),
    celula("falta",d.estado==="concluida"?"--":hms(d.restante_seg),
      d.estado==="concluida"?"":`${d.restantes} épocas`),
    celula("por época",d.por_epoca.toFixed(1)+"s"),
    celula("mAP@50 atual",(a.map50??0).toFixed(4)),
    celula("melhor mAP@50",(b.map50??0).toFixed(4),`época ${b.epoca??"--"}`),
    celula("recall",(a.recall??0).toFixed(4)),
    celula("perdas",`${(a.box??0).toFixed(2)}`,
      `cls ${(a.cls??0).toFixed(2)} · dfl ${(a.dfl??0).toFixed(2)}`),
  ].join("");
  grafico(d.serie||[]);
  document.title=`${d.pct.toFixed(0)}% · treino YOLOv8`;
}
tick(); setInterval(tick,2000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.startswith("/api/status"):
            corpo = json.dumps(build_status()).encode("utf-8")
            tipo = "application/json; charset=utf-8"
        elif self.path in ("/", "/index.html"):
            corpo = PAGE.encode("utf-8")
            tipo = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(corpo)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(corpo)

    def log_message(self, *args) -> None:
        pass  # silencia o log de acesso, que poluiria o terminal do treino


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    url = f"http://127.0.0.1:{args.port}/"
    servidor = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"monitor em {url}   (ctrl+c para encerrar)")
    if not args.no_browser and os.environ.get("DISPLAY"):
        webbrowser.open(url)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nencerrado")


if __name__ == "__main__":
    main()

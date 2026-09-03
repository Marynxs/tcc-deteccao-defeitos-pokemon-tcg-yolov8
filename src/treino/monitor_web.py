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


_GPU_CACHE: dict = {"quando": 0.0, "dado": {}}


def estado_gpu() -> dict:
    """Uso e temperatura da placa, via rocm-smi, com cache de 5 s."""
    import subprocess
    agora = time.time()
    if agora - _GPU_CACHE["quando"] < 5:
        return _GPU_CACHE["dado"]
    d = {}
    try:
        saida = subprocess.run(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showtemp"],
            capture_output=True, text=True, timeout=5).stdout
        for linha in saida.splitlines():
            if "GPU[0]" not in linha:
                continue
            if "GPU use (%)" in linha:
                d["uso"] = float(linha.rsplit(":", 1)[1])
            elif "VRAM Total Used Memory" in linha:
                d["vram_usada"] = float(linha.rsplit(":", 1)[1]) / 1024 ** 3
            elif "VRAM Total Memory" in linha:
                d["vram_total"] = float(linha.rsplit(":", 1)[1]) / 1024 ** 3
            elif "junction" in linha:
                d["temp"] = float(linha.rsplit(":", 1)[1])
    except Exception:
        pass
    _GPU_CACHE.update(quando=agora, dado=d)
    return d


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
        "serie": [
            {"e": d["epoca"], "m50": d["map50"], "m5095": d["map5095"],
             "p": d["precisao"], "r": d["recall"],
             "box": d["box"], "cls": d["cls"], "dfl": d["dfl"]}
            for d in hist
        ],
        "desde_melhor": (feitas - melhor["epoca"]) if melhor else 0,
        "gpu": estado_gpu(),
        "resumo": json.loads(resumo_file.read_text(encoding="utf-8")) if concluida else None,
    }


PAGE = r"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Treino YOLOv8</title>
<style>
:root{
  --fundo:#f7f7f5; --painel:#ffffff; --borda:#e3e2dd;
  --texto:#111111; --fraco:#5c5b57; --tenue:#8b8a84;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
  --ok:#1baf7a; --alerta:#eda100; --erro:#e34948;
}
@media (prefers-color-scheme: dark){:root{
  --fundo:#111311; --painel:#1a1c1a; --borda:#2c2f2c;
  --texto:#f2f3f0; --fraco:#a9aaa4; --tenue:#75766f;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70;
  --ok:#199e70; --alerta:#c98500; --erro:#e66767;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--fundo);color:var(--texto);
  font:14px/1.5 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif;padding:20px 24px 40px}
h1{margin:0 0 2px;font-size:20px;font-weight:650;letter-spacing:-.2px}
.sub{color:var(--fraco);font-size:12.5px;margin-bottom:18px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.card{background:var(--painel);border:1px solid var(--borda);
  border-radius:12px;padding:18px 20px;margin-bottom:14px}
.topo{display:flex;justify-content:space-between;align-items:baseline;gap:16px;margin-bottom:12px}
.pct{font:600 36px/1 ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}
.chip{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.09em;
  padding:4px 11px;border-radius:999px;border:1px solid currentColor}
.treinando{color:var(--alerta)} .concluida{color:var(--ok)}
.parada{color:var(--erro)} .sem_rodada{color:var(--tenue)}
.barra{height:20px;background:var(--fundo);border:1px solid var(--borda);
  border-radius:6px;overflow:hidden}
.preenche{height:100%;width:0;transition:width .6s ease;
  background:repeating-linear-gradient(115deg,var(--alerta) 0 14px,#d19100 14px 28px)}
.concluida-barra{background:repeating-linear-gradient(115deg,var(--ok) 0 14px,#158f63 14px 28px)!important}
.grade{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  background:var(--painel);border:1px solid var(--borda);
  border-radius:10px;overflow:hidden;margin-bottom:14px}
.celula{padding:12px 14px;border-right:1px solid var(--borda);
  border-bottom:1px solid var(--borda)}
.rot{color:var(--fraco);font-size:10.5px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px}
.val{font:600 18px/1.25 ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}
.val small{font-size:11.5px;font-weight:400;color:var(--fraco)}
.tit{font-size:11px;color:var(--fraco);text-transform:uppercase;letter-spacing:.08em;margin:0 0 4px}
.leg{display:flex;gap:16px;flex-wrap:wrap;margin:0 0 10px;font-size:12px;color:var(--fraco)}
.leg span{display:inline-flex;align-items:center;gap:6px}
.leg i{width:16px;height:0;border-top-width:2.5px;border-top-style:solid;display:inline-block}
.tres{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
.plot{position:relative;width:100%}
.plot svg{width:100%;display:block;overflow:visible}
.tip{position:absolute;pointer-events:none;opacity:0;transition:opacity .1s;
  background:var(--painel);border:1px solid var(--borda);border-radius:8px;
  padding:8px 10px;font-size:12px;box-shadow:0 6px 20px rgba(0,0,0,.18);
  font-variant-numeric:tabular-nums;white-space:nowrap;z-index:9}
.tip b{display:block;margin-bottom:5px;font-size:11px;color:var(--fraco);font-weight:600}
.tip div{display:flex;justify-content:space-between;gap:22px;line-height:1.7}
.tip div span:last-child{font-weight:600}
.tip i{width:9px;height:9px;border-radius:2px;display:inline-block;margin-right:5px}
.aviso{color:var(--fraco);text-align:center;padding:40px 0}
</style></head><body>
<h1>Detecção de defeitos, treino YOLOv8</h1>
<div class="sub" id="rodada">carregando...</div>

<div class="card">
  <div class="topo">
    <div><div class="rot">progresso</div><div class="pct" id="pct">--</div></div>
    <div style="flex:1"></div>
    <div class="chip sem_rodada" id="estado">aguardando</div>
  </div>
  <div class="barra"><div class="preenche" id="preenche"></div></div>
  <div class="rot" style="margin-top:9px" id="epocas">--</div>
</div>

<div class="grade" id="grade"></div>

<div class="card">
  <p class="tit">Precisão média (mAP) na validação interna</p>
  <div class="leg" id="leg-map"></div>
  <div class="plot" id="plot-map"><svg viewBox="0 0 1000 300" preserveAspectRatio="none"
    style="height:300px"></svg><div class="tip"></div></div>
</div>

<div class="card">
  <p class="tit">Precisão e revocação na validação interna</p>
  <div class="leg" id="leg-pr"></div>
  <div class="plot" id="plot-pr"><svg viewBox="0 0 1000 220" preserveAspectRatio="none"
    style="height:220px"></svg><div class="tip"></div></div>
</div>

<div class="tres">
  <div class="card"><p class="tit">Perda de caixa (box)</p>
    <div class="plot" id="plot-box"><svg viewBox="0 0 500 180" preserveAspectRatio="none"
      style="height:180px"></svg><div class="tip"></div></div></div>
  <div class="card"><p class="tit">Perda de classe (cls)</p>
    <div class="plot" id="plot-cls"><svg viewBox="0 0 500 180" preserveAspectRatio="none"
      style="height:180px"></svg><div class="tip"></div></div></div>
  <div class="card"><p class="tit">Perda de distribuição (dfl)</p>
    <div class="plot" id="plot-dfl"><svg viewBox="0 0 500 180" preserveAspectRatio="none"
      style="height:180px"></svg><div class="tip"></div></div></div>
</div>

<script>
const q=i=>document.getElementById(i);
const cor=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const num=(v,c=4)=>(v==null||!isFinite(v))?"--":v.toFixed(c).replace(".",",");
const hms=s=>{if(!isFinite(s)||s<0)return"--";s=Math.round(s);
  const h=Math.floor(s/3600),m=Math.floor(s%3600/60),g=s%60;
  return h?`${h}h ${String(m).padStart(2,"0")}min`:m?`${m}min ${String(g).padStart(2,"0")}s`:`${g}s`};
const hora=t=>new Date(t*1000).toLocaleTimeString("pt-BR");
const rots={treinando:"treinando",concluida:"concluída",parada:"parada",sem_rodada:"sem rodada"};

// Cada grafico guarda o que precisa para responder ao mouse: a escala usada e os
// pontos ja convertidos para coordenadas de tela. Sem isso o tooltip teria de
// refazer a projecao a cada movimento.
const graficos={};

function desenhar(id, serie, campos, opcoes={}){
  const box=q(id), svg=box.querySelector("svg");
  const vb=svg.getAttribute("viewBox").split(" ").map(Number);
  const W=vb[2], H=vb[3], E=44, D=26, T=10, B=24;
  if(!serie.length){svg.innerHTML="";return}

  const xs=serie.map(p=>p.e);
  const x0=Math.min(...xs), x1=Math.max(...xs);
  let y1=0;
  for(const c of campos) for(const p of serie) if(isFinite(p[c.k])) y1=Math.max(y1,p[c.k]);
  y1 = opcoes.desdeZero===false ? y1*1.05 : y1*1.08 || 1;
  const y0 = 0;
  const px=e=>E+(x1===x0?0:(e-x0)/(x1-x0))*(W-E-D);
  const py=v=>H-B-((v-y0)/(y1-y0||1))*(H-B-T);

  let g="";
  // grade e rotulos: recessivos, para os dados carregarem a figura
  const passos=4;
  for(let i=0;i<=passos;i++){
    const v=y0+(y1-y0)*i/passos, y=py(v);
    g+=`<line x1="${E}" y1="${y}" x2="${W-D}" y2="${y}" stroke="${cor("--borda")}" stroke-width="1"/>`;
    g+=`<text x="${E-7}" y="${y+4}" text-anchor="end" font-size="11"
         fill="${cor("--tenue")}">${num(v,y1<1?2:1)}</text>`;
  }
  const marcas=Math.min(6,Math.max(2,Math.floor((x1-x0)/10)));
  for(let i=0;i<=marcas;i++){
    const e=Math.round(x0+(x1-x0)*i/marcas);
    g+=`<text x="${px(e)}" y="${H-6}" text-anchor="middle" font-size="11"
         fill="${cor("--tenue")}">${e}</text>`;
  }
  for(const c of campos){
    const pts=serie.filter(p=>isFinite(p[c.k])).map(p=>`${px(p.e).toFixed(1)},${py(p[c.k]).toFixed(1)}`);
    g+=`<polyline points="${pts.join(" ")}" fill="none" stroke="${cor(c.cor)}"
         stroke-width="2" stroke-linejoin="round" stroke-dasharray="${c.traco||""}"
         vector-effect="non-scaling-stroke"/>`;
  }
  g+=`<line id="cruz-${id}" y1="${T}" y2="${H-B}" stroke="${cor("--tenue")}"
       stroke-width="1" stroke-dasharray="3 3" opacity="0"/>`;
  for(const c of campos)
    g+=`<circle id="pt-${id}-${c.k}" r="4.5" fill="${cor(c.cor)}" stroke="${cor("--painel")}"
         stroke-width="2" opacity="0"/>`;
  svg.innerHTML=g;
  graficos[id]={serie,campos,px,py,W,H,E,D};
}

function ligarMouse(id){
  const box=q(id), svg=box.querySelector("svg"), tip=box.querySelector(".tip");
  const mostrar=ev=>{
    const g=graficos[id]; if(!g||!g.serie.length)return;
    const r=svg.getBoundingClientRect();
    const escala=g.W/r.width;
    const xv=(ev.clientX-r.left)*escala;
    let melhor=g.serie[0], dist=Infinity;
    for(const p of g.serie){const d=Math.abs(g.px(p.e)-xv); if(d<dist){dist=d;melhor=p}}
    const cx=g.px(melhor.e);
    const cruz=svg.querySelector(`#cruz-${id}`);
    cruz.setAttribute("x1",cx); cruz.setAttribute("x2",cx); cruz.setAttribute("opacity","1");
    let linhas="";
    for(const c of g.campos){
      const pt=svg.querySelector(`#pt-${id}-${c.k}`);
      if(isFinite(melhor[c.k])){
        pt.setAttribute("cx",cx); pt.setAttribute("cy",g.py(melhor[c.k]));
        pt.setAttribute("opacity","1");
      } else pt.setAttribute("opacity","0");
      linhas+=`<div><span><i style="background:${cor(c.cor)}"></i>${c.rot}</span>
               <span>${num(melhor[c.k],c.casas??4)}</span></div>`;
    }
    tip.innerHTML=`<b>Época ${melhor.e}</b>${linhas}`;
    tip.style.opacity="1";
    const larg=tip.offsetWidth, esq=cx/escala;
    tip.style.left=Math.min(Math.max(esq+12,0),r.width-larg-4)+"px";
    tip.style.top="8px";
  };
  const esconder=()=>{
    const g=graficos[id]; if(!g)return;
    tip.style.opacity="0";
    const cruz=svg.querySelector(`#cruz-${id}`); if(cruz)cruz.setAttribute("opacity","0");
    for(const c of g.campos){
      const pt=svg.querySelector(`#pt-${id}-${c.k}`); if(pt)pt.setAttribute("opacity","0");
    }
  };
  box.addEventListener("mousemove",mostrar);
  box.addEventListener("mouseleave",esconder);
  box.addEventListener("touchmove",e=>{mostrar(e.touches[0]);e.preventDefault()},{passive:false});
}

function legenda(id,campos){
  q(id).innerHTML=campos.map(c=>
    `<span><i style="border-top-color:${cor(c.cor)};border-top-style:${c.traco?"dashed":"solid"}"></i>${c.rot}</span>`
  ).join("");
}

function celula(rot,val,extra){return `<div class="celula"><div class="rot">${rot}</div>
  <div class="val">${val}${extra?` <small>${extra}</small>`:""}</div></div>`}

const CAMPOS_MAP=[{k:"m50",rot:"mAP@0,5",cor:"--s1"},
                  {k:"m5095",rot:"mAP@0,5:0,95",cor:"--s2",traco:"7 4"}];
const CAMPOS_PR =[{k:"p",rot:"precisão",cor:"--s1"},
                  {k:"r",rot:"revocação",cor:"--s3",traco:"7 4"}];

let ligado=false;
async function tick(){
  let d; try{d=await (await fetch("/api/status",{cache:"no-store"})).json()}catch(e){return}
  if(d.estado==="sem_rodada"){q("rodada").textContent="nenhuma rodada encontrada";
    q("grade").innerHTML='<div class="aviso">Aguardando o treino começar.</div>';return}

  q("rodada").textContent=d.rodada;
  q("pct").textContent=d.pct.toFixed(1).replace(".",",")+"%";
  q("preenche").style.width=Math.min(d.pct,100)+"%";
  q("preenche").classList.toggle("concluida-barra",d.estado==="concluida");
  const c=q("estado"); c.className="chip "+d.estado; c.textContent=rots[d.estado]||d.estado;
  q("epocas").textContent=`época ${d.feitas} de ${d.alvo}`+
    (d.epochs_planejadas&&d.alvo<d.epochs_planejadas
      ? ` — limite dado pela paciência de ${d.paciencia}; teto planejado ${d.epochs_planejadas}`:"");

  const a=d.atual||{},b=d.melhor||{},g=d.gpu||{};
  const restam=Math.max((d.paciencia||0)-(d.desde_melhor||0),0);
  q("grade").innerHTML=[
    celula("começou às",hora(d.inicio)),
    celula("já rodou",hms(d.decorrido)),
    celula("falta",d.estado==="concluida"?"--":hms(d.restante_seg),
      d.estado==="concluida"?"":`${d.restantes} épocas`),
    celula("por época",d.por_epoca.toFixed(1).replace(".",",")+"s"),
    celula("melhor mAP@0,5",num(b.map50),`época ${b.epoca??"--"}`),
    celula("sem melhorar há",`${d.desde_melhor} ép.`,`paciência acaba em ${restam}`),
    celula("mAP@0,5 atual",num(a.map50)),
    celula("mAP@0,5:0,95",num(a.map5095)),
    celula("precisão",num(a.precisao)),
    celula("revocação",num(a.recall)),
    celula("perdas",num(a.box,2),`cls ${num(a.cls,2)} · dfl ${num(a.dfl,2)}`),
    celula("GPU",g.uso!=null?g.uso.toFixed(0)+"%":"--",
      g.vram_usada!=null?`${num(g.vram_usada,1)}/${num(g.vram_total||0,1)} GB${g.temp?` · ${g.temp.toFixed(0)}°C`:""}`:""),
  ].join("");

  const s=d.serie||[];
  legenda("leg-map",CAMPOS_MAP); legenda("leg-pr",CAMPOS_PR);
  desenhar("plot-map",s,CAMPOS_MAP);
  desenhar("plot-pr", s,CAMPOS_PR);
  desenhar("plot-box",s,[{k:"box",rot:"box",cor:"--s1",casas:3}]);
  desenhar("plot-cls",s,[{k:"cls",rot:"cls",cor:"--s2",casas:2}]);
  desenhar("plot-dfl",s,[{k:"dfl",rot:"dfl",cor:"--s3",casas:3}]);
  if(!ligado){["plot-map","plot-pr","plot-box","plot-cls","plot-dfl"].forEach(ligarMouse);ligado=true}
  document.title=`${d.pct.toFixed(0)}% · ép. ${d.feitas} · treino YOLOv8`;
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

from pathlib import Path as _P
RAIZ = _P(__file__).resolve().parents[2]
SAIDA = RAIZ / 'resultados' / 'analises'

import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, str(RAIZ / "src" / "treino"))
from compare_paired import DATASET, carrega_gt, casa, f1_por_carta, recall_por_carta
RUNS = DATASET / "runs_206_rev"
imgs = [Path(l.strip()) for l in open(DATASET / "folds_206_rev/fold4_avaliacao.txt") if l.strip()]
gt = carrega_gt("_206_rev", imgs)
def rodada(nome):
    lim = json.load(open(RUNS / nome / "resumo.json"))["conf_do_f1"]
    preds = defaultdict(list)
    for p in json.load(open(RUNS / f"{nome}_avaliacao/predictions.json")):
        if p["score"] >= lim:
            x, y, w, h = p["bbox"]; preds[p["image_id"]].append([x, y, x + w, y + h, p["score"]])
    pc = defaultdict(lambda: [0, 0, 0])
    for stem, (cx, _) in gt.items():
        c = stem.split("_frente")[0].split("_verso")[0]
        cas, fp = casa(np.array(preds.get(stem, [])).reshape(-1, 5), cx)
        pc[c][0] += int(cas.sum()); pc[c][1] += fp; pc[c][2] += int((~cas).sum())
    return pc
a, b, m = rodada("cfg2_fold4_1280"), rodada("cfg2_fold4_640"), rodada("cfg2_fold4_1280_m")
for nome, x in (("640", b), ("yolov8m", m)):
    for met, f in (("F1", f1_por_carta), ("recall", recall_por_carta)):
        ra, rb = f(a), f(x); cs = sorted(set(ra) & set(rb))
        xa = np.array([ra[c] for c in cs]); xb = np.array([rb[c] for c in cs]); d = xa - xb
        print(f"1280 vs {nome} {met}: n={len(cs)} {xa.mean():.3f} vs {xb.mean():.3f} melhor/pior/empate {(d>0).sum()}/{(d<0).sum()}/{(d==0).sum()} p={wilcoxon(xa, xb).pvalue:.2g}")

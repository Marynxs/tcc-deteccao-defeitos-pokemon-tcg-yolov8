from pathlib import Path as _P
RAIZ = _P(__file__).resolve().parents[2]
SAIDA = RAIZ / 'resultados' / 'analises'

import sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, str(RAIZ / "src" / "treino"))
import compare_paired as cp
from pathlib import Path
import json
# recall por carta so no estrato < 8 px @1280, cfg2 x cfg2+P2, 5 dobras retidas
def rodada(rot):
    cfg, resto = rot.split("_", 1)
    pc = defaultdict(lambda: [0, 0])
    for fold in range(5):
        nome = f"{cfg}_fold{fold}_{resto}"
        lim = json.load(open(cp.DATASET / "runs_206_rev" / nome / "resumo.json"))["conf_do_f1"]
        preds = defaultdict(list)
        for p in json.load(open(cp.DATASET / "runs_206_rev" / f"{nome}_avaliacao" / "predictions.json")):
            if p["score"] >= lim:
                x, y, w, h = p["bbox"]; preds[p["image_id"]].append([x, y, x + w, y + h, p["score"]])
        imgs = [Path(l.strip()) for l in open(cp.DATASET / "folds_206_rev" / f"fold{fold}_avaliacao.txt") if l.strip()]
        for stem, (cx, men) in cp.carrega_gt("_206_rev", imgs).items():
            c = stem.split("_frente")[0].split("_verso")[0]
            cas, _ = cp.casa(np.array(preds.get(stem, [])).reshape(-1, 5), cx)
            sel = men < 8
            pc[c][0] += int(cas[sel].sum()); pc[c][1] += int(sel.sum())
    return pc
a, b = rodada("cfg2_1280"), rodada("cfg2_1280_p2")
cartas = sorted(c for c in a if a[c][1] > 0)
ra = np.array([a[c][0] / a[c][1] for c in cartas]); rb = np.array([b[c][0] / b[c][1] for c in cartas])
d = rb - ra
print(f"cartas com defeito <8px: {len(cartas)}; total defeitos {sum(a[c][1] for c in cartas)}; achados cfg2 {sum(a[c][0] for c in cartas)} P2 {sum(b[c][0] for c in cartas)}")
print(f"recall medio por carta cfg2 {ra.mean():.3f} P2 {rb.mean():.3f}; P2 melhor/pior/empate {(d>0).sum()}/{(d<0).sum()}/{(d==0).sum()}")
print("wilcoxon p =", wilcoxon(rb, ra).pvalue if (d != 0).any() else 1.0)

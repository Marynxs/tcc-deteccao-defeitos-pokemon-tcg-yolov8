"""Fusao por tamanho: caixas da imagem inteira + so as caixas PEQUENAS das fatias (menor lado < T px na escala de 1280),
com T e limiar escolhidos na validacao interna; comparada carta a carta com a imagem inteira."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.stats import wilcoxon
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src" / "analise")); sys.path.insert(0, str(RAIZ / "src" / "treino"))
from sahi_eval import nms, contagens, f1_total, melhor_limiar, lista, SAIDA
from compare_paired import carrega_gt, FAIXAS

def carrega(f, papel, nome):
    d = json.load(open(SAIDA / f"fold{f}" / f"{papel}_{nome}.json"))
    return {k: np.array(v).reshape(-1, 5) for k, v in d.items()}

def funde(inteira, fatiada, imgs, T):
    out = {}
    for img in imgs:
        s = img.stem; w, h = Image.open(img).size; esc = 1280 / max(w, h)
        a = inteira.get(s, np.zeros((0, 5))); b = fatiada.get(s, np.zeros((0, 5)))
        menor = np.minimum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) * esc
        out[s] = nms(np.vstack([a, b[menor < T]]))
    return out

por_carta = {"inteira": defaultdict(lambda: [0, 0, 0]), "fusao": defaultdict(lambda: [0, 0, 0])}
faixas = {"inteira": {k[0]: [0, 0] for k in FAIXAS}, "fusao": {k[0]: [0, 0] for k in FAIXAS}}
for f in range(5):
    vi, av = lista(f, "val_interna"), lista(f, "avaliacao")
    gvi, gav = carrega_gt("_206_rev", vi), carrega_gt("_206_rev", av)
    ivi, fvi = carrega(f, "val_interna", "inteira"), carrega(f, "val_interna", "s640_i640")
    lim_i, _ = melhor_limiar(ivi, gvi)
    melhor = None
    for T in (8, 12, 16, 20):
        lim, f1 = melhor_limiar(funde(ivi, fvi, vi, T), gvi)
        if melhor is None or f1 > melhor[2]: melhor = (T, lim, f1)
    T, lim_f, _ = melhor
    iav, fav = carrega(f, "avaliacao", "inteira"), carrega(f, "avaliacao", "s640_i640")
    for rot, preds, lim in (("inteira", iav, lim_i), ("fusao", funde(iav, fav, av, T), lim_f)):
        for s, (tp, fp, fn, cas) in contagens(preds, gav, lim).items():
            c = s.split("_frente")[0].split("_verso")[0]; pc = por_carta[rot][c]; pc[0] += tp; pc[1] += fp; pc[2] += fn
            for nf, lo, hi in FAIXAS:
                sel = (gav[s][1] >= lo) & (gav[s][1] < hi); faixas[rot][nf][0] += int(cas[sel].sum()); faixas[rot][nf][1] += int(sel.sum())
    print(f"dobra {f+1}: T escolhido {T} px, limiar inteira {lim_i:.2f}, fusao {lim_f:.2f}")
for rot in ("inteira", "fusao"):
    pc = por_carta[rot]; tp = sum(v[0] for v in pc.values()); fp = sum(v[1] for v in pc.values()); fn = sum(v[2] for v in pc.values())
    print(rot, "TP", tp, "FP", fp, "rec %.3f prec %.3f" % (tp / (tp + fn), tp / (tp + fp)), {k: f"{v[0]}/{v[1]}" for k, v in faixas[rot].items()})
F = {r: {c: 2 * a / (2 * a + b + d) for c, (a, b, d) in por_carta[r].items() if a + d > 0} for r in por_carta}
cs = sorted(F["inteira"]); x = np.array([F["fusao"][c] for c in cs]); y = np.array([F["inteira"][c] for c in cs]); d = x - y
print("F1 por carta fusao x inteira: %.3f x %.3f, melhor %d pior %d empate %d, p=%.3g" % (x.mean(), y.mean(), (d > 0).sum(), (d < 0).sum(), (d == 0).sum(), wilcoxon(x, y).pvalue))

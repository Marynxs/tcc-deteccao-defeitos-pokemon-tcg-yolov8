"""Comparacao pareada por carta entre rodadas da campanha (Wilcoxon) e recall
por faixa de tamanho, a partir dos predictions.json das dobras retidas."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import wilcoxon

RAIZ = Path(__file__).resolve().parents[2]
DATASET = RAIZ / "Dataset_YOLO"
IOU_MIN = 0.5
# estratos pela dimensao menor da caixa apos letterbox a 1280 (auditor-de-dados)
FAIXAS = [("<8 px", 0, 8), ("8 a 20 px", 8, 20), (">20 px", 20, 1e9)]


def iou_matriz(a, b):
    # a, b em xyxy; devolve matriz |a| x |b|
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def carrega_gt(pool_suf, imagens):
    # devolve {stem: (gt xyxy em px, dim menor @1280 por caixa)}
    gt = {}
    for img in imagens:
        stem = img.stem
        w, h = Image.open(img).size
        escala = 1280 / max(w, h)
        rot = DATASET / f"dataset{pool_suf}" / "labels" / f"{stem}.txt"
        caixas, menores = [], []
        if rot.is_file():
            for linha in rot.read_text().split("\n"):
                p = linha.split()
                if len(p) < 5:
                    continue
                cx, cy, bw, bh = (float(x) for x in p[1:5])
                caixas.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
                menores.append(min(bw * w, bh * h) * escala)
        gt[stem] = (np.array(caixas).reshape(-1, 4), np.array(menores))
    return gt


def casa(pred, gt):
    # casamento guloso por confianca; devolve mascara de gt casados e n de fp
    if len(pred) == 0:
        return np.zeros(len(gt), bool), 0
    ordem = np.argsort(-pred[:, 4])
    m = iou_matriz(pred[ordem, :4], gt)
    casado = np.zeros(len(gt), bool)
    fp = 0
    for i in range(len(ordem)):
        if len(gt) == 0:
            fp += 1
            continue
        cand = np.where(~casado)[0]
        if len(cand) == 0:
            fp += 1
            continue
        j = cand[np.argmax(m[i, cand])]
        if m[i, j] >= IOU_MIN:
            casado[j] = True
        else:
            fp += 1
    return casado, fp


def avalia_rodada(rotulo, pool_suf, runs):
    # rotulo ex.: cfg2_1280 ou cfg2_1280_p2; devolve por carta e por faixa
    por_carta = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn
    por_faixa = {f[0]: [0, 0] for f in FAIXAS}  # recuperados, total
    cfg, resto = rotulo.split("_", 1)
    for fold in range(5):
        nome = f"{cfg}_fold{fold}_{resto}"
        resumo = json.load(open(runs / nome / "resumo.json"))
        limiar = resumo["conf_do_f1"]
        preds = defaultdict(list)
        for p in json.load(open(runs / f"{nome}_avaliacao" / "predictions.json")):
            if p["score"] < limiar:
                continue
            x, y, w, h = p["bbox"]
            preds[p["image_id"]].append([x, y, x + w, y + h, p["score"]])
        imagens = [Path(l.strip()) for l in open(DATASET / f"folds{pool_suf}" / f"fold{fold}_avaliacao.txt") if l.strip()]
        gt = carrega_gt(pool_suf, imagens)
        for stem, (caixas, menores) in gt.items():
            carta = stem.split("_frente")[0].split("_verso")[0]
            pr = np.array(preds.get(stem, [])).reshape(-1, 5)
            casado, fp = casa(pr, caixas)
            por_carta[carta][0] += int(casado.sum())
            por_carta[carta][1] += fp
            por_carta[carta][2] += int((~casado).sum())
            for nome_f, lo, hi in FAIXAS:
                sel = (menores >= lo) & (menores < hi)
                por_faixa[nome_f][0] += int(casado[sel].sum())
                por_faixa[nome_f][1] += int(sel.sum())
    return por_carta, por_faixa


def f1_por_carta(pc):
    out = {}
    for carta, (tp, fp, fn) in pc.items():
        if tp + fn == 0:
            continue  # carta sem anotacao: escore indefinido
        out[carta] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return out


def recall_por_carta(pc):
    return {c: tp / (tp + fn) for c, (tp, fp, fn) in pc.items() if tp + fn > 0}


def compara(a, b, ra, rb, metrica, nome):
    cartas = sorted(set(ra) & set(rb))
    xa = np.array([ra[c] for c in cartas])
    xb = np.array([rb[c] for c in cartas])
    d = xa - xb
    n_dif = int((d != 0).sum())
    p = wilcoxon(xa, xb).pvalue if n_dif > 0 else 1.0
    print(f"  {nome:7s} {a} vs {b}: n={len(cartas)} media {xa.mean():.3f} vs {xb.mean():.3f}, "
          f"mediana da dif {np.median(d):+.3f}, {a} melhor em {(d > 0).sum()}, pior em {(d < 0).sum()}, "
          f"empate {(d == 0).sum()}, Wilcoxon p={p:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_206_rev")
    ap.add_argument("--pool-suf", default="_206_rev")
    ap.add_argument("--rodadas", nargs="+", default=["cfg1_1280", "cfg2_1280", "cfg3_1280", "cfg4_1280", "cfg2_1280_p2"])
    ap.add_argument("--pares", nargs="+", default=["cfg2_1280:cfg2_1280_p2", "cfg2_1280:cfg1_1280", "cfg2_1280:cfg4_1280", "cfg2_1280:cfg3_1280", "cfg1_1280:cfg3_1280", "cfg4_1280:cfg3_1280"])
    args = ap.parse_args()
    runs = DATASET / args.runs
    res = {}
    print("== recall por faixa de tamanho (dimensao menor @1280), 5 dobras retidas, limiar da validacao interna ==")
    for r in args.rodadas:
        pc, pf = avalia_rodada(r, args.pool_suf, runs)
        res[r] = pc
        tp = sum(v[0] for v in pc.values()); fp = sum(v[1] for v in pc.values()); fn = sum(v[2] for v in pc.values())
        faixas = "  ".join(f"{k}: {v[0]}/{v[1]} = {v[0] / max(v[1], 1):.3f}" for k, v in pf.items())
        print(f"  {r:13s} TP {tp:5d} FP {fp:5d} FN {fn:5d} | recall {tp / (tp + fn):.3f} precisao {tp / max(tp + fp, 1):.3f} | {faixas}")
    print("\n== comparacao pareada por carta (Wilcoxon bilateral) ==")
    for par in args.pares:
        a, b = par.split(":")
        compara(a, b, f1_por_carta(res[a]), f1_por_carta(res[b]), "f1", "F1")
        compara(a, b, recall_por_carta(res[a]), recall_por_carta(res[b]), "recall", "recall")


if __name__ == "__main__":
    main()

"""Inferencia fatiada (SAHI, Akyon et al. 2022) sobre os best.pt da Configuracao 2, comparada
pareada por carta com a inferencia na carta inteira, com o mesmo protocolo da campanha."""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import wilcoxon

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src" / "treino"))
from compare_paired import FAIXAS, carrega_gt, casa  # noqa: E402

DATASET = RAIZ / "Dataset_YOLO"
RUNS = DATASET / "runs_206_rev"
SAIDA = DATASET / "runs_sahi"

# Candidatos de fatiamento: (lado da fatia em pixels da imagem original, imgsz da fatia).
# A carta tem ~1330 x 1830 px e o treino a viu reduzida a 1280 no lado maior (escala ~0,7).
# Fatia 640 a 640 = escala 1,0; fatia 896 a 1280 = 1,43; fatia 640 a 1280 = 2,0.
# Qual usar e decidido na validacao interna de cada dobra, nunca na dobra retida.
CANDIDATOS = [(640, 640), (896, 1280), (640, 1280)]
SOBREPOSICAO = 0.2
LIMIARES = np.round(np.arange(0.01, 0.91, 0.01), 2)


def janelas(w, h, lado):
    # fatias de lado x lado com 20% de sobreposicao; a ultima encosta na borda
    passo = int(lado * (1 - SOBREPOSICAO))
    xs = list(range(0, max(w - lado, 0) + 1, passo))
    ys = list(range(0, max(h - lado, 0) + 1, passo))
    if xs[-1] + lado < w:
        xs.append(w - lado)
    if ys[-1] + lado < h:
        ys.append(h - lado)
    return [(x, y, min(x + lado, w), min(y + lado, h)) for y in ys for x in xs]


def caixas(res, dx=0, dy=0):
    b = res.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, 5))
    xyxy = b.xyxy.cpu().numpy() + np.array([dx, dy, dx, dy])
    return np.hstack([xyxy, b.conf.cpu().numpy()[:, None]])


def nms(pred, limiar=0.5):
    # supressao gulosa por IoU entre as caixas da carta inteira e das fatias. A fusao
    # GREEDYNMM do SAHI une caixas e so funciona depois do limiar de confianca; aqui as
    # predicoes ficam com confianca 0,001 para o limiar ser escolhido depois, e a uniao
    # deixava uma caixa grande e fraca engolir as pequenas. A NMS gulosa da o mesmo
    # resultado filtrando antes ou depois, porque so a caixa mais confiante suprime.
    if len(pred) == 0:
        return pred
    pred = pred[np.argsort(-pred[:, 4])]
    area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    vivo = np.ones(len(pred), bool)
    manter = []
    for i in range(len(pred)):
        if not vivo[i]:
            continue
        manter.append(i)
        ix1 = np.maximum(pred[i, 0], pred[:, 0]); iy1 = np.maximum(pred[i, 1], pred[:, 1])
        ix2 = np.minimum(pred[i, 2], pred[:, 2]); iy2 = np.minimum(pred[i, 3], pred[:, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        iou = inter / (area[i] + area - inter + 1e-9)
        vivo &= iou < limiar
    return pred[manter]


def prever(modelo, caminho, modo, device):
    # modo = None (carta inteira) ou (lado, imgsz); parametros de inferencia iguais aos da validacao
    img = cv2.imread(str(caminho))
    kw = dict(conf=0.001, iou=0.7, max_det=300, verbose=False, device=device)
    inteira = caixas(modelo.predict(img, imgsz=1280, **kw)[0])
    if modo is None:
        return inteira
    lado, imgsz = modo
    h, w = img.shape[:2]
    partes = [inteira]
    for (x1, y1, x2, y2) in janelas(w, h, lado):
        partes.append(caixas(modelo.predict(img[y1:y2, x1:x2], imgsz=imgsz, **kw)[0], x1, y1))
    return nms(np.vstack(partes))


def lista(fold, papel):
    return [Path(l.strip()) for l in open(DATASET / "folds_206_rev" / f"fold{fold}_{papel}.txt") if l.strip()]


def contagens(preds, gt, limiar):
    # tp, fp, fn por imagem, no limiar dado
    out = {}
    for stem, (cx, _) in gt.items():
        p = preds.get(stem, np.zeros((0, 5)))
        p = p[p[:, 4] >= limiar]
        casado, fp = casa(p, cx)
        out[stem] = (int(casado.sum()), fp, int((~casado).sum()), casado)
    return out


def f1_total(cont):
    tp = sum(v[0] for v in cont.values()); fp = sum(v[1] for v in cont.values()); fn = sum(v[2] for v in cont.values())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def melhor_limiar(preds, gt):
    f1s = [f1_total(contagens(preds, gt, t)) for t in LIMIARES]
    i = int(np.argmax(f1s))
    return float(LIMIARES[i]), float(f1s[i])


def predicoes(modelo, imagens, modo, device, cache):
    if cache.is_file():
        d = json.load(open(cache))
        return {k: np.array(v).reshape(-1, 5) for k, v in d.items()}
    d = {}
    for img in imagens:
        d[img.stem] = prever(modelo, img, modo, device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    json.dump({k: v.tolist() for k, v in d.items()}, open(cache, "w"))  # sem arredondar: empates no limiar mudam a contagem
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-imagens", type=int, default=0, help="so para teste rapido")
    args = ap.parse_args()
    from ultralytics import YOLO

    modos = {"inteira": None, **{f"s{l}_i{i}": (l, i) for l, i in CANDIDATOS}}
    por_carta = {"inteira": defaultdict(lambda: [0, 0, 0]), "sahi": defaultdict(lambda: [0, 0, 0])}
    faixas = {"inteira": {f[0]: [0, 0] for f in FAIXAS}, "sahi": {f[0]: [0, 0] for f in FAIXAS}}
    relatorio = []
    for fold in args.folds:
        t0 = time.time()
        modelo = YOLO(str(RUNS / f"cfg2_fold{fold}_1280" / "weights" / "best.pt"))
        vi, av = lista(fold, "val_interna"), lista(fold, "avaliacao")
        if args.max_imagens:
            vi, av = vi[:args.max_imagens], av[:args.max_imagens]
        gt_vi, gt_av = carrega_gt("_206_rev", vi), carrega_gt("_206_rev", av)
        sufixo = "_teste" if args.max_imagens else ""
        # 1) na validacao interna: limiar de cada modo e escolha do fatiamento
        escolha = {}
        for nome, modo in modos.items():
            p = predicoes(modelo, vi, modo, args.device, SAIDA / f"fold{fold}{sufixo}" / f"val_interna_{nome}.json")
            escolha[nome] = melhor_limiar(p, gt_vi)
            print(f"dobra {fold + 1} val. interna {nome:12s} limiar {escolha[nome][0]:.2f} F1 {escolha[nome][1]:.3f}", flush=True)
        fatiado = max((n for n in modos if n != "inteira"), key=lambda n: escolha[n][1])
        # 2) na dobra retida: inteira x fatiado escolhido, cada um no proprio limiar da val. interna
        linha = {"dobra": fold + 1, "fatiamento": fatiado}
        for rot, nome in [("inteira", "inteira"), ("sahi", fatiado)]:
            p = predicoes(modelo, av, modos[nome], args.device, SAIDA / f"fold{fold}{sufixo}" / f"avaliacao_{nome}.json")
            cont = contagens(p, gt_av, escolha[nome][0])
            linha[f"limiar_{rot}"] = escolha[nome][0]
            linha[f"f1_{rot}"] = round(f1_total(cont), 4)
            for stem, (tp, fp, fn, casado) in cont.items():
                carta = stem.split("_frente")[0].split("_verso")[0]
                pc = por_carta[rot][carta]
                pc[0] += tp; pc[1] += fp; pc[2] += fn
                menores = gt_av[stem][1]
                for nf, lo, hi in FAIXAS:
                    sel = (menores >= lo) & (menores < hi)
                    faixas[rot][nf][0] += int(casado[sel].sum()); faixas[rot][nf][1] += int(sel.sum())
        linha["segundos"] = round(time.time() - t0)
        relatorio.append(linha)
        print("  ", linha, flush=True)

    print("\n== total nas dobras retidas ==")
    f1c = {}
    for rot in ("inteira", "sahi"):
        pc = por_carta[rot]
        tp = sum(v[0] for v in pc.values()); fp = sum(v[1] for v in pc.values()); fn = sum(v[2] for v in pc.values())
        fx = "  ".join(f"{k}: {v[0]}/{v[1]} = {v[0] / max(v[1], 1):.3f}" for k, v in faixas[rot].items())
        print(f"  {rot:8s} TP {tp} FP {fp} FN {fn} | recall {tp / max(tp + fn, 1):.3f} precisao {tp / max(tp + fp, 1):.3f} | {fx}")
        f1c[rot] = {c: 2 * a / (2 * a + b + d) for c, (a, b, d) in pc.items() if a + d > 0}
        rc = {c: a / (a + d) for c, (a, b, d) in pc.items() if a + d > 0}
        f1c[rot + "_rec"] = rc
    for met, a, b in [("F1", "sahi", "inteira"), ("recall", "sahi_rec", "inteira_rec")]:
        cartas = sorted(set(f1c[a]) & set(f1c[b]))
        xa = np.array([f1c[a][c] for c in cartas]); xb = np.array([f1c[b][c] for c in cartas])
        d = xa - xb
        p = wilcoxon(xa, xb).pvalue if (d != 0).any() else 1.0
        print(f"  {met} por carta, fatiado x inteira: n={len(cartas)} media {xa.mean():.3f} x {xb.mean():.3f}, "
              f"melhor em {(d > 0).sum()}, pior em {(d < 0).sum()}, empate {(d == 0).sum()}, Wilcoxon bilateral p={p:.4g}")
    if not args.max_imagens:
        json.dump(relatorio, open(SAIDA / "relatorio.json", "w"), indent=1)


if __name__ == "__main__":
    main()

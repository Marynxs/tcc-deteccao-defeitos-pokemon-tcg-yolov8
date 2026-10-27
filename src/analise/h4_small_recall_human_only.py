"""Revocacao por faixa de tamanho (dimensao menor @1280) das quatro configuracoes e da config. 2 com P2,
na anotacao completa e na anotacao sem as caixas sugeridas, a partir das predicoes salvas."""
from pathlib import Path as _P
RAIZ = _P(__file__).resolve().parents[2]
SAIDA = RAIZ / 'resultados' / 'analises'

import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.stats import wilcoxon

D = RAIZ / 'Dataset_YOLO'
RUNS = D / 'runs_206_rev'
ORIG = json.load(open(SAIDA / 'origem_caixas.json'))
FAIXAS = [("<8 px", 0, 8), ("8 a 20 px", 8, 20), (">20 px", 20, 1e9)]


def iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def gt(img):
    # caixas em px, dimensao menor @1280 e origem de cada caixa
    w, h = Image.open(img).size
    esc = 1280 / max(w, h)
    bx, men = [], []
    for l in (D / 'dataset_206_rev/labels' / f'{img.stem}.txt').read_text().splitlines():
        p = l.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        bx.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        men.append(min(bw * w, bh * h) * esc)
    return np.array(bx).reshape(-1, 4), np.array(men), np.array(ORIG[img.stem])


def casa(pred, g):
    # mesmo casamento guloso de compare_paired.py; para a revocacao so importa quais
    # anotacoes foram casadas, entao as predicoes ignoradas (sobre caixa sugerida) nao mudam nada
    casado = np.zeros(len(g), bool)
    if len(pred) == 0:
        return casado
    P = pred[np.argsort(-pred[:, 4])]
    m = iou(P[:, :4], g)
    for i in range(len(P)):
        cand = np.where(~casado)[0]
        if len(cand):
            j = cand[np.argmax(m[i, cand])]
            if m[i, j] >= 0.5:
                casado[j] = True
    return casado


def rodada(rot, so_humano, rot_limiar=None):
    # rot_limiar: rodada cujo limiar e aplicado a esta (limiar comum aos dois modelos)
    cfg, resto = rot.split('_', 1)
    faixa = {f[0]: [0, 0] for f in FAIXAS}
    carta = defaultdict(lambda: [0, 0])  # estrato < 8 px por carta
    for k in range(5):
        nome = f'{cfg}_fold{k}_{resto}'
        nome_lim = nome if rot_limiar is None else rot_limiar.replace('_', f'_fold{k}_', 1)
        lim = json.load(open(RUNS / nome_lim / 'resumo.json'))['conf_do_f1']
        preds = defaultdict(list)
        for p in json.load(open(RUNS / f'{nome}_avaliacao' / 'predictions.json')):
            if p['score'] >= lim:
                x, y, w, h = p['bbox']; preds[p['image_id']].append([x, y, x + w, y + h, p['score']])
        for l in open(D / 'folds_206_rev' / f'fold{k}_avaliacao.txt'):
            if not l.strip():
                continue
            img = Path(l.strip())
            g, men, o = gt(img)
            sel = (o != 'prop') if so_humano else np.ones(len(o), bool)
            g, men = g[sel], men[sel]
            cas = casa(np.array(preds.get(img.stem, [])).reshape(-1, 5), g)
            for nf, lo, hi in FAIXAS:
                s = (men >= lo) & (men < hi)
                faixa[nf][0] += int(cas[s].sum()); faixa[nf][1] += int(s.sum())
            s = men < 8
            c = img.stem[:9]
            carta[c][0] += int(cas[s].sum()); carta[c][1] += int(s.sum())
    return faixa, carta


linhas = []
for so_humano, nome_ref in ((False, 'anotacao completa'), (True, 'anotacao sem as caixas sugeridas')):
    linhas.append(f'== {nome_ref} ==')
    res = {r: rodada(r, so_humano) for r in ('cfg1_1280', 'cfg2_1280', 'cfg3_1280', 'cfg4_1280', 'cfg2_1280_p2')}
    for r, (fx, _) in res.items():
        linhas.append(f'  {r:14s} ' + '  '.join(f'{k}: {a}/{b} = {a / max(b, 1):.3f}' for k, (a, b) in fx.items()))
    a, b = res['cfg2_1280'][1], res['cfg2_1280_p2'][1]
    cs = sorted(c for c in a if a[c][1] > 0)
    ra = np.array([a[c][0] / a[c][1] for c in cs]); rb = np.array([b[c][0] / b[c][1] for c in cs]); d = rb - ra
    p = wilcoxon(rb, ra).pvalue if (d != 0).any() else 1.0
    linhas.append(f'  por carta (<8 px): n={len(cs)} recall medio cfg2 {ra.mean():.3f} P2 {rb.mean():.3f}; '
                  f'P2 melhor/pior/empate {(d > 0).sum()}/{(d < 0).sum()}/{(d == 0).sum()}; Wilcoxon p = {p:.3g}')
txt = '\n'.join(linhas)
print(txt)
(SAIDA / 'saida_h4_dois_gabaritos.txt').write_text(txt + '\n')

# Descritivo: revocacao por faixa com limiar comum aos dois modelos, para separar o efeito
# da P2 do efeito de um limiar mais baixo (o limiar de cada rodada vem da propria validacao interna)
linhas = []
for so_humano, nome_ref in ((False, 'anotacao completa'), (True, 'anotacao sem as caixas sugeridas')):
    linhas.append(f'== {nome_ref} ==')
    for base in ('cfg2_1280', 'cfg2_1280_p2'):
        linhas.append(f'  limiar de cada dobra tirado de {base}:')
        for r in ('cfg2_1280', 'cfg2_1280_p2'):
            fx, _ = rodada(r, so_humano, base)
            linhas.append(f'    {r:14s} ' + '  '.join(f'{k}: {a}/{b} = {a / max(b, 1):.3f}' for k, (a, b) in fx.items()))
lims = [(json.load(open(RUNS / f'cfg2_fold{k}_1280' / 'resumo.json'))['conf_do_f1'],
         json.load(open(RUNS / f'cfg2_fold{k}_1280_p2' / 'resumo.json'))['conf_do_f1']) for k in range(5)]
linhas.append('limiares por dobra (sem P2, com P2): ' + '; '.join(f'Dobra {k + 1} {a:.4f} {b:.4f}' for k, (a, b) in enumerate(lims)))
txt = '\n'.join(linhas)
print(txt)
(SAIDA / 'saida_h4_limiar_comum.txt').write_text(txt + '\n')

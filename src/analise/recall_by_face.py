"""Revocacao da config. 2 por face da carta (frente e verso), na anotacao completa, com o limiar
da validacao interna de cada rodada e o mesmo casamento de h4_small_recall_human_only.py."""
from pathlib import Path
import json
from collections import defaultdict
import numpy as np

import importlib.util
RAIZ = Path(__file__).resolve().parents[2]
# reaproveita gt() e casa() sem executar o resto do script de H4
fonte = (RAIZ / 'src/analise/h4_small_recall_human_only.py').read_text().split('\ndef rodada(')[0]
ns = {'__file__': str(RAIZ / 'src/analise/h4_small_recall_human_only.py')}
exec(fonte, ns)
D, RUNS, SAIDA = ns['D'], ns['RUNS'], ns['SAIDA']

face = {'frente': [0, 0], 'verso': [0, 0]}
for k in range(5):
    nome = f'cfg2_fold{k}_1280'
    lim = json.load(open(RUNS / nome / 'resumo.json'))['conf_do_f1']
    preds = defaultdict(list)
    for p in json.load(open(RUNS / f'{nome}_avaliacao' / 'predictions.json')):
        if p['score'] >= lim:
            x, y, w, h = p['bbox']; preds[p['image_id']].append([x, y, x + w, y + h, p['score']])
    for l in open(D / 'folds_206_rev' / f'fold{k}_avaliacao.txt'):
        if not l.strip():
            continue
        img = Path(l.strip())
        g, men, o = ns['gt'](img)
        cas = ns['casa'](np.array(preds.get(img.stem, [])).reshape(-1, 5), g)
        f = 'frente' if 'frente' in img.stem else 'verso'
        face[f][0] += int(cas.sum()); face[f][1] += len(g)
txt = '\n'.join(f'{f}: {a}/{b} = {a / b:.3f}' for f, (a, b) in face.items())
print(txt)
(SAIDA / 'saida_recall_por_face.txt').write_text(txt + '\n')

"""Resumo descritivo dos erros da config. 2 (anotacao completa, limiar da validacao interna):
cartas com ao menos um defeito encontrado e metricas agregadas sem as cartas de desgaste difuso."""
from pathlib import Path
import csv
import json
from collections import defaultdict
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
# reaproveita gt() e casa() sem executar o resto do script de H4
fonte = (RAIZ / 'src/analise/h4_small_recall_human_only.py').read_text().split('\ndef rodada(')[0]
ns = {'__file__': str(RAIZ / 'src/analise/h4_small_recall_human_only.py')}
exec(fonte, ns)
D, RUNS, SAIDA = ns['D'], ns['RUNS'], ns['SAIDA']

# cartas com alguma face marcada como desgaste difuso na revisao da anotacao
difusas = sorted({l['imagem'][:9] for l in csv.DictReader(open(D / 'revisao_206/superficie.csv'))
                  if l['desgaste_difuso'] == '1'})

carta = defaultdict(lambda: [0, 0, 0])  # tp, fp, n anotacoes
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
        g, _, _ = ns['gt'](img)
        pr = np.array(preds.get(img.stem, [])).reshape(-1, 5)
        tp = int(ns['casa'](pr, g).sum())
        c = carta[img.stem[:9]]
        c[0] += tp; c[1] += len(pr) - tp; c[2] += len(g)


def agrega(cartas):
    tp = sum(carta[c][0] for c in cartas); fp = sum(carta[c][1] for c in cartas)
    fn = sum(carta[c][2] for c in cartas) - tp
    p, r = tp / (tp + fp), tp / (tp + fn)
    return f'TP {tp} FP {fp} FN {fn} precisao {p:.3f} revocacao {r:.3f} F1 {2 * p * r / (p + r):.3f}'


todas = sorted(carta)
com_anot = [c for c in todas if carta[c][2] > 0]
achou = [c for c in com_anot if carta[c][0] > 0]
linhas = [f'cartas avaliadas: {len(todas)}; com anotacao: {len(com_anot)}; com ao menos um defeito encontrado: {len(achou)}',
          f'sem nenhum defeito encontrado: {sorted(set(com_anot) - set(achou))}',
          f'cartas de desgaste difuso ({len(difusas)}): {difusas}',
          f'todas as cartas: {agrega(todas)}',
          f'sem as de desgaste difuso ({len(todas) - len(difusas)} cartas): {agrega([c for c in todas if c not in difusas])}']
txt = '\n'.join(linhas)
print(txt)
(SAIDA / 'saida_resumo_erros.txt').write_text(txt + '\n')

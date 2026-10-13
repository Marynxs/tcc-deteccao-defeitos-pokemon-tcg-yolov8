# bootstrap por carta (dobra 2) do AP@0,5 dos pontos da curva, pareado no mesmo conjunto de teste
from pathlib import Path as _P
RAIZ = _P(__file__).resolve().parents[2]
SAIDA = RAIZ / 'resultados' / 'analises'

import json, collections, sys
from pathlib import Path
import numpy as np
sys.argv=['x','0.5']
exec(open(_P(__file__).with_name('reevaluate_human_only.py')).read().split("MODOS={")[0])  # reaproveita iou, gt, casa, ap
RUNS=D/'runs_206_rev'
imgs=[Path(l.strip()) for l in open(D/'folds_206_rev/fold2_avaliacao.txt') if l.strip()]
cartas=sorted({i.stem[:9] for i in imgs})
names={'25':'cfg2_fold2_1280_curva25','50':'cfg2_fold2_1280_curva50','75':'cfg2_fold2_1280_curva75','100':'cfg2_fold2_1280'}
per={}  # nome -> carta -> (lista (score,tp), npos)
for k,n in names.items():
    allp=collections.defaultdict(list)
    for p in json.load(open(RUNS/f'{n}_avaliacao/predictions.json')):
        x,y,w,h=p['bbox']; allp[p['image_id']].append([x,y,x+w,y+h,p['score']])
    d=collections.defaultdict(lambda:[[],0])
    for img in imgs:
        g,o=gt(img); pa=np.array(allp.get(img.stem,[])).reshape(-1,5)
        _,_,lst=casa(pa,g,np.zeros((0,4)))
        d[img.stem[:9]][0]+=lst; d[img.stem[:9]][1]+=len(g)
    per[k]=d
def AP(k,sample):
    lst=[t for c in sample for t in per[k][c][0]]; npos=sum(per[k][c][1] for c in sample)
    return ap(lst,npos)
for k in names: print(k, round(AP(k,cartas),4))
rng=np.random.default_rng(0); B=2000
pares=[('50','25'),('75','50'),('100','75'),('100','50')]
diffs={p:[] for p in pares}; vals={k:[] for k in names}
for b in range(B):
    s=list(rng.choice(cartas,len(cartas),replace=True))
    a={k:AP(k,s) for k in names}
    for k in names: vals[k].append(a[k])
    for x,y in pares: diffs[(x,y)].append(a[x]-a[y])
for k in names: print(f'{k}: IC95 {np.percentile(vals[k],2.5):.3f} a {np.percentile(vals[k],97.5):.3f}, dp {np.std(vals[k]):.3f}')
for p,v in diffs.items():
    v=np.array(v); print(f'{p[0]} - {p[1]}: media {v.mean():+.3f} IC95 [{np.percentile(v,2.5):+.3f}, {np.percentile(v,97.5):+.3f}] P(dif<=0) {np.mean(v<=0):.3f}')

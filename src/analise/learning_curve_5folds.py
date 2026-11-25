# curva de aprendizado nas cinco dobras: AP@0,5 das 206 cartas com bootstrap por carta e Wilcoxon por carta (F1 e recall), com Holm
from pathlib import Path as _P
RAIZ = _P(__file__).resolve().parents[2]

import json, collections, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
sys.argv=['x','0.5']
exec(open(_P(__file__).with_name('reevaluate_human_only.py')).read().split("MODOS={")[0])  # reaproveita iou, gt, casa, ap
sys.path.insert(0,str(RAIZ/'src'/'treino'))
import compare_paired as cp
FR={'25':'_curva25','50':'_curva50','75':'_curva75','100':''}
per={}
for k,suf in FR.items():
    d=collections.defaultdict(lambda:[[],0])
    for f in range(5):
        n=f'cfg2_fold{f}_1280{suf}'
        allp=collections.defaultdict(list)
        for p in json.load(open(RUNS/f'{n}_avaliacao/predictions.json')):
            x,y,w,h=p['bbox']; allp[p['image_id']].append([x,y,x+w,y+h,p['score']])
        imgs=[Path(l.strip()) for l in open(D/f'folds_206_rev/fold{f}_avaliacao.txt') if l.strip()]
        for img in imgs:
            g,o=gt(img); pa=np.array(allp.get(img.stem,[])).reshape(-1,5)
            _,_,lst=casa(pa,g,np.zeros((0,4)))
            c=img.stem.split('_frente')[0].split('_verso')[0]
            d[c][0]+=lst; d[c][1]+=len(g)
    per[k]=d
cartas=sorted(per['100'])
print('n cartas',len(cartas))
def AP(k,s): return ap([t for c in s for t in per[k][c][0]],sum(per[k][c][1] for c in s))
for k in FR: print(f'AP@0,5 agregado {k}%: {AP(k,cartas):.4f}')
rng=np.random.default_rng(0); B=2000
pares=[('50','25'),('75','50'),('100','75'),('100','50')]
dif={p:[] for p in pares}; pts={k:[] for k in FR}
for b in range(B):
    s=list(rng.choice(cartas,len(cartas),replace=True)); a={k:AP(k,s) for k in FR}
    for x,y in pares: dif[(x,y)].append(a[x]-a[y])
    for k in FR: pts[k].append(a[k])
for k,v in pts.items(): print(f'AP {k}% IC95 [{np.percentile(v,2.5):.3f}; {np.percentile(v,97.5):.3f}]')
for p,v in dif.items():
    v=np.array(v); print(f'AP {p[0]}-{p[1]}: {v.mean():+.3f} IC95 [{np.percentile(v,2.5):+.3f}; {np.percentile(v,97.5):+.3f}]')
f1={};rc={}
for k,suf in FR.items():
    pc,_=cp.avalia_rodada('cfg2_1280'+suf,'_206_rev',cp.DATASET/'runs_206_rev')
    f1[k]=cp.f1_por_carta(pc); rc[k]=cp.recall_por_carta(pc)
for nome,M in [('F1',f1),('recall',rc)]:
    print(f'--- {nome} por carta')
    for k in FR:
        print(f'  {k}%: media {np.mean([M[k][c] for c in cartas if c in M[k]]):.4f}')
    ps=[]
    for x,y in pares:
        cs=[c for c in cartas if c in M[x] and c in M[y]]
        dd=np.array([M[x][c]-M[y][c] for c in cs]); p=wilcoxon(dd).pvalue; ps.append(p)
        print(f'  {x} x {y}: n {len(cs)} dif media {dd.mean():+.4f} melhor {int((dd>0).sum())} pior {int((dd<0).sum())} empate {int((dd==0).sum())} p {p:.2g}')
    print('  Holm:',cp.holm(ps))

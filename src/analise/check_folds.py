from pathlib import Path as _P
RAIZ = _P(__file__).resolve().parents[2]
SAIDA = RAIZ / 'resultados' / 'analises'

import json, collections
from pathlib import Path
D=RAIZ/'Dataset_YOLO'/'folds_206_rev'
G=json.load(open(RAIZ/'src'/'dados'/'grupos_mesma_arte.json'))
gid={c:i for i,g in enumerate(G) for c in g}
def cartas(f): return [Path(l.strip()).name[:9] for l in open(f) if l.strip()]
def imgs(f): return [l.strip() for l in open(f) if l.strip()]
L={}
for k in range(5):
    for r in ('treino','val_interna','avaliacao'):
        L[(k,r)]=set(cartas(D/f'fold{k}_{r}.txt'))
        im=imgs(D/f'fold{k}_{r}.txt')
        assert all(Path(x).is_file() for x in im), (k,r)
        assert all('/dataset_206_rev/images/' in x for x in im),(k,r)
        # frente e verso juntos
        c=collections.Counter(Path(x).name[:9] for x in im); assert set(c.values())=={2},(k,r,c)
for k in range(5):
    t,v,a=L[(k,'treino')],L[(k,'val_interna')],L[(k,'avaliacao')]
    grp=lambda s:{gid[c] for c in s if c in gid}
    print(f'dobra {k}: treino {len(t)} val {len(v)} aval {len(a)} | T&V {len(t&v)} T&A {len(t&a)} V&A {len(v&a)} | uniao {len(t|v|a)} | grupos T&V {len(grp(t)&grp(v))} T&A {len(grp(t)&grp(a))} V&A {len(grp(v)&grp(a))}')
aval_de={c:k for k in range(5) for c in L[(k,'avaliacao')]}
print('cada carta avaliada 1 vez:', len(aval_de)==206 and sum(len(L[(k,'avaliacao')]) for k in range(5))==206)
# item 2
v0=L[(0,'val_interna')]
print('fold0 val interna:',len(v0),'cartas; dobra retida delas:',collections.Counter(aval_de[c] for c in v0))
for k in range(1,5):
    print(f'  fold{k}_avaliacao: {len(v0&L[(k,"avaliacao")])} de {len(L[(k,"avaliacao")])} cartas vieram da val interna da dobra 0')
# curva
for f in (25,50,75):
    s=set(cartas(D/'curva'/f'fold4_treino_{f}.txt'))
    print(f'curva {f}%: {len(s)} cartas, subconjunto do treino4 {s<=L[(4,"treino")]}, &val {len(s&L[(4,"val_interna")])} &aval {len(s&L[(4,"avaliacao")])}, grupos partidos {sum(1 for g in G if 0<len(set(g)&s)<len(set(g)&L[(4,"treino")]))}')
s25,s50,s75=[set(cartas(D/'curva'/f'fold4_treino_{f}.txt')) for f in (25,50,75)]
print('aninhado', s25<=s50<=s75)
# anotacoes por fold e por papel
import numpy as np

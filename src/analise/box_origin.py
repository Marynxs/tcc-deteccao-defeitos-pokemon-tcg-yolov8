# classifica cada linha de dataset_206_rev/labels por origem: orig, prop, manual
from pathlib import Path as _P
RAIZ = _P(__file__).resolve().parents[2]
SAIDA = RAIZ / 'resultados' / 'analises'

import csv, json, collections
from pathlib import Path
D=RAIZ/'Dataset_YOLO'
acc=collections.defaultdict(list)
for r in csv.DictReader(open(D/'revisao_206/candidatos.csv')):
    if r['decisao'].strip().lower() in ('s','sim','y','1'):
        acc[r['imagem']].append(('manual' if r['modelos']=='manual' else 'prop', f"0 {r['xc']} {r['yc']} {r['w']} {r['h']}", r['modelos']))
out={}; cnt=collections.Counter(); bad=0
for lbl in sorted((D/'dataset_206_rev/labels').glob('*.txt')):
    L=[l for l in lbl.read_text().splitlines() if l.strip()]
    a=acc.get(lbl.stem,[])
    n0=len(L)-len(a)
    tail=L[n0:]
    ok = n0>=0 and all(t.strip()==x[1] for t,x in zip(tail,a))
    if not ok: bad+=1; print('divergente',lbl.stem)
    orig=['orig']*n0+[x[0] for x in a]
    out[lbl.stem]=orig
    cnt.update(orig)
print(cnt, 'total',sum(cnt.values()),'imagens divergentes',bad)
json.dump(out,open(SAIDA/'origem_caixas.json','w'))

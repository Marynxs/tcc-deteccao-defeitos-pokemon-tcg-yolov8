# reavalia as 25 rodadas com GT completo, so humano (proposta ignorada) e so proposta (humano ignorado)
from pathlib import Path as _P
RAIZ = _P(__file__).resolve().parents[2]
SAIDA = RAIZ / 'resultados' / 'analises'

import json, sys, collections, itertools
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.stats import wilcoxon
D=RAIZ/'Dataset_YOLO'
RUNS=D/'runs_206_rev'
ORIG=json.load(open(SAIDA/'origem_caixas.json'))
ROT=['cfg1_1280','cfg2_1280','cfg3_1280','cfg4_1280','cfg2_1280_p2']
IGN_THR=float(sys.argv[1]) if len(sys.argv)>1 else 0.5

def iou(a,b):
    if len(a)==0 or len(b)==0: return np.zeros((len(a),len(b)))
    ix1=np.maximum(a[:,None,0],b[None,:,0]); iy1=np.maximum(a[:,None,1],b[None,:,1])
    ix2=np.minimum(a[:,None,2],b[None,:,2]); iy2=np.minimum(a[:,None,3],b[None,:,3])
    inter=np.clip(ix2-ix1,0,None)*np.clip(iy2-iy1,0,None)
    aa=(a[:,2]-a[:,0])*(a[:,3]-a[:,1]); ab=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    return inter/(aa[:,None]+ab[None,:]-inter+1e-9)

GT={}
def gt(img):
    s=img.stem
    if s in GT: return GT[s]
    w,h=Image.open(img).size
    bx=[]
    for l in (D/'dataset_206_rev/labels'/f'{s}.txt').read_text().splitlines():
        p=l.split()
        if len(p)<5: continue
        cx,cy,bw,bh=map(float,p[1:5]); bx.append([(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h])
    GT[s]=(np.array(bx).reshape(-1,4),np.array(ORIG[s]))
    return GT[s]

def casa(pred,g,ign):
    """guloso por confianca; devolve casado(g), fp, lista (score, tp?) das nao ignoradas"""
    casado=np.zeros(len(g),bool); fp=0; lst=[]
    if len(pred)==0: return casado,0,lst
    o=np.argsort(-pred[:,4]); P=pred[o]
    m=iou(P[:,:4],g); mi=iou(P[:,:4],ign)
    for i in range(len(P)):
        cand=np.where(~casado)[0]
        if len(cand):
            j=cand[np.argmax(m[i,cand])]
            if m[i,j]>=0.5:
                casado[j]=True; lst.append((P[i,4],1)); continue
        if len(ign) and mi[i].max()>=IGN_THR: continue
        fp+=1; lst.append((P[i,4],0))
    return casado,fp,lst

def ap(lst,npos):
    if npos==0: return float('nan')
    if not lst: return 0.0
    a=np.array(sorted(lst,key=lambda t:-t[0])); tp=np.cumsum(a[:,1]); fp=np.cumsum(1-a[:,1])
    rec=tp/npos; prec=tp/(tp+fp)
    mrec=np.concatenate([[0],rec,[1]]); mpre=np.concatenate([[1],prec,[0]])
    mpre=np.flip(np.maximum.accumulate(np.flip(mpre)))
    x=np.linspace(0,1,101); return float(np.trapezoid(np.interp(x,mrec,mpre),x))

MODOS={'completo':(lambda o:np.ones(len(o),bool)),
       'humano':(lambda o:o!='prop'),
       'proposta':(lambda o:o=='prop'),
       'original':(lambda o:o=='orig'),
       'manual':(lambda o:o=='manual')}
res={}
for r in ROT:
    cfg,resto=r.split('_',1)
    res[r]={m:{'carta':collections.defaultdict(lambda:[0,0,0]),'ap':[]} for m in MODOS}
    for k in range(5):
        nome=f'{cfg}_fold{k}_{resto}'
        lim=json.load(open(RUNS/nome/'resumo.json'))['conf_do_f1']
        allp=collections.defaultdict(list)
        for p in json.load(open(RUNS/f'{nome}_avaliacao'/'predictions.json')):
            x,y,w,h=p['bbox']; allp[p['image_id']].append([x,y,x+w,y+h,p['score']])
        imgs=[Path(l.strip()) for l in open(D/'folds_206_rev'/f'fold{k}_avaliacao.txt') if l.strip()]
        for m,sel in MODOS.items():
            lst_all=[]; npos=0
            for img in imgs:
                g,o=gt(img); s=sel(o); G=g[s]; I=g[~s]
                pa=np.array(allp.get(img.stem,[])).reshape(-1,5)
                # AP com todas as predicoes
                c,_,lst=casa(pa,G,I); lst_all+=lst; npos+=len(G)
                # ponto de operacao: limiar da val interna
                pt=pa[pa[:,4]>=lim]
                c,fp,_=casa(pt,G,I)
                cc=res[r][m]['carta'][img.stem[:9]]
                cc[0]+=int(c.sum()); cc[1]+=fp; cc[2]+=int((~c).sum())
            res[r][m]['ap'].append(ap(lst_all,npos))
        print(r,k,'ok',file=sys.stderr)

def f1c(pc): return {c:2*tp/(2*tp+fp+fn) for c,(tp,fp,fn) in pc.items() if tp+fn>0}
def rcc(pc): return {c:tp/(tp+fn) for c,(tp,fp,fn) in pc.items() if tp+fn>0}
print(f'limiar de IoU para regiao ignorada: {IGN_THR}')
for m in MODOS:
    print(f'\n== recorte {m} ==')
    for r in ROT:
        pc=res[r][m]['carta']; tp=sum(v[0] for v in pc.values()); fp=sum(v[1] for v in pc.values()); fn=sum(v[2] for v in pc.values())
        R=tp/(tp+fn); P=tp/max(tp+fp,1); F=2*P*R/(P+R)
        f=f1c(pc)
        print(f'{r:14s} TP {tp:5d} FP {fp:5d} FN {fn:5d} recall {R:.3f} precisao {P:.3f} F1 {F:.3f} | AP50 medio dobras {np.nanmean(res[r][m]["ap"]):.3f} ({" ".join(f"{a:.3f}" for a in res[r][m]["ap"])}) | F1 por carta medio {np.mean(list(f.values())):.3f} n={len(f)}')
    pares=[('cfg2_1280','cfg1_1280'),('cfg2_1280','cfg3_1280'),('cfg2_1280','cfg4_1280'),('cfg2_1280','cfg2_1280_p2'),('cfg1_1280','cfg3_1280'),('cfg4_1280','cfg3_1280')]
    for met,fn_ in (('F1',f1c),('recall',rcc)):
        ps=[]
        rows=[]
        for a,b in pares:
            A=fn_(res[a][m]['carta']); B=fn_(res[b][m]['carta']); cs=sorted(set(A)&set(B))
            xa=np.array([A[c] for c in cs]); xb=np.array([B[c] for c in cs]); d=xa-xb
            p=wilcoxon(xa,xb).pvalue if (d!=0).any() else 1.0
            pp=wilcoxon(xa,xb,zero_method='pratt').pvalue if (d!=0).any() else 1.0
            ps.append(p); rows.append((a,b,len(cs),xa.mean(),xb.mean(),(d>0).sum(),(d<0).sum(),(d==0).sum(),p,pp))
        # Holm so na familia das hipoteses (os 5 pares de compare_paired.py); o par
        # cfg2 x cfg3 nao testa hipotese e fica fora, sem valor ajustado
        fam=[i for i,(a,b) in enumerate(pares) if (a,b)!=('cfg2_1280','cfg3_1280')]
        o=sorted(fam,key=lambda i:ps[i]); holm=np.full(len(ps),np.nan); run=0
        for rank,i in enumerate(o):
            run=max(run,min(1,(len(fam)-rank)*ps[i])); holm[i]=run
        for (a,b,n,ma,mb,w,l,t,p,pp),h in zip(rows,holm):
            print(f'  {met:6s} {a} vs {b}: n={n} {ma:.3f} vs {mb:.3f} melhor/pior/empate {w}/{l}/{t} p={p:.2g} (pratt {pp:.2g}) Holm {h:.2g} (m=5)' if h==h else f'  {met:6s} {a} vs {b}: n={n} {ma:.3f} vs {mb:.3f} melhor/pior/empate {w}/{l}/{t} p={p:.2g} (pratt {pp:.2g}) fora da familia de Holm')
json.dump({r:{m:{'ap':res[r][m]['ap'],'carta':dict(res[r][m]['carta'])} for m in MODOS} for r in ROT},open(SAIDA/f'reavaliacao_so_humano_{IGN_THR}.json','w'))

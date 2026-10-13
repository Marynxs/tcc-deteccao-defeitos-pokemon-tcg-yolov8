# Teste de interacao aumento x realce, por carta: d = (cfg2-cfg1) - (cfg4-cfg3), Wilcoxon bilateral
from pathlib import Path as _P
RAIZ = _P(__file__).resolve().parents[2]
SAIDA = RAIZ / 'resultados' / 'analises'

import sys
sys.path.insert(0, str(RAIZ / "src" / "treino"))
import numpy as np
from scipy.stats import wilcoxon
import compare_paired as cp
runs = cp.DATASET / "runs_206_rev"
f = {}
for r in ["cfg1_1280", "cfg2_1280", "cfg3_1280", "cfg4_1280"]:
    pc, _ = cp.avalia_rodada(r, "_206_rev", runs)
    f[r] = cp.f1_por_carta(pc)
cartas = sorted(set.intersection(*[set(v) for v in f.values()]))
a = np.array([f["cfg2_1280"][c] - f["cfg1_1280"][c] for c in cartas])
b = np.array([f["cfg4_1280"][c] - f["cfg3_1280"][c] for c in cartas])
d = a - b
print("n cartas:", len(cartas))
print(f"efeito do aumento, RGB: media {a.mean():+.4f} mediana {np.median(a):+.4f}")
print(f"efeito do aumento, realce: media {b.mean():+.4f} mediana {np.median(b):+.4f}")
print(f"interacao (diferenca das diferencas): media {d.mean():+.4f} mediana {np.median(d):+.4f}; positivas {int((d>0).sum())} negativas {int((d<0).sum())} zeros {int((d==0).sum())}")
st = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
print("Wilcoxon bilateral: W =", st.statistic, "p =", st.pvalue)
st2 = wilcoxon(d, zero_method="pratt", alternative="two-sided")
print("Wilcoxon (Pratt, mantem zeros): p =", st2.pvalue)

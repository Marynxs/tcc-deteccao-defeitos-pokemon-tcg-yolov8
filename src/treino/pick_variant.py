"""
Escolhe a variante de realce pela validacao interna e registra a escolha em resultados/.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
TREINADOR = RAIZ / "src" / "treino" / "train_fold.py"

VARIANTES = {
    "a": "luminosidade, emboss 3x3, CLAHE",
    "b": "luminosidade, CLAHE, emboss de 8 direcoes (Sawada 2024)",
    "d": "luminosidade, CLAHE, cristas multiescala por Hessiana (Gruber 2021)",
    "e": "composto: L original, L com CLAHE, Frangi sobre o L com CLAHE",
}

# Margem relativa minima para chamar a diferenca de real. NAO e teste estatistico:
# com uma dobra por variante nao existe teste possivel. O valor vem do ruido medido
# na execucao piloto (o mAP da validacao interna varia em media 21% do proprio
# nivel entre epocas vizinhas). Abaixo disso, a escolha e selecao de hiperparametro,
# nao demonstracao de superioridade.
MARGEM_MINIMA = 0.20


def carrega(runs: Path, padrao: str, v: str) -> dict | None:
    f = runs / padrao.format(v=v) / "resumo.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_206_rev", help="pasta em Dataset_YOLO")
    ap.add_argument("--padrao", default="cfg3_fold0_1280_realce_{v}", help="nome da rodada por variante")
    # A escolha e feita na validacao interna, e nao na dobra retida: escolher pela
    # dobra que depois mede o desempenho tornaria a estimativa otimista (Cawley2010).
    # A dobra retida aparece so como diagnostico.
    ap.add_argument("--criterio", default="f1_val_interna")
    ap.add_argument("--csv", default=str(RAIZ / "resultados" / "selecao_realce_206.csv"))
    ap.add_argument("--fixar", action="store_true", help="grava a vencedora em REALCE_PADRAO do train_fold.py")
    args = ap.parse_args()

    runs = RAIZ / "Dataset_YOLO" / args.runs
    dados = {v: carrega(runs, args.padrao, v) for v in VARIANTES}
    faltando = [v for v, d in dados.items() if d is None]
    if faltando:
        raise SystemExit(f"ainda sem resultado: {', '.join(faltando)}")

    ordem = sorted(dados, key=lambda v: dados[v][args.criterio], reverse=True)
    vencedora, segunda = ordem[0], ordem[1]
    a, b = dados[vencedora][args.criterio], dados[segunda][args.criterio]
    margem = a / b - 1 if b > 0 else float("inf")
    ordem_retida = sorted(dados, key=lambda v: dados[v]["map50"], reverse=True)

    print(f"Selecao do operador de realce ({args.padrao.format(v='*')}), criterio: {args.criterio}\n")
    print(f"{'var':>4} {'F1 val.int.':>11} {'mAP50 val.int.':>15} | {'mAP50 retida':>13} {'F1 retida':>10}")
    for v in ordem:
        d = dados[v]
        marca = "  <<< escolhida" if v == vencedora else ""
        print(f"{v:>4} {d['f1_val_interna']:>11.4f} {d['map50_val_interna']:>15.4f} | "
              f"{d['map50']:>13.4f} {d['f1']:>10.4f}{marca}")
    print(f"\nmargem da escolhida sobre a segunda ({segunda}): {margem*100:.0f}% "
          f"(referencia {MARGEM_MINIMA*100:.0f}%) -> "
          f"{'diferenca acima da referencia' if margem >= MARGEM_MINIMA else 'EMPATE: selecao de hiperparametro, nao demonstracao'}")
    print(f"ordem pela validacao interna : {' > '.join(ordem)}")
    print(f"ordem pela dobra retida (diag.): {' > '.join(ordem_retida)}")
    print(f"a escolhida tambem lidera na dobra retida: {'sim' if ordem_retida[0] == vencedora else 'NAO'}")

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variante", "descricao", "f1_val_interna", "map50_val_interna", "map50_dobra_retida",
                    "f1_dobra_retida", "escolhida", "criterio", "margem_sobre_segunda"])
        for v in ordem:
            d = dados[v]
            w.writerow([v, VARIANTES[v], d["f1_val_interna"], d["map50_val_interna"], d["map50"], d["f1"],
                        int(v == vencedora), args.criterio, f"{margem:.4f}" if v == vencedora else ""])
    print("CSV gravado em", args.csv)

    if args.fixar:
        txt = TREINADOR.read_text(encoding="utf-8")
        novo = re.sub(r'REALCE_PADRAO = "[abde]"', f'REALCE_PADRAO = "{vencedora}"', txt)
        if novo != txt:
            TREINADOR.write_text(novo, encoding="utf-8")
            print(f'REALCE_PADRAO fixado em "{vencedora}"')


if __name__ == "__main__":
    main()

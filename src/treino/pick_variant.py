"""
Escolhe a variante de realce vencedora, fixa o padrao e registra os resultados.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
RUNS = RAIZ / "Dataset_YOLO" / "runs"
TREINADOR = RAIZ / "src" / "treino" / "train_fold.py"
NOTAS = RAIZ / "docs" / "notas_experimentos_tcc2.md"

VARIANTES = {
    "a": "luminancia, emboss 3x3, CLAHE (o que o TCC1 descrevia)",
    "b": "luminancia, CLAHE, emboss de 8 direcoes (Sawada 2024)",
    "d": "luminancia, CLAHE, cristas multiescala por Hessiana (Gruber 2021)",
}


def resumo(v: str) -> dict | None:
    f = RUNS / f"cfg3v{v}_fold0_1280" / "resumo.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


def main() -> None:
    dados = {v: resumo(v) for v in VARIANTES}
    faltando = [v for v, d in dados.items() if d is None]
    if faltando:
        raise SystemExit(f"ainda sem resultado: {', '.join(faltando)}")

    # O criterio e o mAP@0,5 no fold retido, que nunca participou do treino
    # nem da escolha do checkpoint. Nao usar a validacao interna aqui: ela e
    # otimista, porque alimentou a parada antecipada.
    vencedora = max(dados, key=lambda v: dados[v]["map50"])

    print("Teste preliminar do operador de realce, dobra 0, fold retido\n")
    print(f"{'var':>4} {'mAP@0,5':>9} {'mAP@0,5:0,95':>13} {'precisao':>9} "
          f"{'revocacao':>10} {'epocas':>7}")
    for v, d in sorted(dados.items()):
        marca = "  <<< vencedora" if v == vencedora else ""
        print(f"{v:>4} {d['map50']:>9.4f} {d['map50_95']:>13.4f} "
              f"{d['precisao']:>9.4f} {d['recall']:>10.4f} "
              f"{d['epocas_rodadas']:>7}{marca}")

    # fixa o padrao no proprio script de treino
    txt = TREINADOR.read_text(encoding="utf-8")
    novo = re.sub(r'REALCE_PADRAO = "[abd]"', f'REALCE_PADRAO = "{vencedora}"', txt)
    if novo != txt:
        TREINADOR.write_text(novo, encoding="utf-8")
        print(f"\nREALCE_PADRAO fixado em \"{vencedora}\"")

    # registra nas notas, sem tocar no .tex
    linhas = ["", "### Resultado do teste preliminar", "",
              "| variante | pipeline | mAP@0,5 | mAP@0,5:0,95 | revocação | épocas |",
              "|---|---|---|---|---|---|"]
    for v, d in sorted(dados.items()):
        forte = "**" if v == vencedora else ""
        linhas.append(
            f"| {forte}{v.upper()}{forte} | {VARIANTES[v]} | {forte}{d['map50']:.4f}{forte} "
            f"| {d['map50_95']:.4f} | {d['recall']:.4f} | {d['epocas_rodadas']} |")
    base = resumo_cfg1()
    if base:
        linhas.append(f"| — | RGB, sem realce (Config. 1) | {base['map50']:.4f} "
                      f"| {base['map50_95']:.4f} | {base['recall']:.4f} "
                      f"| {base['epocas_rodadas']} |")
    linhas += ["",
               f"Vencedora: **variante {vencedora.upper()}**, adotada nas "
               "Configurações 3 e 4. Critério: mAP@0,5 no fold retido, que não "
               "participou do treinamento nem da seleção do checkpoint.",
               "", "Figura: `curvas_variantes_realce`.", ""]

    s = NOTAS.read_text(encoding="utf-8")
    alvo = "RESULTADOS: preencher quando as três terminarem."
    if alvo in s:
        NOTAS.write_text(s.replace(alvo, "\n".join(linhas).strip()), encoding="utf-8")
        print(f"resultados registrados em {NOTAS.name}")


def resumo_cfg1() -> dict | None:
    f = RUNS / "cfg1_fold0_1280" / "resumo.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


if __name__ == "__main__":
    main()

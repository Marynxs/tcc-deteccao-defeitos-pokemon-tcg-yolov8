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


# Margem relativa minima para a escolha ser tratada como resolvida. NAO e teste
# estatistico: com UMA dobra por variante nao existe teste possivel. O valor vem
# do ruido medido na execucao piloto, em que o mAP da validacao interna varia em
# media 21% do proprio nivel entre epocas vizinhas. Uma diferenca abaixo dessa
# ordem de grandeza nao se distingue de flutuacao.
MARGEM_MINIMA = 0.20


def avaliar(dados: dict, vencedora: str, segunda: str) -> dict:
    """Diz se a escolha se sustenta ou se ficou dentro do ruido."""
    a, b = dados[vencedora]["map50"], dados[segunda]["map50"]
    margem = (a / b - 1) if b > 0 else float("inf")
    # coerencia: a vencedora tambem lidera na metrica mais exigente?
    lidera_5095 = dados[vencedora]["map50_95"] == max(d["map50_95"] for d in dados.values())

    base = resumo_cfg1()
    supera_rgb = base is not None and a > base["map50"]
    ganho_rgb = (a / base["map50"] - 1) if base and base["map50"] > 0 else None

    conclusivo = margem >= MARGEM_MINIMA and lidera_5095
    return {"margem": margem, "lidera_5095": lidera_5095, "conclusivo": conclusivo,
            "supera_rgb": supera_rgb, "ganho_rgb": ganho_rgb, "base": base}


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
    ordenadas = sorted(dados, key=lambda v: dados[v]["map50"], reverse=True)
    vencedora, segunda = ordenadas[0], ordenadas[1]
    veredito = avaliar(dados, vencedora, segunda)

    print("Teste preliminar do operador de realce, dobra 0, fold retido\n")
    print(f"{'var':>4} {'mAP@0,5':>9} {'mAP@0,5:0,95':>13} {'precisao':>9} "
          f"{'revocacao':>10} {'epocas':>7}")
    for v, d in sorted(dados.items()):
        marca = "  <<< vencedora" if v == vencedora else ""
        print(f"{v:>4} {d['map50']:>9.4f} {d['map50_95']:>13.4f} "
              f"{d['precisao']:>9.4f} {d['recall']:>10.4f} "
              f"{d['epocas_rodadas']:>7}{marca}")

    print()
    if veredito["conclusivo"]:
        print(f"VEREDITO: escolha resolvida. A variante {vencedora.upper()} supera a "
              f"segunda em {veredito['margem']*100:.0f}% e tambem lidera o mAP@0,5:0,95.")
    else:
        print(f"VEREDITO: >>> INCONCLUSIVO <<<")
        print(f"  margem sobre a segunda colocada: {veredito['margem']*100:.0f}% "
              f"(minimo adotado: {MARGEM_MINIMA*100:.0f}%)")
        print(f"  lidera tambem no mAP@0,5:0,95: {'sim' if veredito['lidera_5095'] else 'NAO'}")
        print(f"  A campanha segue com a {vencedora.upper()}, mas a escolha NAO esta")
        print(f"  demonstrada e precisa ser revista com as 5 dobras.")
    if veredito["base"]:
        if veredito["supera_rgb"]:
            print(f"  A vencedora supera a linha de base RGB em "
                  f"{veredito['ganho_rgb']*100:.0f}%.")
        else:
            print(f"  ATENCAO: NENHUMA variante superou a linha de base RGB "
                  f"({veredito['base']['map50']:.4f}). O realce pode nao estar ajudando.")

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
               "participou do treinamento nem da seleção do checkpoint.", ""]

    if veredito["conclusivo"]:
        linhas.append(
            f"**Escolha resolvida.** A vencedora supera a segunda colocada em "
            f"{veredito['margem']*100:.0f}% no mAP@0,5 e também lidera o "
            f"mAP@0,5:0,95. Ainda assim, trata-se de UMA dobra: a margem é "
            f"grande o bastante para orientar a decisão, não para constituir "
            f"demonstração estatística.")
    else:
        linhas += [
            "### ATENÇÃO: a escolha ficou INCONCLUSIVA",
            "",
            f"A variante vencedora supera a segunda colocada em apenas "
            f"{veredito['margem']*100:.0f}% no mAP@0,5, abaixo da margem de "
            f"{MARGEM_MINIMA*100:.0f}% adotada como referência, e "
            f"{'lidera' if veredito['lidera_5095'] else 'NÃO lidera'} o "
            f"mAP@0,5:0,95.",
            "",
            "A margem de referência não é teste estatístico: com uma dobra por "
            "variante não existe teste possível. Ela vem do ruído medido na "
            "execução piloto, em que o mAP da validação interna varia em média "
            "21% do próprio nível entre épocas vizinhas. Uma diferença abaixo "
            "dessa ordem de grandeza não se distingue de flutuação.",
            "",
            "**Consequência para o texto.** A escolha do operador NÃO pode ser "
            "apresentada como demonstrada. Duas saídas honestas: relatar que as "
            "variantes ficaram equivalentes na dobra testada e que se adotou a "
            "de melhor valor absoluto; ou repetir o teste nas cinco dobras, ao "
            "custo de cerca de 6 h, e comparar por carta.",
            "",
            f"A campanha prosseguiu com a variante {vencedora.upper()} para não "
            f"paralisar o cronograma, mas essa decisão está em aberto.",
        ]

    if veredito["base"] and not veredito["supera_rgb"]:
        linhas += [
            "",
            "### ATENÇÃO: nenhuma variante superou a linha de base RGB",
            "",
            f"O melhor realce obteve {dados[vencedora]['map50']:.4f} contra "
            f"{veredito['base']['map50']:.4f} da Configuração 1, em RGB. Se isso "
            "se confirmar nas cinco dobras, o resultado do fator "
            "'representação de entrada' é NEGATIVO, o que é achado legítimo e "
            "publicável, mas muda a redação de toda a Seção 3.4 e da conclusão.",
        ]

    linhas += ["", "Figura: `curvas_variantes_realce`.", ""]

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

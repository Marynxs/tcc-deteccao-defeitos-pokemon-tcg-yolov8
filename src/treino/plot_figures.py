"""
Gera as figuras dos experimentos a partir dos results.csv das rodadas.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator

RAIZ = Path(__file__).resolve().parents[2]
RUNS = RAIZ / "Dataset_YOLO" / "runs"
SAIDA = RAIZ / "tcc2" / "assets" / "figuras"

# Slots 1 e 2 da paleta categorica de referencia, usados na ordem documentada.
# A monografia pode ser impressa em preto e branco, entao a cor NUNCA e o unico
# portador de identidade: cada serie tambem tem estilo de linha proprio.
AZUL, LARANJA = "#2a78d6", "#eb6834"
VERDE, CINZA = "#1baf7a", "#6f6e69"
TINTA, TINTA_FRACA, GRADE = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.edgecolor": TINTA_FRACA,
    "axes.labelcolor": TINTA,
    "text.color": TINTA,
    "xtick.color": TINTA_FRACA,
    "ytick.color": TINTA_FRACA,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


def ler(rodada: str) -> list[dict]:
    arq = RUNS / rodada / "results.csv"
    if not arq.is_file():
        raise SystemExit(f"nao encontrei {arq}")
    linhas = []
    with arq.open(encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            try:
                linhas.append({
                    "epoca": int(float(r["epoch"])),
                    "map50": float(r["metrics/mAP50(B)"]),
                    "map5095": float(r["metrics/mAP50-95(B)"]),
                })
            except (KeyError, ValueError):
                continue
    return linhas


# A monografia e em portugues e a ABNT pede virgula decimal. O matplotlib
# formata com ponto por padrao, entao o eixo e os rotulos passam por aqui.
def _virgula(valor, _=None) -> str:
    return f"{valor:.2f}".replace(".", ",")


def _num(valor: float, casas: int = 3) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def _limpar(ax) -> None:
    """Grade e eixos recessivos: os dados e que devem carregar a figura."""
    # Passos so em 1, 2 e 5: com passo 0,025 o formatador de duas casas
    # arredondaria para 0,03 e 0,08, rotulos que nao correspondem ao valor.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(_virgula))
    ax.grid(True, color=GRADE, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_linewidth(0.7)


def _gravar(fig, nome: str) -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(SAIDA / f"{nome}.{ext}")
    plt.close(fig)
    print(f"  {nome}.pdf e {nome}.png")


def curvas_configuracoes(rodadas: list[tuple[str, str]]) -> None:
    """mAP@0,5 por epoca, uma linha por configuracao."""
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    estilos = [(AZUL, "-"), (LARANJA, "--")]

    for (rodada, rotulo), (cor, traco) in zip(rodadas, estilos):
        d = ler(rodada)
        x = [p["epoca"] for p in d]
        y = [p["map50"] for p in d]
        # O ponto marca onde esta o pico; o valor vai na legenda. Rotulo flutuando
        # ao lado do pico colide com a outra curva, e nenhum deslocamento fixo
        # resolve isso para todas as rodadas. Com as quatro configuracoes no
        # mesmo quadro seria pior ainda.
        i = max(range(len(y)), key=lambda k: y[k])
        ax.plot(x, y, color=cor, linestyle=traco, linewidth=1.6,
                label=f"{rotulo}  (pico {_num(y[i])} na época {x[i]})",
                solid_capstyle="round")
        ax.plot(x[i], y[i], marker="o", markersize=5, color=cor,
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)

    _limpar(ax)
    ax.set_xlabel("Época de treinamento")
    ax.set_ylabel("mAP@0,5 na validação interna")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.14)
    # Canto superior esquerdo: as duas curvas partem de zero, entao essa regiao
    # fica livre em qualquer rodada, e a legenda nao colide com os dados.
    leg = ax.legend(frameon=False, loc="upper left", handlelength=2.6)
    for t in leg.get_texts():
        t.set_color(TINTA)  # texto usa tinta, nunca a cor da serie
    _gravar(fig, "curvas_map_configuracoes")


def ruido_validacao(rodada: str, janela: int = 5) -> None:
    """A curva crua contra a media movel, para mostrar os minimos locais."""
    d = ler(rodada)
    x = [p["epoca"] for p in d]
    y = [p["map50"] for p in d]
    sx, sy = [], []
    for i in range(len(y) - janela + 1):
        sx.append(x[i + janela // 2])
        sy.append(sum(y[i:i + janela]) / janela)

    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    ax.plot(x, y, color=AZUL, linewidth=1.0, alpha=0.55,
            label="mAP@0,5 por época")
    ax.plot(sx, sy, color=LARANJA, linestyle="--", linewidth=1.8,
            label=f"média móvel de {janela} épocas")

    i = max(range(len(y)), key=lambda k: y[k])
    j = max(range(len(sy)), key=lambda k: sy[k])
    for xx, yy, cor, txt in ((x[i], y[i], AZUL, f"pico bruto: época {x[i]}"),
                             (sx[j], sy[j], LARANJA, f"pico suavizado: época {sx[j]}")):
        ax.plot(xx, yy, marker="o", markersize=5, color=cor,
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)
        ax.annotate(txt, xy=(xx, yy), xytext=(7, 6), textcoords="offset points",
                    fontsize=8, color=TINTA_FRACA)

    _limpar(ax)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("Época de treinamento")
    ax.set_ylabel("mAP@0,5 na validação interna")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.10)
    leg = ax.legend(frameon=False, loc="upper right", handlelength=2.6)
    for t in leg.get_texts():
        t.set_color(TINTA)
    _gravar(fig, "ruido_validacao_interna")


def curvas_variantes(rodadas: list[tuple[str, str]], base: tuple[str, str] | None) -> None:
    """As variantes de realce contra a linha de base RGB, na mesma dobra.

    Figura de argumentacao da escolha do operador: mostra lado a lado o que cada
    variante alcanca sob protocolo identico, e nao apenas o numero da vencedora.
    """
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    estilos = [(AZUL, "-"), (LARANJA, "--"), (VERDE, "-."), (CINZA, ":")]
    series = ([base] if base else []) + rodadas

    for (rodada, rotulo), (cor, traco) in zip(series, estilos):
        try:
            d = ler(rodada)
        except SystemExit:
            print(f"  (pulando {rodada}, ainda nao existe)")
            continue
        x = [p["epoca"] for p in d]
        y = [p["map50"] for p in d]
        i = max(range(len(y)), key=lambda k: y[k])
        ax.plot(x, y, color=cor, linestyle=traco, linewidth=1.6,
                label=f"{rotulo}  (pico {_num(y[i])})", solid_capstyle="round")
        ax.plot(x[i], y[i], marker="o", markersize=5, color=cor,
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)

    _limpar(ax)
    ax.set_xlabel("Época de treinamento")
    ax.set_ylabel("mAP@0,5 na validação interna")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.16)
    leg = ax.legend(frameon=False, loc="upper left", handlelength=2.6, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(TINTA)
    _gravar(fig, "curvas_variantes_realce")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg1", default="cfg1_fold0_1280")
    p.add_argument("--cfg2", default="cfg2_fold0_1280")
    p.add_argument("--variantes", action="store_true",
                   help="gera tambem a figura das tres variantes de realce")
    args = p.parse_args()

    print(f"gravando em {SAIDA}")
    curvas_configuracoes([
        (args.cfg1, "Config. 1: RGB, sem aumento de dados"),
        (args.cfg2, "Config. 2: RGB, com aumento de dados"),
    ])
    ruido_validacao(args.cfg1)
    if args.variantes:
        curvas_variantes(
            [("cfg3va_fold0_1280", "A: gradiente direcional, depois CLAHE"),
             ("cfg3vb_fold0_1280", "B: CLAHE, depois gradiente de 8 direções"),
             ("cfg3vd_fold0_1280", "D: CLAHE, depois cristas por Hessiana")],
            base=(args.cfg1, "RGB, sem realce"),
        )


if __name__ == "__main__":
    main()

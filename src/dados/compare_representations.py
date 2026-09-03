"""
Recorta um defeito anotado e mostra como ele aparece em cada representacao.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
DADOS = RAIZ / "Dataset_YOLO"
SAIDA = RAIZ / "tcc2" / "assets" / "figuras"

REPRESENTACOES = [
    ("dataset", "RGB"),
    ("dataset_realce_a", "A emboss+CLAHE"),
    ("dataset_realce_b", "B CLAHE+emboss8"),
    ("dataset_realce_d", "D Frangi"),
]


def contraste_local(img: np.ndarray, caixa: tuple, margem: int) -> float:
    """Quantos desvios do anel ao redor o pico dentro da caixa se destaca.

    E a medida que importa: uma convolucao compara o defeito com os vizinhos
    imediatos, nunca com o outro canto da carta. Olhando a carta inteira o
    defeito some, porque a arte impressa consome a faixa dinamica do monitor;
    olhando a vizinhanca, ele e a coisa mais forte do quadro.
    """
    x1, y1, x2, y2 = caixa
    H, W = img.shape
    ax1, ay1 = max(x1 - margem, 0), max(y1 - margem, 0)
    ax2, ay2 = min(x2 + margem, W), min(y2 + margem, H)
    viz = img[ay1:ay2, ax1:ax2].astype(np.float32)
    cx, cy = x1 - ax1, y1 - ay1
    dentro = viz[cy:cy + (y2 - y1), cx:cx + (x2 - x1)]
    anel = viz.copy()
    anel[cy:cy + (y2 - y1), cx:cx + (x2 - x1)] = np.nan
    sd = np.nanstd(anel)
    if sd < 1e-6 or dentro.size == 0:
        return 0.0
    return float(abs(dentro.max() - np.nanmean(anel)) / sd)


def escolher(min_px: int, max_px: int, margem: int, limite: int):
    """Procura o defeito de melhor contraste local na variante de segunda ordem."""
    melhor = None
    for p in sorted((DADOS / "dataset" / "images").glob("*.png"))[:limite]:
        rot = DADOS / "dataset" / "labels" / (p.stem + ".txt")
        texto = rot.read_text().strip() if rot.is_file() else ""
        if not texto:
            continue
        ref = cv2.imread(str(DADOS / "dataset_realce_d" / "images" / p.name), 0)
        if ref is None:
            continue
        H, W = ref.shape
        for linha in texto.splitlines():
            cx, cy, w, h = (float(v) for v in linha.split()[1:5])
            if not (min_px < min(w * W, h * H) < max_px):
                continue
            caixa = (int((cx - w / 2) * W), int((cy - h / 2) * H),
                     int((cx + w / 2) * W), int((cy + h / 2) * H))
            z = contraste_local(ref, caixa, margem)
            if melhor is None or z > melhor[0]:
                melhor = (z, p.name, caixa)
    return melhor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--min-px", type=int, default=10)
    p.add_argument("--max-px", type=int, default=40)
    p.add_argument("--margem", type=int, default=40, help="contexto em volta, em px")
    p.add_argument("--zoom", type=int, default=4)
    p.add_argument("--limite", type=int, default=80, help="quantas imagens vasculhar")
    p.add_argument("--nome", default="defeito_por_representacao")
    args = p.parse_args()

    achado = escolher(args.min_px, args.max_px, args.margem, args.limite)
    if achado is None:
        raise SystemExit("nenhum defeito na faixa de tamanho pedida")
    z, arquivo, (x1, y1, x2, y2) = achado
    print(f"defeito: {arquivo}")
    print(f"caixa {x2-x1}x{y2-y1} px, contraste local {z:.1f} desvios na variante D")

    m = args.margem
    paineis = []
    for pasta, rotulo in REPRESENTACOES:
        img = cv2.imread(str(DADOS / pasta / "images" / arquivo), 0)
        if img is None:
            raise SystemExit(f"faltando {pasta}/images/{arquivo}")
        H, W = img.shape
        ax1, ay1 = max(x1 - m, 0), max(y1 - m, 0)
        ax2, ay2 = min(x2 + m, W), min(y2 + m, H)
        rec = img[ay1:ay2, ax1:ax2].astype(np.float32)
        # esticar o contraste LOCALMENTE, para reproduzir o que a rede processa
        lo, hi = np.percentile(rec, (2, 98))
        v = np.clip((rec - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        v = cv2.resize(v, None, fx=args.zoom, fy=args.zoom,
                       interpolation=cv2.INTER_NEAREST)
        v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(v, ((x1 - ax1) * args.zoom, (y1 - ay1) * args.zoom),
                      ((x2 - ax1) * args.zoom, (y2 - ay1) * args.zoom), (0, 0, 255), 2)
        paineis.append((v, rotulo))

    alt = min(v.shape[0] for v, _ in paineis)
    tira = np.hstack([v[:alt] for v, _ in paineis])
    faixa = np.full((30, tira.shape[1], 3), 255, np.uint8)
    larg = tira.shape[1] // len(paineis)
    for i, (_, rotulo) in enumerate(paineis):
        cv2.putText(faixa, rotulo, (i * larg + 8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)

    SAIDA.mkdir(parents=True, exist_ok=True)
    destino = SAIDA / f"{args.nome}.png"
    cv2.imwrite(str(destino), np.vstack([faixa, tira]))
    print(f"gravado {destino}")


if __name__ == "__main__":
    main()

"""
Gera as representacoes de realce de micro-relevo a partir das imagens recortadas.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
ORIGEM = RAIZ / "Dataset_YOLO" / "dataset"

# Kernel de emboss. A soma dos coeficientes e ZERO, de modo que o operador
# devolve a derivada direcional pura; somada a 128, a saida fica centrada no
# cinza medio, que e o aspecto classico de uma imagem embossada.
#
# A variante com soma 1, usada por algumas bibliotecas, devolve a imagem somada
# ao gradiente. Somar 128 aquilo satura: numa carta clara a saida passa de 255 e
# e cortada, com media perto de 200 em vez de 128, perdendo justamente o relevo
# nas regioes claras. Conferido contra a figura ja presente no TCC, cuja media e
# 127,07: so o kernel de soma zero a reproduz.
EMBOSS = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], np.float32)


def luminancia(bgr: np.ndarray) -> np.ndarray:
    """Canal L do CIELAB.

    O relevo se manifesta como variacao de brilho, nao de cor. Trabalhar no L
    tambem evita o desvio cromatico que o CLAHE aplicado aos tres canais RGB
    produz, ja que cada canal seria esticado por um fator diferente.
    """
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)[:, :, 0]


def _clahe(limite: float, lado: int) -> "cv2.CLAHE":
    return cv2.createCLAHE(clipLimit=limite, tileGridSize=(lado, lado))


def variante_a(bgr: np.ndarray, limite: float, lado: int, **_) -> np.ndarray:
    """Gradiente direcional seguido de CLAHE, como descrito no TCC1."""
    g = luminancia(bgr).astype(np.float32)
    e = np.clip(cv2.filter2D(g, -1, EMBOSS) + 128, 0, 255).astype(np.uint8)
    return _clahe(limite, lado).apply(e)


def variante_b(bgr: np.ndarray, limite: float, lado: int, **_) -> np.ndarray:
    """CLAHE seguido de gradiente direcional de oito direcoes (Sawada2024)."""
    c = _clahe(limite, lado).apply(luminancia(bgr)).astype(np.float32)
    # O kernel ja soma zero, entao filter2D devolve a derivada direcional pura:
    # nao se subtrai a imagem, o que agora removeria sinal em vez de o termo DC.
    respostas = [np.abs(cv2.filter2D(c, -1, np.rot90(EMBOSS, k))) for k in range(4)]
    return _normalizar(np.max(np.stack(respostas), axis=0))


def _hessiana(g: np.ndarray, sigma: float) -> np.ndarray:
    """Autovalor de maior modulo da matriz Hessiana, na escala dada."""
    f = cv2.GaussianBlur(g, (0, 0), sigma)
    gxx = cv2.Sobel(f, cv2.CV_32F, 2, 0, ksize=3)
    gyy = cv2.Sobel(f, cv2.CV_32F, 0, 2, ksize=3)
    gxy = cv2.Sobel(f, cv2.CV_32F, 1, 1, ksize=3)
    tr, det = gxx + gyy, gxx * gyy - gxy * gxy
    raiz = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    l1, l2 = tr / 2 + raiz, tr / 2 - raiz
    return np.where(np.abs(l1) > np.abs(l2), l1, l2)


def variante_d(bgr: np.ndarray, limite: float, lado: int, sigmas=(1.5, 3.0, 5.0)) -> np.ndarray:
    """Resposta multiescala aos autovalores da Hessiana (Gruber2021).

    Risco e vinco sao cristas e vales, estruturas de segunda ordem, e nao
    bordas. As escalas foram fixadas na faixa de tamanhos medida no conjunto:
    mediana de 8,66 px na menor dimensao, na escala 1280.
    """
    c = _clahe(limite, lado).apply(luminancia(bgr)).astype(np.float32)
    return _normalizar(np.max(np.stack([np.abs(_hessiana(c, s)) for s in sigmas]), axis=0))


def _normalizar(r: np.ndarray) -> np.ndarray:
    """Leva a resposta para 0..255 cortando as caudas, para nao deixar um unico
    pixel extremo comprimir todo o resto da imagem contra o zero."""
    lo, hi = np.percentile(r, (1.0, 99.5))
    if hi <= lo:
        return np.zeros(r.shape, np.uint8)
    return np.clip((r - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


VARIANTES = {
    "a": ("gradiente direcional seguido de CLAHE", variante_a),
    "b": ("CLAHE seguido de gradiente de 8 direcoes", variante_b),
    "d": ("cristas multiescala por Hessiana", variante_d),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variante", choices=sorted(VARIANTES), required=True)
    p.add_argument("--clip-limit", type=float, default=3.0)
    p.add_argument("--tile", type=int, default=8)
    args = p.parse_args()

    rotulo, funcao = VARIANTES[args.variante]
    destino = RAIZ / "Dataset_YOLO" / f"dataset_realce_{args.variante}"
    (destino / "images").mkdir(parents=True, exist_ok=True)
    (destino / "labels").mkdir(parents=True, exist_ok=True)

    print(f"variante {args.variante}: {rotulo}")
    print(f"clipLimit {args.clip_limit}  tiles {args.tile}x{args.tile}")
    print(f"destino {destino}\n")

    imagens = sorted((ORIGEM / "images").glob("*.png"))
    for i, origem in enumerate(imagens, 1):
        bgr = cv2.imread(str(origem))
        if bgr is None:
            raise SystemExit(f"nao consegui ler {origem}")
        realce = funcao(bgr, args.clip_limit, args.tile)
        # O YOLOv8 espera tres canais, e os pesos do COCO foram treinados assim.
        # O realce e uma grandeza unica, entao vai replicado nos tres.
        cv2.imwrite(str(destino / "images" / origem.name),
                    cv2.merge([realce, realce, realce]))

        rot = ORIGEM / "labels" / (origem.stem + ".txt")
        alvo = destino / "labels" / rot.name
        if rot.is_file() and not alvo.exists():
            os.link(rot, alvo)   # ligacao rigida: nao duplica os rotulos em disco
        if i % 40 == 0 or i == len(imagens):
            print(f"  {i}/{len(imagens)}")

    n_img = len(list((destino / "images").glob("*.png")))
    n_rot = len(list((destino / "labels").glob("*.txt")))
    print(f"\n{n_img} imagens, {n_rot} rotulos")
    if n_img != len(imagens) or n_rot != len(imagens):
        raise SystemExit("contagem nao bate com a origem")
    print("OK")


if __name__ == "__main__":
    main()

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
ORIGEM = RAIZ / "Dataset_YOLO" / "dataset"   # reapontado por --pool

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


def oito_direcoes() -> list[np.ndarray]:
    """Os oito kernels de emboss, a 45 graus um do outro.

    Nao se obtem isso com np.rot90, que gira de 90 em 90: como a resposta e
    tomada em modulo, o giro de 180 graus coincide com o original e o de 270
    com o de 90, restando apenas DUAS direcoes distintas. As oito saem girando
    ciclicamente o anel de oito vizinhos do 3x3, uma posicao por vez.
    """
    # Honestidade sobre o alcance: tomado o modulo da resposta, as oito
    # reduzem-se a QUATRO eixos distintos, porque uma derivada direcional tem
    # eixo e nao sentido, e K e -K dao o mesmo modulo. Quatro e o maximo
    # geometrico de um kernel 3x3 com passo de 45 graus, e ainda assim e o
    # dobro das duas direcoes que np.rot90 entregava.
    anel = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
    valores = [EMBOSS[i, j] for i, j in anel]
    kernels = []
    for giro in range(8):
        k = np.zeros((3, 3), np.float32)
        for pos, (i, j) in enumerate(anel):
            k[i, j] = valores[(pos - giro) % 8]
        kernels.append(k)
    return kernels


KERNELS_8 = oito_direcoes()


def variante_b(bgr: np.ndarray, limite: float, lado: int, **_) -> np.ndarray:
    """CLAHE seguido de gradiente direcional de oito direcoes (Sawada2024).

    O artigo indica a tecnica ("8-directional emboss filter") mas nao publica os
    coeficientes; o kernel base adotado aqui e escolha deste trabalho.
    """
    c = _clahe(limite, lado).apply(luminancia(bgr)).astype(np.float32)
    respostas = [np.abs(cv2.filter2D(c, -1, k)) for k in KERNELS_8]
    return _normalizar(np.max(np.stack(respostas), axis=0))


BETA = 0.5   # sensibilidade a "quao alongada" e a estrutura, valor usual de Frangi


def frangi_escala(g: np.ndarray, sigma: float) -> np.ndarray:
    """Medida de Frangi numa escala, para cristas claras e escuras.

    Nao basta o maior autovalor da Hessiana: ele responde igual a uma crista
    fina e a um borrao. Frangi separa as duas com a razao entre os autovalores
    (Rb), que vale perto de zero numa estrutura alongada e perto de um num
    borrao, e pesa o resultado pela intensidade da estrutura (S), que suprime
    ruido em regiao lisa.
    """
    f = cv2.GaussianBlur(g, (0, 0), sigma)
    # normalizacao por sigma^2: sem ela, escalas maiores respondem sempre menos
    # e a combinacao multiescala fica dominada pela menor.
    s2 = sigma ** 2
    gxx = cv2.Sobel(f, cv2.CV_32F, 2, 0, ksize=3) * s2
    gyy = cv2.Sobel(f, cv2.CV_32F, 0, 2, ksize=3) * s2
    gxy = cv2.Sobel(f, cv2.CV_32F, 1, 1, ksize=3) * s2

    tr, det = gxx + gyy, gxx * gyy - gxy * gxy
    raiz = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    a, b = tr / 2 + raiz, tr / 2 - raiz
    # l1 e o de MENOR modulo, l2 o de maior, como na definicao de Frangi
    troca = np.abs(a) > np.abs(b)
    l1 = np.where(troca, b, a)
    l2 = np.where(troca, a, b)

    rb2 = (l1 / (l2 + 1e-10)) ** 2
    s = np.sqrt(l1 ** 2 + l2 ** 2)
    c = 0.5 * s.max() if s.max() > 0 else 1.0
    v = np.exp(-rb2 / (2 * BETA ** 2)) * (1.0 - np.exp(-(s ** 2) / (2 * c ** 2)))

    # Um risco costuma ser escuro sobre fundo claro (l2 > 0) e o whitening de
    # canto e claro sobre fundo escuro (l2 < 0). A carta tem os dois, entao a
    # resposta cobre as duas polaridades em vez de descartar uma delas.
    return np.where(np.isfinite(v), v, 0.0)


def variante_d(bgr: np.ndarray, limite: float, lado: int, sigmas=(1.5, 3.0, 5.0)) -> np.ndarray:
    """Medida de Frangi multiescala (Gruber2021, que a usa em inspecao de superficie).

    Risco e vinco sao cristas e vales, estruturas de segunda ordem, e nao
    bordas. As escalas cobrem a faixa de tamanhos medida no conjunto, cuja
    mediana da menor dimensao e 8,66 px na escala 1280; para uma crista de
    largura w o sigma util fica em torno de w/2. Os valores de sigma e o beta
    sao escolha deste trabalho: o artigo indica a tecnica, nao os parametros.
    """
    c = _clahe(limite, lado).apply(luminancia(bgr)).astype(np.float32)
    return _normalizar(np.max(np.stack([frangi_escala(c, s) for s in sigmas]), axis=0))


def _normalizar(r: np.ndarray) -> np.ndarray:
    """Leva a resposta para 0..255 cortando as caudas, para nao deixar um unico
    pixel extremo comprimir todo o resto da imagem contra o zero."""
    lo, hi = np.percentile(r, (1.0, 99.5))
    if hi <= lo:
        return np.zeros(r.shape, np.uint8)
    return np.clip((r - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def variante_e(bgr: np.ndarray, limite: float, lado: int, **_) -> np.ndarray:
    """Composto de tres canais informativos, em vez de um canal replicado.

    As variantes A, B e D SUBSTITUEM a imagem pelo realce e replicam o resultado
    nos tres canais. Isso descarta a informacao original, e os pesos pre-treinados
    no COCO passam a receber algo que nao se parece com fotografia. Aqui o realce
    e ACRESCENTADO: a rede recebe a imagem e o relevo ao mesmo tempo, e aprende
    sozinha a pesar os dois.

    Canal 0 (B): luminancia original, intacta
    Canal 1 (G): luminancia com CLAHE, contraste local normalizado
    Canal 2 (R): resposta de Frangi, o relevo
    """
    L = luminancia(bgr)
    c = _clahe(limite, lado).apply(L)
    relevo = _normalizar(np.max(np.stack(
        [frangi_escala(c.astype(np.float32), s) for s in (1.5, 3.0, 5.0)]), axis=0))
    return cv2.merge([L, c, relevo])


VARIANTES = {
    "a": ("gradiente direcional seguido de CLAHE", variante_a),
    "b": ("CLAHE seguido de gradiente de 8 direcoes", variante_b),
    "d": ("cristas multiescala por Hessiana", variante_d),
    "e": ("composto: L original, L com CLAHE, Frangi", variante_e),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variante", choices=sorted(VARIANTES), required=True)
    p.add_argument("--pool", default="pool",
                   help="pool de dados: pool (82 cartas) ou pool_206 (206 cartas)")
    p.add_argument("--clip-limit", type=float, default=3.0)
    p.add_argument("--tile", type=int, default=8)
    args = p.parse_args()

    rotulo, funcao = VARIANTES[args.variante]
    global ORIGEM
    sufixo = args.pool[len("pool"):]
    ORIGEM = RAIZ / "Dataset_YOLO" / f"dataset{sufixo}"
    destino = RAIZ / "Dataset_YOLO" / f"dataset{sufixo}_realce_{args.variante}"
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
        # O YOLOv8 espera tres canais. As variantes de canal unico vao replicadas;
        # a composta ja devolve os tres, cada um com informacao diferente.
        if realce.ndim == 2:
            realce = cv2.merge([realce, realce, realce])
        cv2.imwrite(str(destino / "images" / origem.name), realce)

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

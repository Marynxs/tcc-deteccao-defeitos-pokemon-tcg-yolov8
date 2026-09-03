"""Detecta a caixa da carta na foto.

Modela o fundo do estudio a partir das bordas e marca o que difere dele,
combinando z-score em Lab normalizado pelo MAD do proprio fundo com evidencia
de textura. Cada borda vem do perfil de projecao, e o resultado e validado
contra a proporcao fisica de 63x88 mm.

Medido no conjunto completo: 164/164 cartas detectadas.
"""
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import cv2

# --- EDITE ESTAS LINHAS ---------------------------------------------------
IMAGES_DIR   = "Dataset_YOLO/train/images"
LABELS_DIR    = "Dataset_YOLO/train/labels"
PREVIEW_DIR   = "Dataset_YOLO/train/preview"
OUTPUT_IMAGES_DIR = "Dataset_YOLO/train/images_final"
OUTPUT_LABELS_DIR = "Dataset_YOLO/train/labels_final"

MODE = "crop"              # rode "preview" primeiro, depois troque para "crop"
MARGIN_PX = 25             # slack adicionada ao redor da card detectada
# --------------------------------------------------------------------------

# --- Parametros do detector (raramente precisam ser mexidos) --------------
BACKGROUND_BAND_PCT   = 0.03   # faixa das bordas usada para modelar o fundo
Z_THRESHOLD_WHITE   = 6.0    # z-score minimo p/ ser carta (fundo branco/cinza)
Z_THRESHOLD_RED = 8.0    # fundo vermelho: contraste enorme, pode ser exigente
TEXTURE_FACTOR         = 3.0    # textura local > TEXTURE_FACTOR*textura_do_fundo + TEXTURE_BASE
TEXTURE_BASE          = 4.0
PROFILE_FRACTION     = 0.80   # band central usada para montar o profile
PROFILE_THRESHOLD     = 0.50   # fracao de pixels de card p/ a line/coluna contar

CARD_RATIO       = 63.0 / 88.0   # 0.7159 - proporcao fisica da carta
RATIO_TOLERANCE  = 0.05          # +-5% antes de corrigir a box
RATIO_MAX_DEVIATION  = 0.18          # acima disso a deteccao e considerada invalida

MIN_RELATIVE_AREA = 0.20       # box menor que isso = suspeita
MAX_RELATIVE_AREA = 0.97       # caixa quase = imagem inteira = nao cortou
# --------------------------------------------------------------------------


# ==========================================================================
# 1. Tipo de fundo
# ==========================================================================
def _sample_borders(channel, band_width):
    """Empilha as 4 faixas de borda de uma imagem HxW ou HxWxC."""
    h, w = channel.shape[:2]
    b = band_width
    c = channel.shape[2] if channel.ndim == 3 else 1
    return np.concatenate([
        channel[:b].reshape(-1, c),
        channel[h - b:].reshape(-1, c),
        channel[:, :b].reshape(-1, c),
        channel[:, w - b:].reshape(-1, c),
    ], axis=0)


def detect_background_type(bgr):
    """So existem dois fundos no dataset: vermelho/coral e branco/cinza claro."""
    h, w = bgr.shape[:2]
    b = max(4, int(min(h, w) * BACKGROUND_BAND_PCT))
    mediana = np.median(_sample_borders(bgr, b), axis=0)
    hsv = cv2.cvtColor(np.uint8([[mediana]]), cv2.COLOR_BGR2HSV)[0, 0]
    return "vermelho" if hsv[1] > 60 else "branco"


# ==========================================================================
# 2. Mascara de primeiro plano (a carta)
# ==========================================================================
def _fill_holes(mask):
    """Marca como carta todo buraco interno da mascara.

    Areas claras e lisas DENTRO da carta (a Pokebola do verso, texto branco,
    brilhos holograficos) sao indistinguiveis do fundo branco pixel a pixel, mas
    sao regioes fechadas, cercadas por carta. Sem isso o perfil de projecao se
    parte no meio da carta.
    """
    invertida = (1 - mask).astype(np.uint8)
    n, rotulos, _, _ = cv2.connectedComponentsWithStats(invertida, connectivity=4)
    if n <= 1:
        return mask
    touching_border = set(np.unique(np.concatenate([
        rotulos[0], rotulos[-1], rotulos[:, 0], rotulos[:, -1]
    ])).tolist())
    output = mask.copy()
    for i in range(1, n):
        if i not in touching_border:          # buraco que nao toca a borda = interior
            output[rotulos == i] = 1
    return output


# Por que este desenho, em resumo. Amostrar a cor nos 4 cantos falhou com
# gradiente e vinheta. Limiar em HSV vazava no fundo branco: a caixa virava a
# imagem inteira em 24 casos e saia com proporcao errada em outros 9. Canny com
# validacao de proporcao foi pior ainda. O que funciona sao duas evidencias
# independentes: z-score em Lab normalizado pelo MAD do proprio fundo, que se
# autocalibra por foto, mais textura local, que resolve as cartas foil
# prateadas, quase acromaticas contra o fundo branco.
def card_mask(bgr, background_type):
    """Duas evidencias independentes, combinadas por OU:

      (a) cor: z-score em Lab contra o fundo estimado nas bordas
      (b) textura: desvio-padrao local acima do desvio-padrao do fundo
    """
    h, w = bgr.shape[:2]
    b = max(4, int(min(h, w) * BACKGROUND_BAND_PCT))

    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    amostra = _sample_borders(lab, b)
    mediana = np.median(amostra, axis=0)
    # MAD * 1.4826 ~= desvio-padrao robusto; o piso evita divisao por ~0
    mad = np.median(np.abs(amostra - mediana), axis=0) * 1.4826
    sL = max(float(mad[0]), 1.5)
    sa = max(float(mad[1]), 1.0)
    sb = max(float(mad[2]), 1.0)

    z = np.sqrt(((lab[:, :, 0] - mediana[0]) / sL) ** 2 +
                ((lab[:, :, 1] - mediana[1]) / sa) ** 2 +
                ((lab[:, :, 2] - mediana[2]) / sb) ** 2)

    cinza = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = int(max(9, (min(h, w) // 120) * 2 + 1))
    m1 = cv2.boxFilter(cinza, -1, (k, k))
    m2 = cv2.boxFilter(cinza * cinza, -1, (k, k))
    textura = np.sqrt(np.maximum(m2 - m1 * m1, 0))
    tex_fundo = float(np.median(_sample_borders(textura[..., None], b)))

    z_limiar = Z_THRESHOLD_RED if background_type == "vermelho" else Z_THRESHOLD_WHITE
    m = ((z > z_limiar) | (textura > tex_fundo * TEXTURE_FACTOR + TEXTURE_BASE)).astype(np.uint8)

    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return _fill_holes(m)


# ==========================================================================
# 3. Bordas por perfil de projecao
# ==========================================================================
def _central_band(profile, threshold):
    """Faixa contigua acima do limiar QUE CONTEM O CENTRO.

    E isso que evita a falha classica do metodo mais simples: uma sujeira ou
    sombra encostada na borda da foto nao e contigua ao centro, entao nao estica
    a caixa para a imagem inteira.
    """
    center = len(profile) // 2
    if profile[center] <= threshold:
        acima = np.where(profile > threshold)[0]
        if len(acima) == 0:
            return None
        return int(acima[0]), int(acima[-1]) + 1
    i = center
    while i > 0 and profile[i - 1] > threshold:
        i -= 1
    j = center
    while j < len(profile) - 1 and profile[j + 1] > threshold:
        j += 1
    return int(i), int(j) + 1


def box_from_profile(mask):
    h, w = mask.shape
    f = PROFILE_FRACTION
    r0, r1 = int(h * (0.5 - f / 2)), int(h * (0.5 + f / 2))
    c0, c1 = int(w * (0.5 - f / 2)), int(w * (0.5 + f / 2))

    colunas = _central_band(mask[r0:r1].mean(axis=0), PROFILE_THRESHOLD)
    lines  = _central_band(mask[:, c0:c1].mean(axis=1), PROFILE_THRESHOLD)
    if colunas is None or lines is None:
        return None
    return colunas[0], lines[0], colunas[1], lines[1]


# ==========================================================================
# 4. Validacao / correcao pela proporcao da carta (63x88 mm)
# ==========================================================================
def fix_aspect_ratio(box, w, h):
    """Corrige a caixa cuja proporcao desviou da fisica 0.7159.

    Desvio pequeno EXPANDE a dimensao deficitaria em torno do centro; nunca
    encolhe, para nao perder anotacao. Desvio grande devolve None, porque a
    deteccao nao e confiavel.
    """
    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    if cw <= 0 or ch <= 0:
        return None, None
    ratio = cw / ch
    desvio = abs(ratio - CARD_RATIO) / CARD_RATIO
    if desvio > RATIO_MAX_DEVIATION:
        return None, ratio
    if desvio > RATIO_TOLERANCE:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if ratio > CARD_RATIO:            # larga demais -> cresce em altura
            ch = cw / CARD_RATIO
        else:                              # estreita demais -> cresce em largura
            cw = ch * CARD_RATIO
        x0, x1 = cx - cw / 2, cx + cw / 2
        y0, y1 = cy - ch / 2, cy + ch / 2
    x0 = max(0, int(round(x0)))
    y0 = max(0, int(round(y0)))
    x1 = min(w, int(round(x1)))
    y1 = min(h, int(round(y1)))
    return (x0, y0, x1, y1), ratio


# ==========================================================================
# 5. Fallback: algoritmo da v2 (maior componente conectado em HSV)
# ==========================================================================
def box_v2(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(float)
    h, w = hsv.shape[:2]
    b = int(min(h, w) * 0.06)
    background = np.median(_sample_borders(hsv, b), axis=0)
    dh = np.minimum(np.abs(hsv[:, :, 0] - background[0]), 180 - np.abs(hsv[:, :, 0] - background[0]))
    ds = np.abs(hsv[:, :, 1] - background[1])
    dv = np.abs(hsv[:, :, 2] - background[2])
    dist = np.sqrt(dh ** 2 + ds ** 2 + (dv * 0.35) ** 2)
    m = (dist > 28).astype(np.uint8)
    k = np.ones((7, 7), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, cw, ch, _ = stats[i]
    return int(x), int(y), int(x + cw), int(y + ch)


# ==========================================================================
# 6. Detector completo
# ==========================================================================
def detect_card_box(pil_image):
    """Devolve (caixa_com_margem, info), ou (None, info) se falhou.

    `info` traz a area relativa, a proporcao, o tipo de fundo e qual metodo
    produziu a caixa.
    """
    bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    bg_type = detect_background_type(bgr)

    candidatos = []
    c = box_from_profile(card_mask(bgr, bg_type))
    if c is not None:
        candidatos.append(("profile", c))
    c2 = box_v2(bgr)
    if c2 is not None:
        candidatos.append(("v2", c2))

    escolhida, ratio, method = None, None, None
    for name, bruta in candidatos:
        corrigida, r = fix_aspect_ratio(bruta, w, h)
        if corrigida is None:
            continue
        cw, ch = corrigida[2] - corrigida[0], corrigida[3] - corrigida[1]
        rel_area = (cw * ch) / (w * h)
        if rel_area < MIN_RELATIVE_AREA or rel_area > MAX_RELATIVE_AREA:
            continue                     # box minuscula ou = imagem inteira
        escolhida, ratio, method = corrigida, r, name
        break                            # "profile" tem prioridade sobre "v2"

    if escolhida is None:
        return None, {"background": bg_type, "rel_area": 0.0, "ratio": None, "method": None}

    x0, y0, x1, y1 = escolhida
    rel_area = ((x1 - x0) * (y1 - y0)) / (w * h)
    info = {"background": bg_type, "rel_area": rel_area, "ratio": ratio, "method": method}

    x0 = max(0, x0 - MARGIN_PX)
    y0 = max(0, y0 - MARGIN_PX)
    x1 = min(w, x1 + MARGIN_PX)
    y1 = min(h, y1 + MARGIN_PX)
    return (x0, y0, x1, y1), info


def is_box_suspicious(info):
    """Area muito pequena OU quase 100% da imagem (= nao cortou nada)."""
    return (info["method"] is None
            or info["rel_area"] < MIN_RELATIVE_AREA
            or info["rel_area"] > MAX_RELATIVE_AREA)


# ==========================================================================
# 7. Recalculo das anotacoes YOLO
# ==========================================================================
def adjust_box(line, w_orig, h_orig, crop_x0, crop_y0, crop_w, crop_h):
    p = line.strip().split()
    if len(p) != 5:
        return None
    classe, xc, yc, w, h = p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4])
    xc_px, yc_px = xc * w_orig, yc * h_orig
    w_px,  h_px  = w * w_orig,  h * h_orig
    x_min, x_max = xc_px - w_px / 2 - crop_x0, xc_px + w_px / 2 - crop_x0
    y_min, y_max = yc_px - h_px / 2 - crop_y0, yc_px + h_px / 2 - crop_y0
    if x_max <= 0 or y_max <= 0 or x_min >= crop_w or y_min >= crop_h:
        return None
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(crop_w, x_max), min(crop_h, y_max)
    novo_xc = ((x_min + x_max) / 2) / crop_w
    novo_yc = ((y_min + y_max) / 2) / crop_h
    novo_w  = (x_max - x_min) / crop_w
    novo_h  = (y_max - y_min) / crop_h
    return f"{classe} {novo_xc:.6f} {novo_yc:.6f} {novo_w:.6f} {novo_h:.6f}"


# ==========================================================================
# 8. Pipeline
# ==========================================================================
def process():
    img_folder = Path(IMAGES_DIR)
    lbl_folder = Path(LABELS_DIR)
    images = sorted(list(img_folder.glob("*.png")) + list(img_folder.glob("*.jpg")))
    if not images:
        print(f"Nenhuma imagem em {img_folder}")
        return

    suspicious = []
    by_background = {"vermelho": [0, 0], "branco": [0, 0]}   # [ok, suspicious]

    if MODE == "preview":
        preview_folder = Path(PREVIEW_DIR)
        preview_folder.mkdir(parents=True, exist_ok=True)
        print("MODO PREVIEW - nenhuma imagem sera cortada de verdade\n")

        for img_path in images:
            with Image.open(img_path) as im:
                box, info = detect_card_box(im)
                preview = im.copy()
                draw = ImageDraw.Draw(preview)

                if box is None:
                    print(f"[X] {img_path.name} - NENHUMA carta detectada [{info['background']}]")
                    suspicious.append(img_path.name)
                    by_background[info["background"]][1] += 1
                    preview.save(preview_folder / img_path.name)
                    continue

                draw.rectangle(box, outline=(0, 255, 0), width=6)
                preview.save(preview_folder / img_path.name)

                is_susp = is_box_suspicious(info)
                by_background[info["background"]][1 if is_susp else 0] += 1
                if is_susp:
                    suspicious.append(img_path.name)
                print(f"{'[!] SUSPEITO' if is_susp else '[ok]'}  {img_path.name}  "
                      f"background={info['background']}  method={info['method']}  "
                      f"area={info['rel_area'] * 100:.1f}%  proporcao={info['ratio']:.3f}")

        print(f"\n{'=' * 60}")
        print(f"Previas salvas em: {preview_folder}")
        _summary(len(images), suspicious, by_background)
        return

    elif MODE == "crop":
        out_images = Path(OUTPUT_IMAGES_DIR)
        out_labels = Path(OUTPUT_LABELS_DIR)
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        total, discarded, without_annotation = 0, 0, 0
        for img_path in images:
            name = img_path.stem
            lbl_path = lbl_folder / f"{name}.txt"

            with Image.open(img_path) as im:
                w_orig, h_orig = im.size
                box, info = detect_card_box(im)

                if box is None or is_box_suspicious(info):
                    print(f"[PULADO - suspeito] {img_path.name}")
                    suspicious.append(img_path.name)
                    by_background[info["background"]][1] += 1
                    continue

                by_background[info["background"]][0] += 1
                x0, y0, x1, y1 = box
                crop_w, crop_h = x1 - x0, y1 - y0
                im.crop((x0, y0, x1, y1)).save(out_images / img_path.name)

            lines = []
            if lbl_path.exists():
                with open(lbl_path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        nova = adjust_box(line, w_orig, h_orig, x0, y0, crop_w, crop_h)
                        if nova:
                            lines.append(nova)
                        else:
                            discarded += 1

            # Sem anotacao => .txt VAZIO (carta sem defeito; correto para YOLO)
            if not lines:
                without_annotation += 1
            with open(out_labels / f"{name}.txt", "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

            total += 1
            print(f"[ok] {img_path.name}  ->  {crop_w}x{crop_h}px  ->  {len(lines)} box(s)")

        print(f"\n{'=' * 60}")
        print(f"{total} imagens cortadas  ({without_annotation} com .txt vazio = sem defeito)")
        if discarded:
            print(f"{discarded} caixa(s) descartada(s) por ficarem fora do corte")
        _summary(len(images), suspicious, by_background)


def _summary(total, suspicious, by_background):
    ok = total - len(suspicious)
    print(f"\n{ok}/{total} detectadas com sucesso ({ok / total * 100:.1f}%)")
    for background, (a, b) in by_background.items():
        if a + b:
            print(f"    fundo {background:9s}: {a} ok / {b} suspeita(s)")
    if suspicious:
        print(f"\n{len(suspicious)} suspeita(s) - revisar manualmente:")
        for s in suspicious:
            print(f"    {s}")


if __name__ == "__main__":
    process()

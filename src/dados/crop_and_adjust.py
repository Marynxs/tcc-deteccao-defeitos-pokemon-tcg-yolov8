"""Corta a foto na carta detectada e recalcula as anotacoes YOLO.

Etapa 2 do pipeline. Reaproveita card_detector.detect_card_box, de modo que
deteccao e corte sempre concordam, e converte cada anotacao para o novo
enquadramento, clipando as que ficam parcialmente fora em vez de descarta-las.

Fotos que ja chegam recortadas na carta (sem fundo) passam inteiras, e um
corte que descartaria anotacao e rejeitado: anotacao nunca fica fora da carta.

Medido nas 82 cartas: 1038/1038 anotacoes preservadas, 0 descartadas.
"""
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

import card_detector as detector

IMAGES_DIR   = "Dataset_YOLO/train/images"
LABELS_DIR    = "Dataset_YOLO/train/labels"
PREVIEW_DIR   = "Dataset_YOLO/train/preview_corte"
OUTPUT_IMAGES_DIR = "Dataset_YOLO/train/images_final"
OUTPUT_LABELS_DIR = "Dataset_YOLO/train/labels_final"

MODE = "crop"         # "preview" = so desenha as caixas recalculadas
                      # "crop"    = grava as imagens cortadas + os labels recalculados

# Medido no conjunto anotado, distancia de cada caixa ate a borda detectada da
# carta, em escala de 1280px: p0 = -13.2 px, p1 = -4.9, p5 = 2.0, p25 = 11.7,
# mediana = 51.9. Trinta e oito anotacoes ficam FORA da borda detectada, porque
# o whitening de canto e anotado vazando para fora da carta. Cortar rente
# decapitaria justamente os defeitos que mais importam.
MARGIN_PX = 25        # folga de seguranca alem da borda detectada da carta.
                      # Medido: a anotacao mais "vazada" passa 13px da borda,
                      # entao 25 deixa ~12px de folga. Nao baixe de 20 sem
                      # rodar de novo a auditoria de folga (--audit-margin).
# --------------------------------------------------------------------------


# ==========================================================================
# Recalculo de uma anotacao YOLO para o novo enquadramento
# ==========================================================================
def adjust_box(line, w_orig, h_orig, crop_x0, crop_y0, crop_w, crop_h):
    """
    normalizada(original) -> pixels(original) -> desloca pelo corte ->
    clipped nos limites -> normalizada(cortada).

    Devolve (texto_yolo, was_clipped) ou (None, False) se a box estiver
    100% fora do corte (unico caso em que a anotacao e descartada).
    """
    p = line.strip().split()
    if len(p) != 5:
        return None, False

    classe = p[0]
    xc, yc, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])

    # 1) normalizada -> pixels na imagem ORIGINAL
    xc_px, yc_px = xc * w_orig, yc * h_orig
    w_px,  h_px  = w * w_orig,  h * h_orig

    # 2) desloca pela origem do corte
    x_min, x_max = xc_px - w_px / 2 - crop_x0, xc_px + w_px / 2 - crop_x0
    y_min, y_max = yc_px - h_px / 2 - crop_y0, yc_px + h_px / 2 - crop_y0

    # descarta SOMENTE se estiver inteiramente fora do crop_img
    if x_max <= 0 or y_max <= 0 or x_min >= crop_w or y_min >= crop_h:
        return None, False

    # 3) clip: caixa parcialmente fora vira a parte visivel, nao e descartada
    cx_min, cy_min = max(0.0, x_min), max(0.0, y_min)
    cx_max, cy_max = min(float(crop_w), x_max), min(float(crop_h), y_max)
    was_clipped = (cx_min, cy_min, cx_max, cy_max) != (x_min, y_min, x_max, y_max)

    if cx_max - cx_min <= 1 or cy_max - cy_min <= 1:
        return None, False        # sobrou uma fatia degenerada = fora

    # 4) renormaliza pelas NOVAS dimensoes
    novo_xc = ((cx_min + cx_max) / 2) / crop_w
    novo_yc = ((cy_min + cy_max) / 2) / crop_h
    novo_w  = (cx_max - cx_min) / crop_w
    novo_h  = (cy_max - cy_min) / crop_h

    return f"{classe} {novo_xc:.6f} {novo_yc:.6f} {novo_w:.6f} {novo_h:.6f}", was_clipped


def read_labels(lbl_path):
    if not lbl_path.exists():
        return []
    with open(lbl_path) as f:
        return [l for l in f if l.strip()]


ANNOTATION_PAD_PX = 5     # folga ao redor de anotacao que forca o corte a crescer


def union_with_annotations(box, lines, w_orig, h_orig):
    """Expande a caixa de corte ate conter toda anotacao, com folga."""
    x0, y0, x1, y1 = box
    for line in lines:
        p = line.strip().split()
        if len(p) != 5:
            continue
        xc, yc, w, h = (float(v) for v in p[1:])
        bx0 = xc * w_orig - w * w_orig / 2 - ANNOTATION_PAD_PX
        by0 = yc * h_orig - h * h_orig / 2 - ANNOTATION_PAD_PX
        bx1 = xc * w_orig + w * w_orig / 2 + ANNOTATION_PAD_PX
        by1 = yc * h_orig + h * h_orig / 2 + ANNOTATION_PAD_PX
        x0, y0 = min(x0, int(bx0)), min(y0, int(by0))
        x1, y1 = max(x1, int(bx1) + 1), max(y1, int(by1) + 1)
    return max(0, x0), max(0, y0), min(w_orig, x1), min(h_orig, y1)


def recalculate_all(lines, w_orig, h_orig, x0, y0, crop_w, crop_h):
    new_lines, n_clipped, n_discarded = [], 0, 0
    for line in lines:
        nova, clipped = adjust_box(line, w_orig, h_orig, x0, y0, crop_w, crop_h)
        if nova is None:
            n_discarded += 1
        else:
            new_lines.append(nova)
            n_clipped += int(clipped)
    return new_lines, n_clipped, n_discarded


# ==========================================================================
# Pipeline
# ==========================================================================
def process():
    detector.MARGIN_PX = MARGIN_PX          # a Etapa 2 e a dona da margem

    img_folder = Path(IMAGES_DIR)
    lbl_folder = Path(LABELS_DIR)
    images = sorted(list(img_folder.glob("*.png")) + list(img_folder.glob("*.jpg")))
    if not images:
        print(f"Nenhuma imagem em {img_folder}")
        return

    preview_folder = Path(PREVIEW_DIR)
    out_images = Path(OUTPUT_IMAGES_DIR)
    out_labels = Path(OUTPUT_LABELS_DIR)

    if MODE == "preview":
        preview_folder.mkdir(parents=True, exist_ok=True)
        print(f"MODO PREVIEW - nada e gravado em {OUTPUT_IMAGES_DIR}/{OUTPUT_LABELS_DIR}\n")
    else:
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)
        print(f"MODO CROP - gravando em {OUTPUT_IMAGES_DIR} e {OUTPUT_LABELS_DIR}\n")

    skipped, without_annotation = [], 0
    total, total_clipped, total_discarded, total_boxes = 0, 0, 0, 0
    discarded_per_image = []
    dims_before, dims_after = [], []
    occ_before, occ_after = [], []

    for img_path in images:
        name = img_path.stem
        lbl_path = lbl_folder / f"{name}.txt"

        lines = read_labels(lbl_path)

        with Image.open(img_path) as im:
            w_orig, h_orig = im.size
            box, info = detector.detect_card_box(im)

            # Fotos que ja chegam recortadas na carta nao tem fundo para o
            # detector achar. Sem deteccao a imagem passa INTEIRA, em vez de
            # ser pulada: a carta ja ocupa o quadro todo, nao ha o que cortar.
            passthrough_reason = None
            if box is None or detector.is_box_suspicious(info):
                passthrough_reason = "sem deteccao" if box is None else f"area={info['rel_area']*100:.1f}%"
                box = (0, 0, w_orig, h_orig)
                info = dict(info, rel_area=1.0)

            x0, y0, x1, y1 = box
            crop_w, crop_h = x1 - x0, y1 - y0
            new_lines, n_clipped, n_discarded = recalculate_all(lines, w_orig, h_orig, x0, y0, crop_w, crop_h)

            # Uma anotacao nunca fica fora da carta. Se o corte descartaria
            # alguma, a deteccao entrou PARA DENTRO da carta (acontece quando a
            # foto ja veio recortada e a borda escura da carta e lida como
            # fundo). Nesse caso o corte esta errado, e a imagem passa inteira.
            if n_discarded and passthrough_reason is None:
                passthrough_reason = f"{n_discarded} anotacao(oes) cairia(m) fora"
                x0, y0, x1, y1 = 0, 0, w_orig, h_orig
                info = dict(info, rel_area=1.0)

            # Erro fino: o corte entra so um pouco numa anotacao. Em fundo
            # branco o detector as vezes poe a borda da carta DENTRO da faixa
            # cinza do contorno, e um defeito anotado ali seria decapitado. A
            # caixa cresce ate conter toda anotacao, com folga; onde nenhuma
            # anotacao encosta na borda do corte, nada muda.
            elif n_clipped:
                x0, y0, x1, y1 = union_with_annotations(box, lines, w_orig, h_orig)

            if passthrough_reason or n_clipped:
                crop_w, crop_h = x1 - x0, y1 - y0
                new_lines, n_clipped, n_discarded = recalculate_all(lines, w_orig, h_orig, x0, y0, crop_w, crop_h)

            if passthrough_reason:
                skipped.append((img_path.name, passthrough_reason))

            crop_img = im.crop((x0, y0, x1, y1))

        total_boxes += len(lines)
        total_clipped += n_clipped
        total_discarded += n_discarded
        if n_discarded:
            discarded_per_image.append((img_path.name, n_discarded))

        # area da CARTA em pixels (info["rel_area"] e medido antes da margem)
        area_carta = info["rel_area"] * w_orig * h_orig
        occ_before.append(area_carta / (w_orig * h_orig))
        occ_after.append(area_carta / (crop_w * crop_h))
        dims_before.append((w_orig, h_orig))
        dims_after.append((crop_w, crop_h))

        if MODE == "preview":
            # Desenha as caixas RECALCULADAS sobre a imagem CORTADA. Se elas
            # caem em cima dos defeitos aqui, o recalculo esta certo.
            draw = ImageDraw.Draw(crop_img)
            thickness = max(2, int(min(crop_w, crop_h) / 400))
            for l in new_lines:
                _, xc, yc, ww, hh = l.split()
                xc, yc, ww, hh = float(xc), float(yc), float(ww), float(hh)
                draw.rectangle(
                    [(xc - ww / 2) * crop_w, (yc - hh / 2) * crop_h,
                     (xc + ww / 2) * crop_w, (yc + hh / 2) * crop_h],
                    outline=(255, 0, 0), width=thickness)
            crop_img.save(preview_folder / img_path.name)
        else:
            crop_img.save(out_images / img_path.name)
            # Sem anotacao => .txt VAZIO (carta sem defeito). Para o YOLO essas
            # imagens sao exemplos negativos, nao lixo.
            with open(out_labels / f"{name}.txt", "w") as f:
                f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

        if not new_lines:
            without_annotation += 1
        total += 1

        marca = ""
        if n_discarded:
            marca += f"  [!] {n_discarded} descartada(s)"
        if n_clipped:
            marca += f"  [clipped] {n_clipped}"
        print(f"[ok] {img_path.name}  {w_orig}x{h_orig} -> {crop_w}x{crop_h}  "
              f"{len(new_lines)} box(s){marca}")

    _report(total, len(images), dims_before, dims_after, occ_before, occ_after,
               total_boxes, total_clipped, total_discarded, discarded_per_image,
               without_annotation, skipped)


def _report(total, n_entrada, dims_before, dims_after, occ_before, occ_after,
               total_boxes, total_clipped, total_discarded, discarded_per_image,
               without_annotation, skipped):
    print(f"\n{'=' * 66}")
    print("RELATORIO")
    print(f"{'=' * 66}")
    if not total:
        print("Nenhuma imagem processada.")
        return

    da = np.array(dims_before, dtype=float)
    dd = np.array(dims_after, dtype=float)
    oa = np.array(occ_before)
    od = np.array(occ_after)
    gain = (od / oa - 1) * 100

    print(f"Imagens de entrada ................. {n_entrada}")
    print(f"Processadas ........................ {total}")
    print(f"  cortadas ......................... {total - len(skipped)}")
    print(f"  passadas inteiras (sem corte) .... {len(skipped)}")
    print()
    print(f"Dimensao media ANTES ............... {da[:,0].mean():.0f} x {da[:,1].mean():.0f} px")
    print(f"Dimensao media DEPOIS .............. {dd[:,0].mean():.0f} x {dd[:,1].mean():.0f} px")
    print(f"Area media descartada .............. {(1 - (dd[:,0]*dd[:,1]).mean()/(da[:,0]*da[:,1]).mean())*100:.1f}% do quadro")
    print()
    print(f"Ocupacao da carta ANTES ............ {oa.mean()*100:.1f}%  (min {oa.min()*100:.1f}%)")
    print(f"Ocupacao da carta DEPOIS ........... {od.mean()*100:.1f}%  (min {od.min()*100:.1f}%)")
    print(f"GANHO MEDIO DE OCUPACAO ............ +{gain.mean():.1f}%   "
          f"(min +{gain.min():.1f}% / max +{gain.max():.1f}%)")
    print(f"Ganho equivalente em resolucao ..... {np.sqrt(od/oa).mean():.3f}x linear "
          f"(o defeito fica {(np.sqrt(od/oa).mean()-1)*100:.1f}% maior apos o resize)")
    print()
    print(f"Anotacoes de entrada ............... {total_boxes}")
    print(f"Anotacoes recalculadas ............. {total_boxes - total_discarded}")
    print(f"Anotacoes clipadas (parciais) ...... {total_clipped}")
    print(f"Anotacoes DESCARTADAS (100% fora) .. {total_discarded}", end="")
    print("   <- deve ser 0" if total_discarded == 0 else "   <<< CORTE AGRESSIVO DEMAIS, aumente MARGIN_PX")
    for name, n in discarded_per_image:
        print(f"      {name}: {n}")
    print()
    print(f"Imagens com .txt vazio (sem defeito) {without_annotation}")
    if skipped:
        print(f"\nImagens PASSADAS INTEIRAS (a carta ja ocupava o quadro):")
        for name, reason in skipped:
            print(f"    {name}  ({reason})")
    else:
        print("\nToda imagem foi cortada.")


# ==========================================================================
# Auditoria de margem: quanto da para apertar o corte sem cortar defeito
# ==========================================================================
def audit_margin(margins=(0, 10, 15, 20, 25, 30, 40)):
    """
    Para cada margem candidate, mede a ocupacao media da card no quadro e
    quantas anotacoes seriam CLIPADAS pela borda do corte.

    Rode isto antes de mexer em MARGIN_PX.
    """
    img_folder = Path(IMAGES_DIR)
    lbl_folder = Path(LABELS_DIR)
    images = sorted(list(img_folder.glob("*.png")) + list(img_folder.glob("*.jpg")))

    detector.MARGIN_PX = 0
    cache = []
    for img_path in images:
        with Image.open(img_path) as im:
            w, h = im.size
            box, info = detector.detect_card_box(im)
        if box is None or detector.is_box_suspicious(info):
            continue
        boxes = []
        for l in read_labels(lbl_folder / f"{img_path.stem}.txt"):
            p = l.split()
            if len(p) != 5:
                continue
            xc, yc, ww, hh = map(float, p[1:])
            boxes.append(((xc - ww / 2) * w, (yc - hh / 2) * h,
                           (xc + ww / 2) * w, (yc + hh / 2) * h))
        cache.append((w, h, box, info["rel_area"] * w * h, boxes))

    print(f"\n{'margem':>7} {'ocupacao':>10} {'ganho':>8} {'anotacoes clipadas':>20} {'folga minima':>14}")
    print("-" * 64)
    for m in margins:
        oc, n_clipped, min_slack = [], 0, 1e9
        for w, h, (x0, y0, x1, y1), area, boxes in cache:
            X0, Y0 = max(0, x0 - m), max(0, y0 - m)
            X1, Y1 = min(w, x1 + m), min(h, y1 + m)
            oc.append(area / ((X1 - X0) * (Y1 - Y0)))
            scale_f = 1280.0 / w
            for bx in boxes:
                slack = min(bx[0] - X0, bx[1] - Y0, X1 - bx[2], Y1 - bx[3]) * scale_f
                min_slack = min(min_slack, slack)
                if slack < 0:
                    n_clipped += 1
        oc = np.array(oc)
        alert = "" if n_clipped == 0 else "   <<< corta defeito anotado"
        print(f"{m:>7} {oc.mean()*100:>9.1f}% {oc.mean()/np.mean([a/(w*h) for w,h,_,a,_ in cache]):>7.3f}x "
              f"{n_clipped:>19} {min_slack:>12.1f}px{alert}")
    print("\n(slack minima em px na scale_f 1280 de largura; negativa = a box\n"
          " anotada seria cortada pela borda do crop_img)")


if __name__ == "__main__":
    import sys
    if "--audit-margin" in sys.argv:
        audit_margin()
    else:
        process()

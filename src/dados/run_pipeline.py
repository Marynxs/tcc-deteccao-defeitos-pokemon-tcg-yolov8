"""Roda as duas etapas do pipeline sobre o pool completo de 164 imagens.

So sobrescreve os caminhos de entrada e saida de card_detector e
crop_and_adjust; o comportamento validado deles fica intacto.

    python src/dados/run_pipeline.py                # preview + corte
    python src/dados/run_pipeline.py --stage2-only  # so o corte
    python src/dados/run_pipeline.py --audit-margin # auditoria de margem
"""
import sys

import card_detector as detector
import crop_and_adjust as cropper

POOL_IMAGES = "Dataset_YOLO/pool/images"
POOL_LABELS = "Dataset_YOLO/pool/labels"


# A Etapa 1 roda em preview de proposito. Em crop ela gravaria nas mesmas pastas
# de saida da Etapa 2 e, pior, DESCARTARIA as anotacoes parcialmente fora do
# recorte em vez de clipa-las, perdendo as 38 caixas de whitening de canto que a
# Etapa 2 existe para preservar. Quem grava o dataset final e a Etapa 2.
def _point_to_stage1():
    detector.IMAGES_DIR = POOL_IMAGES
    detector.LABELS_DIR = POOL_LABELS
    detector.PREVIEW_DIR = "Dataset_YOLO/pool/preview"
    detector.MODE = "preview"


def _point_to_stage2():
    cropper.IMAGES_DIR = POOL_IMAGES
    cropper.LABELS_DIR = POOL_LABELS
    cropper.PREVIEW_DIR = "Dataset_YOLO/pool/preview_corte"
    cropper.OUTPUT_IMAGES_DIR = "Dataset_YOLO/pool/images_final"
    cropper.OUTPUT_LABELS_DIR = "Dataset_YOLO/pool/labels_final"
    cropper.MODE = "crop"


if __name__ == "__main__":
    if "--audit-margin" in sys.argv:
        _point_to_stage2()
        cropper.audit_margin()
    elif "--stage2-only" in sys.argv:
        _point_to_stage2()
        cropper.process()
    else:
        print("=" * 70)
        print("ETAPA 1 - card_detector (preview da deteccao)")
        print("=" * 70)
        _point_to_stage1()
        detector.process()

        print("\n" + "=" * 70)
        print("ETAPA 2 - crop_and_adjust (corte + recalculo das anotacoes)")
        print("=" * 70)
        _point_to_stage2()
        cropper.process()

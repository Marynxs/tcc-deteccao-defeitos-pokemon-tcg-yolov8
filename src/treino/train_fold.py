"""
Treina uma configuracao do fatorial 2x2 em um fold e avalia no fold retido.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
FOLDS = ROOT / "Dataset_YOLO" / "folds"
RUNS = ROOT / "Dataset_YOLO" / "runs"

SEED = 42
K = 5

# Os quatro dicionarios abaixo reproduzem a Tabela 4 do TCC2. Qualquer divergencia
# entre esta tabela e o codigo invalida o fatorial, entao o script imprime os
# valores resolvidos antes de treinar para permitir a conferencia a olho.
#
# Por que zerar explicitamente: os padroes da Ultralytics ja trazem aumento de
# dados agressivo (mosaic=1.0, scale=0.5, hsv_s=0.7, hsv_v=0.4, translate=0.1,
# fliplr=0.5). Rodar a configuracao "sem aumento de dados" com os padroes daria
# a ela MAIS aumento do que a configuracao "com aumento" descreve, e o fator
# medido pelo fatorial passaria a ser ruido.
SEM_AUMENTO = dict(
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0, translate=0.0, fliplr=0.0,
)

COM_AUMENTO_RGB = dict(
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=10.0, translate=0.1, fliplr=0.5,
)

# Na entrada de realce de micro-relevo o CLAHE ja normalizou o contraste local,
# entao jitter de cor perturbaria justamente o sinal que o realce construiu.
# Restam apenas as transformacoes geometricas.
COM_AUMENTO_REALCE = dict(
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=10.0, translate=0.1, fliplr=0.5,
)

# Desligados em TODAS as configuracoes, inclusive nas que usam aumento de dados.
# mosaic junta quatro imagens em uma e reduz cada uma a metade da dimensao
# linear: um defeito de 8 px viraria 4 px, abaixo do stride 8 da camada P3.
# scale=0.5 admite a mesma reducao por outro caminho. Os demais ja valem 0,0
# por padrao e sao repetidos aqui so para deixar o protocolo explicito.
SEMPRE_DESLIGADOS = dict(
    mosaic=0.0, scale=0.0,
    shear=0.0, perspective=0.0, flipud=0.0, bgr=0.0,
    mixup=0.0, cutmix=0.0, copy_paste=0.0,
    close_mosaic=0,
)

RGB = "dataset"
REALCE = "dataset_realce"

CONFIGS = {
    1: ("RGB sem aumento", RGB, SEM_AUMENTO),
    2: ("RGB com aumento", RGB, COM_AUMENTO_RGB),
    3: ("realce sem aumento", REALCE, SEM_AUMENTO),
    4: ("realce com aumento", REALCE, COM_AUMENTO_REALCE),
}


def montar_yaml(destino: Path, treino: Path, validacao: Path) -> Path:
    """Escreve um data.yaml apontando para as listas de imagens dadas."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        yaml.safe_dump(
            {
                "train": str(treino),
                "val": str(validacao),
                "nc": 1,
                "names": ["defeito"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return destino


def resolver(config: int, fold: int) -> tuple[str, dict, dict]:
    if config not in CONFIGS:
        raise SystemExit(f"configuracao {config} nao existe, use 1..4")
    if not 0 <= fold < K:
        raise SystemExit(f"fold {fold} fora da faixa 0..{K - 1}")

    rotulo, origem, aumento = CONFIGS[config]
    if not (ROOT / "Dataset_YOLO" / origem / "images").is_dir():
        raise SystemExit(
            f"a configuracao {config} ({rotulo}) precisa de "
            f"Dataset_YOLO/{origem}/images, que ainda nao foi gerado"
        )

    hiper = dict(SEMPRE_DESLIGADOS)
    hiper.update(aumento)
    return rotulo, origem, hiper


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=int, required=True, help="1..4 do fatorial")
    p.add_argument("--fold", type=int, required=True, help="0..4")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--model", default="yolov8s.pt")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache", default="disk", help="disk, ram ou vazio")
    p.add_argument("--no-amp", action="store_true", help="treina em fp32")
    p.add_argument("--name", default=None)
    p.add_argument("--dry-run", action="store_true", help="so mostra o que faria")
    args = p.parse_args()

    rotulo, origem, hiper = resolver(args.config, args.fold)
    nome = args.name or f"cfg{args.config}_fold{args.fold}_{args.imgsz}"
    saida = RUNS / nome

    treino = FOLDS / f"fold{args.fold}_treino.txt"
    val_interna = FOLDS / f"fold{args.fold}_val_interna.txt"
    avaliacao = FOLDS / f"fold{args.fold}_avaliacao.txt"
    for lista in (treino, val_interna, avaliacao):
        if not lista.is_file():
            raise SystemExit(f"lista ausente: {lista}")

    print(f"\nconfiguracao {args.config}: {rotulo}")
    print(f"fold          {args.fold}")
    print(f"imagens       Dataset_YOLO/{origem}/images")
    print(f"modelo        {args.model}   imgsz {args.imgsz}   batch {args.batch}")
    print(f"epocas        {args.epochs} (patience {args.patience})")
    print(f"treino        {sum(1 for _ in treino.open())} imagens")
    print(f"val interna   {sum(1 for _ in val_interna.open())} imagens")
    print(f"avaliacao     {sum(1 for _ in avaliacao.open())} imagens (fold retido)")
    print("\nhiperparametros de aumento de dados:")
    for chave in sorted(hiper):
        print(f"  {chave:<13} {hiper[chave]}")

    if args.dry_run:
        print("\n--dry-run: nada foi treinado")
        return

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    modelo = YOLO(args.model)
    inicio = time.time()
    modelo.train(
        data=str(montar_yaml(saida / "treino.yaml", treino, val_interna)),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        cache=args.cache or False,
        seed=SEED,
        deterministic=True,
        amp=not args.no_amp,
        # A Ultralytics liga o formato NHWC sozinha quando este parametro fica
        # em None e o dispositivo se reporta como CUDA (trainer.py, linha 321),
        # o que inclui o ROCm. Em placas NVIDIA isso alimenta os Tensor Cores e
        # compensa. Medido aqui, yolov8s a 1280 com batch 4:
        #   torch 2.10.0+rocm7.0  0,21 s/it desligado  contra  20,99 s/it ligado
        #   torch 2.14.0+rocm7.2  0,15 s/it desligado  contra   0,15 s/it ligado
        # Ou seja, na pilha antiga o MIOpen do gfx1200 nao tinha convolucao NHWC
        # e caia num caminho de transposicao cem vezes mais lento; na pilha atual
        # o formato deixou de importar. Fica desligado de proposito: nao custa
        # nada hoje e nao depende de um comportamento automatico da biblioteca.
        channels_last=False,
        project=str(RUNS),
        name=nome,
        exist_ok=True,
        plots=True,
        **hiper,
    )
    duracao = time.time() - inicio

    # A avaliacao do fold retido NAO pode passar pelo yaml do treino: aquele
    # arquivo aponta para a validacao interna, que serviu ao early stopping e
    # portanto ja influenciou o modelo.
    melhor = YOLO(saida / "weights" / "best.pt")
    metricas = melhor.val(
        data=str(montar_yaml(saida / "avaliacao.yaml", treino, avaliacao)),
        imgsz=args.imgsz,
        batch=args.batch,
        channels_last=False,
        project=str(RUNS),
        name=f"{nome}_avaliacao",
        exist_ok=True,
        plots=True,
    )

    pico = (
        torch.cuda.max_memory_reserved() / 1024**3
        if torch.cuda.is_available()
        else 0.0
    )
    epocas_reais = len(
        (saida / "results.csv").read_text(encoding="utf-8").strip().splitlines()
    ) - 1

    resumo = {
        "config": args.config,
        "rotulo": rotulo,
        "fold": args.fold,
        "modelo": args.model,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "epocas_pedidas": args.epochs,
        "epocas_rodadas": epocas_reais,
        "segundos_total": round(duracao, 1),
        "segundos_por_epoca": round(duracao / max(epocas_reais, 1), 1),
        "pico_vram_gb": round(pico, 2),
        "map50": round(float(metricas.box.map50), 4),
        "map50_95": round(float(metricas.box.map), 4),
        "precisao": round(float(metricas.box.mp), 4),
        "recall": round(float(metricas.box.mr), 4),
        "aumento": hiper,
    }
    (saida / "resumo.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{'=' * 60}")
    print(f"epocas rodadas     {epocas_reais}")
    print(f"tempo total        {duracao / 60:.1f} min")
    print(f"tempo por epoca    {duracao / max(epocas_reais, 1):.1f} s")
    print(f"pico de VRAM       {pico:.2f} GB de 15,92")
    print(f"mAP@50 (retido)    {resumo['map50']:.4f}")
    print(f"mAP@50-95 (retido) {resumo['map50_95']:.4f}")
    print(f"recall (retido)    {resumo['recall']:.4f}")
    print(f"\nresumo em {saida / 'resumo.json'}")


if __name__ == "__main__":
    main()

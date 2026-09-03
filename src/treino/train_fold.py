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
# Qual variante de realce usar. O teste preliminar compara tres; depois de
# escolhida, a vencedora fica fixa nas Configuracoes 3 e 4.
REALCE_PADRAO = "d"

CONFIGS = {
    1: ("RGB sem aumento", RGB, SEM_AUMENTO),
    2: ("RGB com aumento", RGB, COM_AUMENTO_RGB),
    3: ("realce sem aumento", None, SEM_AUMENTO),
    4: ("realce com aumento", None, COM_AUMENTO_REALCE),
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


def resolver(config: int, fold: int, realce: str = REALCE_PADRAO) -> tuple[str, dict, dict]:
    if config not in CONFIGS:
        raise SystemExit(f"configuracao {config} nao existe, use 1..4")
    if not 0 <= fold < K:
        raise SystemExit(f"fold {fold} fora da faixa 0..{K - 1}")

    rotulo, origem, aumento = CONFIGS[config]
    if origem is None:
        origem = f"dataset_realce_{realce}"
        rotulo = f"{rotulo} (variante {realce})"
    if not (ROOT / "Dataset_YOLO" / origem / "images").is_dir():
        raise SystemExit(
            f"a configuracao {config} ({rotulo}) precisa de "
            f"Dataset_YOLO/{origem}/images, que ainda nao foi gerado"
        )

    hiper = dict(SEMPRE_DESLIGADOS)
    hiper.update(aumento)
    return rotulo, origem, hiper


# Lote maximo que roda estavel nesta placa por resolucao. Nao e limite de
# memoria: com batch 4 a 1280 o pico e de 7,1 GB dos 15,9 GB, e batch 8 chegaria
# a cerca de 12,5 GB, que caberia. O que acontece a batch 8 e o MIOpen falhar ao
# lancar a convolucao (miopenStatusInternalError, HIP 719), a GPU travar e o
# driver amdgpu resetar a placa, derrubando a sessao grafica junto. Duas
# tentativas, dois travamentos. O ganho de velocidade seria da ordem de 10%,
# porque 92% do tempo de epoca e a fase de treino e ela escala com a quantidade
# de trabalho, nao com o numero de iteracoes. Nao vale o risco.
LOTE_SEGURO = {1280: 4, 960: 6, 640: 8}


def verificar_lote(args) -> None:
    limite = LOTE_SEGURO.get(args.imgsz)
    if limite is None or args.batch <= limite or args.force_batch:
        return
    raise SystemExit(
        f"batch {args.batch} a {args.imgsz} trava a GPU nesta placa "
        f"(gfx1200): o MIOpen falha ao lancar a convolucao e o driver reseta o\n"
        f"dispositivo, derrubando a sessao grafica. O limite medido e "
        f"batch {limite}.\nUse --force-batch se quiser tentar assim mesmo."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=int, required=True, help="1..4 do fatorial")
    p.add_argument("--fold", type=int, required=True, help="0..4")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--force-batch", action="store_true",
                   help="permite lote acima do limite seguro medido nesta placa")
    p.add_argument("--epochs", type=int, default=300)
    # A Secao 3.1.6 do TCC2 previa paciencia inicial de 50, a ser confirmada nos
    # testes preliminares. O piloto mostrou que 50 encerra cedo demais: a melhor
    # epoca caiu num pico de ruido da validacao interna, que tem so 26 imagens e
    # cujo mAP salta 21% do proprio nivel entre epocas vizinhas.
    # Prechelt (1998), sobre 1296 execucoes, mede que criterios mais lentos
    # elevam de cerca de 60% para cerca de 80% a chance de terminar com o melhor
    # resultado da execucao, custando cerca de 4x mais tempo. Aqui o tempo cabe,
    # e o custo de errar o checkpoint e maior do que no caso que ele estudou:
    # o trabalho compara quatro configuracoes, entao o erro de selecao vira
    # variancia dentro do proprio teste pareado.
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--model", default="yolov8s.pt")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache", default="disk", help="disk, ram ou vazio")
    p.add_argument("--no-amp", action="store_true", help="treina em fp32")
    p.add_argument("--resume", action="store_true",
                   help="retoma do ultimo checkpoint se a execucao foi interrompida")
    p.add_argument("--realce", default=REALCE_PADRAO, choices=("a", "b", "d"),
                   help="variante de realce das Configuracoes 3 e 4")
    p.add_argument("--force", action="store_true",
                   help="refaz mesmo que ja exista resumo.json")
    p.add_argument("--name", default=None)
    p.add_argument("--dry-run", action="store_true", help="so mostra o que faria")
    args = p.parse_args()

    verificar_lote(args)
    rotulo, origem, hiper = resolver(args.config, args.fold, args.realce)
    nome = args.name or f"cfg{args.config}_fold{args.fold}_{args.imgsz}"
    saida = RUNS / nome

    treino = FOLDS / f"fold{args.fold}_treino.txt"
    val_interna = FOLDS / f"fold{args.fold}_val_interna.txt"
    avaliacao = FOLDS / f"fold{args.fold}_avaliacao.txt"
    for lista in (treino, val_interna, avaliacao):
        if not lista.is_file():
            raise SystemExit(f"lista ausente: {lista}")

    # Uma execucao so e considerada concluida quando gravou o resumo.json, que e
    # a ultima coisa que este script escreve. Assim uma campanha interrompida no
    # meio pode ser relancada inteira: as rodadas prontas sao puladas e a que
    # morreu retoma do ultimo checkpoint.
    if (saida / "resumo.json").is_file() and not args.force:
        print(f"\n{nome}: ja concluida (resumo.json existe). Use --force para refazer.")
        return

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

    # A Ultralytics grava weights/last.pt ao fim de CADA epoca. Como uma epoca
    # custa cerca de 12 s, uma queda de energia ou um travamento da placa custa
    # no maximo uma epoca de trabalho, nao a execucao inteira. Com resume=True a
    # biblioteca le os argumentos de dentro do proprio checkpoint, entao nao se
    # passa mais nenhum: mudar um hiperparametro aqui seria silenciosamente
    # ignorado e a execucao retomada nao corresponderia ao que se pediu.
    ultimo = saida / "weights" / "last.pt"
    if args.resume and ultimo.is_file():
        print(f"\nretomando de {ultimo}")
        modelo = YOLO(ultimo)
        inicio = time.time()
        modelo.train(resume=True)
        _avaliar(args, nome, saida, rotulo, hiper, treino, avaliacao,
                 time.time() - inicio)
        return

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
    _avaliar(args, nome, saida, rotulo, hiper, treino, avaliacao, duracao)


def _avaliar(args, nome, saida, rotulo, hiper, treino, avaliacao, duracao) -> None:
    """Avalia o melhor checkpoint no fold retido e grava o resumo da rodada."""
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
        # A analise estatistica do trabalho tem a CARTA como unidade, nao o fold:
        # cada carta e avaliada uma unica vez, em exatamente uma configuracao por
        # rodada, o que da 82 pares para o teste de Wilcoxon em vez de 5. Isso
        # exige a predicao individual de cada imagem, e nao apenas a metrica
        # agregada do fold. Sem este save_json seria preciso repetir as 20
        # execucoes so para recuperar as predicoes.
        save_json=True,
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

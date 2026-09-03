"""
Executa as 20 rodadas do fatorial (4 configuracoes x 5 folds) e retoma de onde parou.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
RUNS = RAIZ / "Dataset_YOLO" / "runs"
TREINADOR = Path(__file__).resolve().parent / "train_fold.py"
PYTHON = RAIZ / ".venv" / "bin" / "python"


def nome_da_rodada(config: int, fold: int, imgsz: int) -> str:
    return f"cfg{config}_fold{fold}_{imgsz}"


def concluida(nome: str) -> bool:
    return (RUNS / nome / "resumo.json").is_file()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="1,2,3,4", help="lista separada por virgula")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--tentativas", type=int, default=3,
                   help="quantas vezes retomar uma rodada que caiu")
    p.add_argument("--dry-run", action="store_true",
                   help="lista o plano sem treinar nada")
    args = p.parse_args()

    configs = [int(c) for c in args.configs.split(",")]
    folds = [int(f) for f in args.folds.split(",")]
    rodadas = [(c, f) for c in configs for f in folds]

    feitas = [r for r in rodadas if concluida(nome_da_rodada(*r, args.imgsz))]
    print(f"{len(rodadas)} rodadas no plano, {len(feitas)} ja concluidas\n")

    if args.dry_run:
        for config, fold in rodadas:
            nome = nome_da_rodada(config, fold, args.imgsz)
            print(f"  {nome}: {'concluida' if concluida(nome) else 'pendente'}")
        print("\n--dry-run: nada foi treinado")
        return

    inicio_campanha = time.time()
    falhas = []

    for i, (config, fold) in enumerate(rodadas, 1):
        nome = nome_da_rodada(config, fold, args.imgsz)
        if concluida(nome):
            print(f"[{i}/{len(rodadas)}] {nome}: pulada, ja concluida")
            continue

        base = [
            str(PYTHON), str(TREINADOR),
            "--config", str(config), "--fold", str(fold),
            "--imgsz", str(args.imgsz), "--batch", str(args.batch),
            "--epochs", str(args.epochs), "--patience", str(args.patience),
        ]

        for tentativa in range(1, args.tentativas + 1):
            # A primeira tentativa comeca do zero; as seguintes retomam do
            # last.pt, que a Ultralytics grava ao fim de cada epoca. Uma queda
            # da placa custa no maximo a epoca em andamento.
            cmd = base + (["--resume"] if tentativa > 1 else [])
            marca = f"[{i}/{len(rodadas)}] {nome} tentativa {tentativa}"
            print(f"{marca}: iniciando", flush=True)
            t0 = time.time()
            codigo = subprocess.call(cmd)
            dt = (time.time() - t0) / 60

            if codigo == 0 and concluida(nome):
                print(f"{marca}: concluida em {dt:.1f} min\n", flush=True)
                break
            print(f"{marca}: falhou (codigo {codigo}) apos {dt:.1f} min", flush=True)
        else:
            falhas.append(nome)
            print(f"{nome}: desistindo apos {args.tentativas} tentativas\n", flush=True)

    total = (time.time() - inicio_campanha) / 3600
    print(f"\n{'=' * 60}")
    print(f"campanha encerrada em {total:.1f} h")

    prontas = [r for r in rodadas if concluida(nome_da_rodada(*r, args.imgsz))]
    print(f"{len(prontas)} de {len(rodadas)} rodadas com resumo.json")
    if falhas:
        print(f"falharam: {', '.join(falhas)}")
        sys.exit(1)

    tabela = []
    for config, fold in prontas:
        d = json.loads((RUNS / nome_da_rodada(config, fold, args.imgsz)
                        / "resumo.json").read_text(encoding="utf-8"))
        tabela.append(d)
    print("\n cfg fold  epocas  min   mAP50   mAP50-95")
    for d in sorted(tabela, key=lambda x: (x["config"], x["fold"])):
        print(f"  {d['config']}   {d['fold']}    {d['epocas_rodadas']:4d}"
              f"  {d['segundos_total'] / 60:5.1f}"
              f"  {d['map50']:.4f}  {d['map50_95']:.4f}")


if __name__ == "__main__":
    main()

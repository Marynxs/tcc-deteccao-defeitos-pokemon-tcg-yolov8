"""Curva de aprendizado: gera listas de treino aninhadas (25%, 50%, 75% das
CARTAS de treino de uma dobra) e lanca train_fold.py para cada fracao.

A validacao interna e a dobra retida NAO mudam; so o tamanho do treino.
O ponto de 100% e a propria rodada cfgN_foldK_1280 da campanha.

    python src/treino/learning_curve.py --fold 4 --config 2 --pool pool_206_rev
    python src/treino/learning_curve.py --fold 4 --config 2 --pool pool_206_rev --dry-run
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRACOES = (0.25, 0.50, 0.75)
SEMENTE = 206


def carta_de(caminho: str) -> str:
    return Path(caminho).name[:9]  # carta_NNN


def gerar_listas(pool: str, fold: int, destino: Path) -> dict:
    sufixo = pool[len("pool"):]
    folds = ROOT / "Dataset_YOLO" / f"folds{sufixo}"
    linhas = [l.strip() for l in (folds / f"fold{fold}_treino.txt").open() if l.strip()]
    # unidade do sorteio e o GRUPO de mesma arte (mesma regra de generate_folds.py);
    # carta fora de grupo e um grupo de uma so
    grupos = json.loads((ROOT / "docs" / "grupos_mesma_arte.json").read_text())
    grupo_de = {c: i for i, g in enumerate(grupos) for c in g}
    cartas = sorted({carta_de(l) for l in linhas})
    unidades = {}
    for c in cartas:
        unidades.setdefault(f"g{grupo_de[c]:03d}" if c in grupo_de else c, []).append(c)
    chaves = sorted(unidades)
    random.Random(SEMENTE).shuffle(chaves)
    # ordem fixa embaralhada: a fracao f pega o prefixo, logo 25% esta dentro de 50%, que esta dentro de 75%
    destino.mkdir(parents=True, exist_ok=True)
    saida = {}
    for f in FRACOES:
        alvo = round(f * len(cartas))
        escolhidas, n = set(), 0
        for k in chaves:
            if n >= alvo:
                break
            escolhidas.update(unidades[k]); n += len(unidades[k])
        lista = destino / f"fold{fold}_treino_{int(f * 100)}.txt"
        sel = [l for l in linhas if carta_de(l) in escolhidas]
        lista.write_text("\n".join(sel) + "\n")
        saida[f] = (lista, len(escolhidas), len(sel))
    return saida


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--config", type=int, default=2)
    ap.add_argument("--pool", default="pool_206_rev")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sufixo = args.pool[len("pool"):]
    destino = ROOT / "Dataset_YOLO" / f"folds{sufixo}" / "curva"
    listas = gerar_listas(args.pool, args.fold, destino)
    total = sum(1 for _ in (ROOT / "Dataset_YOLO" / f"folds{sufixo}" / f"fold{args.fold}_treino.txt").open())
    for f, (lista, ncartas, nimg) in listas.items():
        print(f"{int(f * 100):3d}%: {ncartas:3d} cartas, {nimg:3d} imagens (de {total})  -> {lista.relative_to(ROOT)}")
    print(f"100%: rodada cfg{args.config}_fold{args.fold}_{args.imgsz} da campanha (ja existe)")
    for f, (lista, _, _) in listas.items():
        nome = f"cfg{args.config}_fold{args.fold}_{args.imgsz}_curva{int(f * 100)}"
        cmd = [sys.executable, str(ROOT / "src" / "treino" / "train_fold.py"),
               "--config", str(args.config), "--fold", str(args.fold), "--pool", args.pool,
               "--imgsz", str(args.imgsz), "--cache", "ram", "--name", nome, "--train-list", str(lista)]
        runs = ROOT / "Dataset_YOLO" / f"runs{sufixo}" / nome
        if (runs / "weights" / "last.pt").is_file() and not (runs / "resumo.json").is_file():
            cmd.append("--resume")
        print("\n$", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

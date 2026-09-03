"""Divide o dataset em cinco folds estratificados, por carta.

A unidade e a carta, nunca a imagem: frente e verso caem sempre no mesmo fold.
A estratificacao e pelo numero de faces com defeito, para que as cinco cartas
sem defeito fiquem uma por fold. Cada carta e avaliada exatamente uma vez, o
que produz as 82 observacoes pareadas usadas na comparacao estatistica.

Os folds sao gravados em disco e ficam fixos entre as quatro configuracoes.

    python src/dados/generate_folds.py           # gera e confere
    python src/dados/generate_folds.py --verify  # so confere o que ja existe
"""
import json, re, sys, collections, random
from pathlib import Path

ROOT = Path("Dataset_YOLO")
SRC_IMAGES, SRC_LABELS = ROOT / "pool/images_final", ROOT / "pool/labels_final"
DEST = ROOT / "dataset"
FOLDS = ROOT / "folds"
K = 5
INTERNAL_VAL_FRACTION = 0.20
SEED = 42


# O Ultralytics acha o label trocando "/images/" por "/labels/" no caminho da
# imagem (img2label_paths). Pastas chamadas "images_final"/"labels_final"
# derrubam essa troca: nenhum label e encontrado, toda imagem e tratada como
# vazia, e o treino roda ate o fim SEM ERRO E SEM AVISO. Dai esta estrutura
# canonica, montada com hardlink para nao duplicar 1,1 GB.
def build_structure():
    """Cria dataset/images e dataset/labels por hardlink."""
    (DEST / "images").mkdir(parents=True, exist_ok=True)
    (DEST / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for img in sorted(SRC_IMAGES.glob("*.png")):
        alvo = DEST / "images" / img.name
        if not alvo.exists():
            alvo.hardlink_to(img); n += 1
        lbl = SRC_LABELS / f"{img.stem}.txt"
        alvo_l = DEST / "labels" / lbl.name
        if not alvo_l.exists():
            alvo_l.hardlink_to(lbl)
    return n


def collect_cards():
    """card -> {face: (nome_do_arquivo, n_anotacoes)}"""
    cards = collections.defaultdict(dict)
    for img in sorted((DEST / "images").glob("*.png")):
        m = re.match(r"(carta_\d+)_(frente|verso)", img.name)
        if not m:
            raise ValueError(f"name fora do padrao: {img.name}")
        lbl = DEST / "labels" / f"{img.stem}.txt"
        n = sum(1 for l in open(lbl) if l.strip()) if lbl.exists() else 0
        cards[m.group(1)][m.group(2)] = (img.name, n)
    return dict(cards)


def stratify(cards):
    """Estrato = numero de faces com pelo menos uma anotacao (0, 1 ou 2)."""
    strata_of = collections.defaultdict(list)
    for c, faces in cards.items():
        strata_of[sum(1 for _, n in faces.values() if n > 0)].append(c)
    return strata_of


def assign_folds(strata):
    """Reparte cada estrato ciclicamente entre os K folds."""
    rnd = random.Random(SEED)
    fold_of = {}
    for chave in sorted(strata):
        group = sorted(strata[chave])
        rnd.shuffle(group)
        # o deslocamento por estrato evita que os primeiros folds
        # recebam sempre a sobra de todos os strata
        offset = (chave * 2) % K
        for i, card in enumerate(group):
            fold_of[card] = (i + offset) % K
    return fold_of


def build_rounds(cards, fold_of):
    rnd = random.Random(SEED + 1)
    rounds = []
    for k in range(K):
        eval_cards = sorted(c for c, f in fold_of.items() if f == k)
        block = sorted(c for c, f in fold_of.items() if f != k)
        rnd.shuffle(block)
        n_val = round(len(block) * INTERNAL_VAL_FRACTION)
        rounds.append({
            "fold": k,
            "avaliacao": eval_cards,
            "val_interna": sorted(block[:n_val]),
            "treino": sorted(block[n_val:]),
        })
    return rounds


def images_of(cards, lista):
    return [str((DEST / "images" / name).resolve())
            for c in lista for name, _ in cards[c].values()]


def save(cards, fold_of, rounds):
    FOLDS.mkdir(parents=True, exist_ok=True)
    (FOLDS / "folds.json").write_text(json.dumps({
        "semente": SEED, "k": K,
        "fracao_val_interna": INTERNAL_VAL_FRACTION,
        "fold_por_carta": fold_of,
        "rounds": rounds,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    for r in rounds:
        k = r["fold"]
        for role in ("treino", "val_interna", "avaliacao"):
            arq = FOLDS / f"fold{k}_{role}.txt"
            arq.write_text("\n".join(images_of(cards, r[role])) + "\n", encoding="utf-8")
        treino = (FOLDS / f"fold{k}_treino.txt").resolve()
        val = (FOLDS / f"fold{k}_val_interna.txt").resolve()
        (FOLDS / f"fold{k}_data.yaml").write_text(
            "# Gerado por generate_folds.py. NAO EDITAR A MAO.\n"
            "# A avaliacao do fold retido NAO passa por este arquivo:\n"
            f"# chamar model.val() sobre fold{k}_avaliacao.txt\n"
            f"train: {treino}\n"
            f"val: {val}\n"
            "nc: 1\n"
            "names: ['defeito']\n", encoding="utf-8")


def verify(cards, fold_of, rounds):
    print("\n" + "=" * 66)
    print("CONFERENCIA")
    print("=" * 66)
    ok = True

    def chk(cond, txt):
        nonlocal ok
        print(f"  [{'OK ' if cond else 'ERRO'}] {txt}")
        ok = ok and cond

    chk(len(cards) == 82, f"82 cards ({len(cards)})")
    chk(all(len(f) == 2 for f in cards.values()), "toda card com frente e verso")
    n_img = sum(len(f) for f in cards.values())
    chk(n_img == 164, f"164 images ({n_img})")
    n_anot = sum(n for f in cards.values() for _, n in f.values())
    chk(n_anot == 1038, f"1038 anotacoes ({n_anot})")
    chk(len(fold_of) == len(cards), "toda card tem fold")

    times = collections.Counter(c for r in rounds for c in r["avaliacao"])
    chk(set(times.values()) == {1}, "cada card avaliada exatamente 1 vez")
    chk(len(times) == 82, f"82 cards avaliadas ao todo ({len(times)})")

    for r in rounds:
        s = set(r["treino"]) | set(r["val_interna"]) | set(r["avaliacao"])
        chk(len(s) == 82 and not (set(r["treino"]) & set(r["avaliacao"])),
            f"fold {r['fold']}: particao completa e sem sobreposicao")

    print(f"\n  {'fold':>5} {'avaliacao':>10} {'val.int':>9} {'treino':>8} {'imgs eval_cards':>10}")
    for r in rounds:
        print(f"  {r['fold']:>5} {len(r['avaliacao']):>10} {len(r['val_interna']):>9} "
              f"{len(r['treino']):>8} {len(r['avaliacao'])*2:>10}")

    print("\n  distribuicao dos strata por fold (faces com defeito):")
    strata_of = {c: sum(1 for _, n in f.values() if n > 0) for c, f in cards.items()}
    tab = collections.defaultdict(collections.Counter)
    for c, f in fold_of.items():
        tab[f][strata_of[c]] += 1
    print(f"  {'fold':>5} {'0 faces':>9} {'1 face':>8} {'2 faces':>9} {'total':>7}")
    for f in range(K):
        print(f"  {f:>5} {tab[f][0]:>9} {tab[f][1]:>8} {tab[f][2]:>9} {sum(tab[f].values()):>7}")
    total = collections.Counter(strata_of.values())
    print(f"  {'TODOS':>5} {total[0]:>9} {total[1]:>8} {total[2]:>9} {sum(total.values()):>7}")

    print("\n" + ("  TUDO CERTO" if ok else "  >>> HA ERROS ACIMA <<<"))
    return ok


if __name__ == "__main__":
    if "--verify" not in sys.argv:
        n = build_structure()
        print(f"estrutura dataset/images e dataset/labels: {n} hardlinks novos")
    cards = collect_cards()
    if "--verify" in sys.argv:
        d = json.loads((FOLDS / "folds.json").read_text(encoding="utf-8"))
        fold_of, rounds = d["fold_por_carta"], d["rounds"]
    else:
        fold_of = assign_folds(stratify(cards))
        rounds = build_rounds(cards, fold_of)
        save(cards, fold_of, rounds)
        print(f"gravado em {FOLDS}/")
    sys.exit(0 if verify(cards, fold_of, rounds) else 1)

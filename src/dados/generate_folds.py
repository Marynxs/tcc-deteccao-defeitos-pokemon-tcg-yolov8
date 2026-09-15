"""Divide o dataset em cinco folds estratificados, por carta.

A unidade e a carta, nunca a imagem: frente e verso caem sempre no mesmo fold.
Cartas de mesma ilustracao (docs/grupos_mesma_arte.json) tambem ficam juntas,
como uma unidade indivisivel, para o modelo nao chegar a avaliacao ja tendo
visto aquele fundo no treino. A estratificacao e pelo numero de faces com
defeito, para as cartas sem defeito se espalharem entre os folds.

Os folds sao gravados em disco e ficam fixos entre as quatro configuracoes.

    python src/dados/generate_folds.py                    # pool/ (padrao)
    python src/dados/generate_folds.py --pool pool_206    # outro pool
    python src/dados/generate_folds.py --verify           # so confere
    python src/dados/generate_folds.py --groups caminho.json
"""
import json, re, sys, collections, random
from pathlib import Path


def _arg(flag, default):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default


ROOT = Path("Dataset_YOLO")
POOL_NAME = _arg("--pool", "pool")
SUFFIX = POOL_NAME[len("pool"):]          # "" para pool/, "_206" para pool_206/
SRC_IMAGES = ROOT / POOL_NAME / "images_final"
SRC_LABELS = ROOT / POOL_NAME / "labels_final"
DEST = ROOT / f"dataset{SUFFIX}"
FOLDS = ROOT / f"folds{SUFFIX}"
GROUPS_FILE = Path(_arg("--groups", "docs/grupos_mesma_arte.json"))
K = 5
INTERNAL_VAL_FRACTION = 0.20
SEED = 42


# O Ultralytics acha o label trocando "/images/" por "/labels/" no caminho da
# imagem (img2label_paths). Pastas chamadas "images_final"/"labels_final"
# derrubam essa troca: nenhum label e encontrado, toda imagem e tratada como
# vazia, e o treino roda ate o fim SEM ERRO E SEM AVISO. Dai esta estrutura
# canonica, montada com hardlink para nao duplicar o dataset.
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


def load_groups(cards):
    """Grupos de mesma ilustracao restritos as cartas presentes no pool."""
    if not GROUPS_FILE.exists():
        print(f"aviso: {GROUPS_FILE} nao existe, cada carta e uma unidade")
        return []
    raw = json.loads(GROUPS_FILE.read_text(encoding="utf-8"))
    groups = [[c for c in g if c in cards] for g in raw]
    return [g for g in groups if len(g) > 1]


def build_units(cards, groups):
    """Unidade = grupo de mesma arte, ou a carta sozinha. unit_id -> [cartas]"""
    in_group = {c for g in groups for c in g}
    units = {f"g{i}": sorted(g) for i, g in enumerate(groups)}
    for c in cards:
        if c not in in_group:
            units[c] = [c]
    return units


def stratify(cards, units):
    """Estrato = menor numero de faces com defeito entre as cartas da unidade.

    O minimo faz uma unidade que contem carta sem defeito ser tratada como
    rara, e distribuida com o mesmo cuidado que uma carta sem defeito sozinha.
    """
    def faces_com_defeito(c):
        return sum(1 for _, n in cards[c].values() if n > 0)
    strata_of = collections.defaultdict(list)
    for u, membros in units.items():
        strata_of[min(faces_com_defeito(c) for c in membros)].append(u)
    return strata_of


def assign_folds(strata, units):
    """Reparte cada estrato ciclicamente entre os K folds, por unidade."""
    rnd = random.Random(SEED)
    fold_of = {}
    for chave in sorted(strata):
        group = sorted(strata[chave])
        rnd.shuffle(group)
        # o deslocamento por estrato evita que os primeiros folds
        # recebam sempre a sobra de todos os strata
        offset = (chave * 2) % K
        for i, u in enumerate(group):
            for c in units[u]:
                fold_of[c] = (i + offset) % K
    return fold_of


def build_rounds(cards, fold_of, units):
    rnd = random.Random(SEED + 1)
    unit_of = {c: u for u, ms in units.items() for c in ms}
    rounds = []
    for k in range(K):
        eval_cards = sorted(c for c, f in fold_of.items() if f == k)
        # a validacao interna tambem e recortada por unidade, para um grupo
        # de mesma arte nao ficar metade no treino e metade na validacao
        block_units = sorted({unit_of[c] for c, f in fold_of.items() if f != k})
        rnd.shuffle(block_units)
        n_cards = sum(len(units[u]) for u in block_units)
        n_val = round(n_cards * INTERNAL_VAL_FRACTION)
        val, acc = [], 0
        for u in block_units:
            if acc >= n_val:
                break
            val.extend(units[u]); acc += len(units[u])
        val = set(val)
        block = [c for u in block_units for c in units[u]]
        rounds.append({
            "fold": k,
            "avaliacao": eval_cards,
            "val_interna": sorted(val),
            "treino": sorted(c for c in block if c not in val),
        })
    return rounds


def images_of(cards, lista):
    return [str((DEST / "images" / name).resolve())
            for c in lista for name, _ in cards[c].values()]


def save(cards, fold_of, rounds, groups):
    FOLDS.mkdir(parents=True, exist_ok=True)
    (FOLDS / "folds.json").write_text(json.dumps({
        "semente": SEED, "k": K,
        "fracao_val_interna": INTERNAL_VAL_FRACTION,
        "grupos_mesma_arte": groups,
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


def verify(cards, fold_of, rounds, groups):
    print("\n" + "=" * 66)
    print("CONFERENCIA")
    print("=" * 66)
    ok = True

    def chk(cond, txt):
        nonlocal ok
        print(f"  [{'OK ' if cond else 'ERRO'}] {txt}")
        ok = ok and cond

    n_cards = len(cards)
    n_src = len(list(SRC_IMAGES.glob("*.png")))
    n_img = sum(len(f) for f in cards.values())
    n_anot = sum(n for f in cards.values() for _, n in f.values())
    n_anot_src = sum(sum(1 for l in open(p) if l.strip()) for p in SRC_LABELS.glob("*.txt"))
    chk(n_img == n_src, f"{n_src} imagens do pool presentes em dataset/ ({n_img})")
    chk(all(len(f) == 2 for f in cards.values()), "toda carta com frente e verso")
    chk(n_anot == n_anot_src, f"{n_anot_src} anotacoes do pool preservadas ({n_anot})")
    chk(len(fold_of) == n_cards, "toda carta tem fold")

    times = collections.Counter(c for r in rounds for c in r["avaliacao"])
    chk(set(times.values()) == {1}, "cada carta avaliada exatamente 1 vez")
    chk(len(times) == n_cards, f"{n_cards} cartas avaliadas ao todo ({len(times)})")

    for r in rounds:
        s = set(r["treino"]) | set(r["val_interna"]) | set(r["avaliacao"])
        chk(len(s) == n_cards and not (set(r["treino"]) & set(r["avaliacao"])),
            f"fold {r['fold']}: particao completa e sem sobreposicao")

    for g in groups:
        folds_do_grupo = {fold_of[c] for c in g}
        chk(len(folds_do_grupo) == 1, f"grupo {'+'.join(g)} inteiro no fold {folds_do_grupo}")
        for r in rounds:
            papeis = {p for p in ("treino", "val_interna", "avaliacao")
                      for c in g if c in r[p]}
            chk(len(papeis) == 1, f"  fold {r['fold']}: grupo nao dividido entre papeis")

    print(f"\n  {'fold':>5} {'avaliacao':>10} {'val.int':>9} {'treino':>8} {'imgs eval':>10}")
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
    print(f"pool: {POOL_NAME}  ->  {DEST}/  e  {FOLDS}/")
    if "--verify" not in sys.argv:
        n = build_structure()
        print(f"estrutura {DEST}/images e {DEST}/labels: {n} hardlinks novos")
    cards = collect_cards()
    groups = load_groups(cards)
    print(f"grupos de mesma arte no pool: {len(groups)} ({sum(len(g) for g in groups)} cartas)")
    if "--verify" in sys.argv:
        d = json.loads((FOLDS / "folds.json").read_text(encoding="utf-8"))
        fold_of, rounds = d["fold_por_carta"], d["rounds"]
        groups = d.get("grupos_mesma_arte", groups)
    else:
        units = build_units(cards, groups)
        fold_of = assign_folds(stratify(cards, units), units)
        rounds = build_rounds(cards, fold_of, units)
        save(cards, fold_of, rounds, groups)
        print(f"gravado em {FOLDS}/")
    sys.exit(0 if verify(cards, fold_of, rounds, groups) else 1)

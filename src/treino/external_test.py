"""Avalia os modelos da campanha das 82 cartas nas 124 cartas novas (teste externo).

Os 20 best.pt de runs/ nunca viram as cartas 083 a 206: nem as fotos, nem o
fundo branco, nem as resolucoes novas. Avalia-los nessas 248 imagens mede a
generalizacao entre LOTES, que a validacao cruzada (dentro do mesmo lote) nao
mede. Custa so inferencia.

O limiar do F1 vem da validacao interna de cada rodada (pool antigo), como no
train_fold, e e aplicado fixo ao conjunto externo. As configuracoes 3 e 4 leem
as imagens de dataset_206_realce_<v>.

    python src/treino/external_test.py            # 20 rodadas, imgsz 1280
    python src/treino/external_test.py --configs 2 --folds 0
"""
import argparse, json, re, sys, time, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_fold as tf                      # noqa: E402
from ultralytics import YOLO                 # noqa: E402

ROOT = tf.ROOT
RUNS_OLD = ROOT / "Dataset_YOLO" / "runs"
DEST = ROOT / "Dataset_YOLO" / "runs_externo"
tf.RUNS = DEST      # a val da validacao interna feita pelo F1 grava aqui, nao em runs/
NOVO = ROOT / "Dataset_YOLO" / "dataset_206"
PRIMEIRA_NOVA = 83


def lista_externa(origem: str) -> Path:
    """Lista das 248 imagens novas, em RGB ou na pasta de realce pedida."""
    pasta = ROOT / "Dataset_YOLO" / origem / "images"
    imgs = []
    for p in sorted(pasta.glob("*.png")):
        m = re.match(r"carta_(\d+)_", p.name)
        if m and int(m.group(1)) >= PRIMEIRA_NOVA:
            imgs.append(str(p.resolve()))
    if len(imgs) != 248:
        raise SystemExit(f"esperava 248 imagens novas em {pasta}, achei {len(imgs)}")
    DEST.mkdir(parents=True, exist_ok=True)
    lista = DEST / f"externo_{origem}.txt"
    lista.write_text("\n".join(imgs) + "\n", encoding="utf-8")
    yaml = DEST / f"externo_{origem}.yaml"
    yaml.write_text(f"train: {lista}\nval: {lista}\nnc: 1\nnames: ['defeito']\n",
                    encoding="utf-8")
    return yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="1,2,3,4")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--realce", default=tf.REALCE_PADRAO)
    args = p.parse_args()

    resultados = []
    for config in (int(c) for c in args.configs.split(",")):
        origem = "dataset_206" if config in (1, 2) else f"dataset_206_realce_{args.realce}"
        yaml = lista_externa(origem)
        for fold in (int(f) for f in args.folds.split(",")):
            nome = f"cfg{config}_fold{fold}_{args.imgsz}"
            saida = RUNS_OLD / nome
            pesos = saida / "weights" / "best.pt"
            if not pesos.is_file():
                print(f"{nome}: sem best.pt, pulando"); continue
            antigo = json.loads((saida / "resumo.json").read_text(encoding="utf-8"))

            t0 = time.time()
            modelo = YOLO(pesos)
            ext = modelo.val(data=str(yaml), imgsz=args.imgsz, batch=args.batch,
                             channels_last=False, project=str(DEST), name=nome,
                             exist_ok=True, plots=False, verbose=False, save_json=True)
            f1 = tf._f1_com_limiar_fixo(modelo, types.SimpleNamespace(imgsz=args.imgsz, batch=args.batch),
                                        nome, saida, ext)
            r = {
                "config": config, "fold": fold, "rotulo": antigo["rotulo"],
                "retido_map50": antigo["map50"], "retido_map50_95": antigo["map50_95"],
                "retido_f1": antigo.get("f1"),
                "externo_map50": round(float(ext.box.map50), 4),
                "externo_map50_95": round(float(ext.box.map), 4),
                "externo_precisao_otimista": round(float(ext.box.mp), 4),
                "externo_recall_otimista": round(float(ext.box.mr), 4),
                **{f"externo_{k}": v for k, v in f1.items()},
                "segundos": round(time.time() - t0, 1),
            }
            resultados.append(r)
            print(f"{nome}: retido mAP50 {r['retido_map50']:.4f} -> externo {r['externo_map50']:.4f}   "
                  f"F1 externo {r['externo_f1']:.4f} (limiar {r['externo_conf_do_f1']:.3f})", flush=True)

    (DEST / "resultados.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print(f"\n{len(resultados)} rodadas em {DEST / 'resultados.json'}")


if __name__ == "__main__":
    main()

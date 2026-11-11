"""Demonstracao do detector da Configuracao 2: recebe a foto de uma carta, recorta a carta e devolve as caixas de defeito."""
import base64
import io
import json
import re
import sys
import time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/dados"))
BRUTAS = ROOT / "Dataset_YOLO/pool_206/images"          # fotos como vieram, antes do recorte
ROTULOS_BRUTOS = ROOT / "Dataset_YOLO/pool_206/labels"  # rotulos que o pipeline usou para decidir o recorte
IMAGENS = ROOT / "Dataset_YOLO/dataset_206_rev/images"
ROTULOS = ROOT / "Dataset_YOLO/dataset_206_rev/labels"
FOLDS = ROOT / "Dataset_YOLO/folds_206_rev"
RUNS = ROOT / "Dataset_YOLO/runs_206_rev"
PAGINA = Path(__file__).with_name("index.html")
PORTA = 8765
LADO = 1280  # resolucao de entrada usada no treino
LADO_ORIGINAL = 1000  # a foto original so aparece na etapa 1, em tamanho de tela

# cartas sugeridas na tela inicial: frente e verso, holografica e comum, de dobras diferentes
EXEMPLOS = ["carta_155_frente", "carta_206_frente", "carta_039_frente", "carta_116_frente", "carta_090_frente",
            "carta_184_verso"]


def stem_da_carta(nome):
    m = re.search(r"carta_(\d{3})_(frente|verso)", nome)
    return f"carta_{m.group(1)}_{m.group(2)}" if m else None


@lru_cache(maxsize=1)
def dobra_de_cada_carta():
    # cada carta foi avaliada por exatamente um modelo, o da dobra em que ela ficou fora do treino
    mapa = {}
    for k in range(5):
        for linha in (FOLDS / f"fold{k}_avaliacao.txt").read_text().split():
            s = stem_da_carta(Path(linha).name)
            if s:
                mapa[s] = k
    return mapa


@lru_cache(maxsize=5)
def modelo(k):
    from ultralytics import YOLO
    pasta = RUNS / f"cfg2_fold{k}_1280"
    limiar = json.loads((pasta / "resumo.json").read_text())["conf_do_f1"]
    return YOLO(str(pasta / "weights/best.pt")), limiar


def arquivo_do_conjunto(stem, pasta, ext):
    achados = sorted(pasta.glob(f"{stem}_*{ext}")) or sorted(pasta.glob(f"{stem}*{ext}"))
    return achados[0] if achados else None


def anotacoes(stem, w, h):
    rot = arquivo_do_conjunto(stem, ROTULOS, ".txt")
    if rot is None:
        return None
    caixas = []
    for linha in rot.read_text().splitlines():
        p = linha.split()
        if len(p) >= 5:
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            caixas.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return caixas


def recortar(img, linhas=None):
    # mesma Etapa 1 e Etapa 2 do pipeline (card_detector + crop_and_adjust), com as mesmas regras de seguranca:
    # sem deteccao a foto passa inteira; corte que descartaria anotacao e rejeitado; corte que clipa anotacao cresce
    import card_detector
    import crop_and_adjust as corte
    card_detector.MARGIN_PX = corte.MARGIN_PX  # a Etapa 2 e a dona da margem (25 px)
    w, h = img.size
    caixa, info = card_detector.detect_card_box(img)
    motivo = None
    if caixa is None or card_detector.is_box_suspicious(info):
        motivo = "a carta já ocupa a foto inteira" if caixa is not None else "a carta não foi localizada; a foto segue inteira"
        caixa = (0, 0, w, h)
    elif linhas:
        x0, y0, x1, y1 = caixa
        _, n_clip, n_desc = corte.recalculate_all(linhas, w, h, x0, y0, x1 - x0, y1 - y0)
        if n_desc:
            motivo, caixa = "o corte descartaria anotação; a foto segue inteira", (0, 0, w, h)
        elif n_clip:
            caixa = corte.union_with_annotations(caixa, linhas, w, h)
    return tuple(int(v) for v in caixa), info.get("background"), motivo


def jpeg(img, lado, qualidade=90):
    esc = lado / max(img.size)
    w, h = round(img.width * esc), round(img.height * esc)
    buf = io.BytesIO()
    img.resize((w, h), Image.LANCZOS).save(buf, "JPEG", quality=qualidade)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), w, h, esc


def detectar(original, stem):
    # segue a ordem do processo do trabalho: 1) localiza a carta na foto, 2) recorta com 25 px de folga,
    # 3) o YOLOv8s procura defeitos na carta recortada, redimensionada para 1280 px no lado maior
    t0 = time.time()
    dobra = dobra_de_cada_carta().get(stem) if stem else None
    do_conjunto = dobra is not None
    original = original.convert("RGB")
    linhas = None
    if do_conjunto:
        rot = arquivo_do_conjunto(stem, ROTULOS_BRUTOS, ".txt")
        linhas = [l for l in rot.read_text().splitlines() if l.strip()] if rot else []
    caixa, fundo, motivo = recortar(original, linhas)
    img = original.crop(caixa)
    anot_validas = do_conjunto
    if do_conjunto:
        final = arquivo_do_conjunto(stem, IMAGENS, ".png") or arquivo_do_conjunto(stem, IMAGENS, ".jpg")
        # o recorte refeito aqui tem de bater com o que foi gravado no conjunto, senao a anotacao nao vale
        anot_validas = final is not None and Image.open(final).size == img.size
        if not anot_validas:
            print("aviso: recorte diferente do conjunto em", stem, img.size)
    yolo, limiar = modelo(dobra if do_conjunto else 4)
    r = yolo.predict(img, imgsz=LADO, conf=limiar, verbose=False)[0]
    tela, w, h, esc = jpeg(img, LADO)
    caixas = [{"x0": float(b[0]) * esc, "y0": float(b[1]) * esc, "x1": float(b[2]) * esc, "y1": float(b[3]) * esc,
               "conf": float(c)} for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
    caixas.sort(key=lambda c: -c["conf"])
    orig, ow, oh, oesc = jpeg(original, LADO_ORIGINAL, 85)
    return {
        "imagem": tela, "largura": w, "altura": h, "caixas": caixas,
        "anotacoes": anotacoes(stem, w, h) if anot_validas else None,
        "original": orig, "orig_largura": ow, "orig_altura": oh,
        "orig_px": list(original.size), "recorte_px": [img.width, img.height],
        "corte": [v * oesc for v in caixa], "fundo": fundo, "motivo": motivo,
        "carta": stem, "dobra": (dobra + 1) if do_conjunto else None, "dobra_modelo": (dobra if do_conjunto else 4) + 1,
        "do_conjunto": do_conjunto, "limiar": limiar,
        "ms": round((time.time() - t0) * 1000),
    }


def foto_bruta(stem):
    return arquivo_do_conjunto(stem, BRUTAS, ".png") or arquivo_do_conjunto(stem, BRUTAS, ".jpg")


class Tratador(BaseHTTPRequestHandler):
    def _json(self, dados, status=200):
        corpo = json.dumps(dados).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def do_GET(self):
        rota = self.path.split("?")[0]
        if rota in ("/", "/index.html"):
            corpo = PAGINA.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            self.wfile.write(corpo)
        elif rota == "/api/cartas":
            # todas as faces do conjunto, em ordem, para PgUp/PgDn e "ir para carta"
            self._json(sorted(dobra_de_cada_carta()))
        elif rota == "/api/exemplos":
            itens = []
            for s in EXEMPLOS:
                arq = arquivo_do_conjunto(s, IMAGENS, ".png") or arquivo_do_conjunto(s, IMAGENS, ".jpg")
                if arq is None:
                    continue
                im = Image.open(arq).convert("RGB")
                im.thumbnail((180, 250))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=80)
                itens.append({"carta": s, "miniatura": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()})
            self._json(itens)
        else:
            self.send_error(404)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            corpo = self.rfile.read(n)
            if self.path == "/api/exemplo":
                stem = json.loads(corpo)["carta"]
                self._json(detectar(Image.open(foto_bruta(stem)), stem))
            elif self.path == "/api/detectar":
                nome = self.headers.get("X-Nome", "")
                stem = stem_da_carta(nome)
                img = Image.open(io.BytesIO(corpo))
                if stem and stem in dobra_de_cada_carta():
                    # foto de uma carta do conjunto: parte da foto como veio, antes do recorte, para o
                    # recorte refeito aqui ser o mesmo do pipeline e a anotacao valer
                    img = Image.open(foto_bruta(stem))
                elif stem:
                    stem = None
                self._json(detectar(img, stem))
            else:
                self.send_error(404)
        except Exception as erro:
            self._json({"erro": f"Não consegui processar a imagem: {erro}"}, 400)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    print(f"carregando o modelo da dobra 5 para aquecer...")
    modelo(4)
    print(f"demonstracao em http://127.0.0.1:{PORTA}  (Ctrl+C para sair)")
    ThreadingHTTPServer(("127.0.0.1", PORTA), Tratador).serve_forever()

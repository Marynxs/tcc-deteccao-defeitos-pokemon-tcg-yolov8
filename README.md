# Detecção de Defeitos em Cartas Pokémon TCG com YOLOv8

Trabalho de Conclusão de Curso em Ciência da Computação, Centro Universitário
SENAC Santo Amaro.

**Autor:** Matheus da Silva Marini
**Orientador:** Prof. Afonso César Lelis Brandão
**Entrega:** 27/11/2026

## O trabalho

Avaliação de estratégias de treinamento aplicadas ao YOLOv8 para detecção de
defeitos superficiais em cartas Pokémon TCG, em cenário com baixo volume de
dados. O objetivo não é construir um sistema de graduação automática, mas medir,
de forma controlada, como diferentes estratégias de treinamento afetam o
desempenho do detector quando há poucos exemplos anotados.

O desenho central é um estudo fatorial 2x2, que cruza dois fatores:

| | Representação RGB | Realce de micro-relevo |
|---|---|---|
| **Sem aumento de dados** | Configuração 1 | Configuração 3 |
| **Com aumento de dados** | Configuração 2 | Configuração 4 |

Sobre a configuração de melhor desempenho (Configuração 2), uma ablação avalia um componente por vez:
a cabeça P2 e a resolução de entrada (640 contra 1280
pixels). Completam o estudo, sem hipótese formal, uma curva de aprendizado, um teste
de capacidade (yolov8m), uma representação de controle (só a luminosidade, sem
realce), a inferência por fatias e um treino sem as cartas de desgaste difuso. A avaliação usa validação cruzada estratificada em cinco dobras, fixas
entre as configurações, e comparação por teste de Wilcoxon pareado com a carta
como unidade de análise (206 pares), com correção de Holm.

## Conjunto de dados

Fotografias de 206 cartas, cedidas por empresa brasileira especializada em graduação,
totalizando 412 imagens (frente e verso) e 4.808 instâncias anotadas em classe
única, `defeito`, pelo autor, com apoio nos laudos de graduação. Parte das caixas
(892) foi sugerida por modelos treinados nos testes preliminares e validada uma
a uma pelo autor; os resultados também são conferidos sem essas caixas. Uma campanha
anterior, com 82 cartas, ficou como testes preliminares (`resultados/campanha_82/`).

O conjunto **não é versionado** neste repositório, por seu volume (6,5 GB) e por
ser material cedido por terceiro.

## Estrutura

```
tcc1/     Monografia do TCC1, entregue e aprovada. Congelada.
tcc2/       Monografia do TCC2 (versão completa), mais os slides da defesa em PDF.
src/dados/  Preparação dos dados e particionamento.
src/treino/ Treinamento, campanha, comparação pareada e escolha do realce.
src/analise/ Análises complementares que geram números citados no TCC.
src/demo/   Demonstração do detector no navegador (ver a seção Demonstração).
resultados/ Resumo de cada rodada, predições compactadas e saídas das análises.
```

A apresentação da defesa foi feita no Canva e está em `tcc2/apresentacao_tcc2.pdf`:
43 slides apresentados e, depois do agradecimento, 16 de apoio para as perguntas.
O arquivo `tcc2/apresentacao_tcc2.tex` é o modelo da disciplina e não foi usado.

## Pipeline de dados

Os scripts em `src/dados/` executam nesta ordem, sempre a partir da raiz do
repositório:

| Ordem | Script | O que faz |
|---|---|---|
| 1 | `organize_cards.py` | Padroniza a nomenclatura das imagens de origem e gera a planilha de mapeamento. |
| 2 | `card_detector.py` | Detecta a carta na foto. Modelo de fundo por z-score em Lab normalizado pelo desvio robusto, com evidência de textura e validação pela proporção física 63x88 mm. |
| 3 | `crop_and_adjust.py` | Corta pela caixa detectada e recalcula todas as anotações YOLO para o novo enquadramento. Importa a detecção da etapa anterior, de modo que as duas etapas enxergam a mesma caixa. |
| 4 | `run_pipeline.py --pool pool_206_rev` | Executa as etapas 2 e 3 sobre as 412 imagens. |
| 5 | `generate_folds.py --pool pool_206_rev` | Gera as cinco dobras por carta, estratificadas, com os grupos de mesma ilustração (`src/dados/grupos_mesma_arte.json`) sempre juntos. |

```bash
python src/dados/run_pipeline.py --pool pool_206_rev     # etapas 2 e 3 sobre as 412 imagens
python src/dados/generate_folds.py --pool pool_206_rev   # particionamento, com conferência automática
```

Resultado da preparação (conjunto de 206 cartas): 412 de 412 imagens
processadas, nenhuma anotação descartada nem cortada. A ocupação média da carta
no quadro passa de 86,5% para 93,8%. Dobras com 39, 39, 38, 45 e 45 cartas.

## Treinamento

`src/treino/train_fold.py` treina uma das quatro configurações do fatorial em
uma dobra e avalia na dobra retida:

```bash
python src/treino/train_fold.py --config 1 --fold 0 --imgsz 1280 --batch 4
python src/treino/train_fold.py --config 1 --fold 0 --dry-run   # só mostra o que faria
```

O `--dry-run` imprime os hiperparâmetros de aumento de dados já resolvidos, para
conferência direta contra a Tabela 6 da monografia. A avaliação da dobra retida
não passa pelo arquivo de configuração do treino: aquele aponta para a validação
interna, que alimentou a parada antecipada e portanto já influenciou os pesos.
Cada execução grava um `resumo.json` com tempo, pico de memória de vídeo e as
métricas finais.

A campanha completa, quatro configurações em cinco dobras, sai de
`run_campaign.py`. Ele pula as rodadas que já gravaram `resumo.json` e retoma do
último checkpoint aquelas que caíram, de modo que relançar o mesmo comando após
uma interrupção continua de onde parou:

```bash
python src/treino/run_campaign.py --dry-run   # mostra o plano sem treinar
python src/treino/run_campaign.py             # roda o que falta
```

`monitor_web.py` sobe um servidor local que acompanha o treino em andamento no
navegador, com progresso, tempo decorrido, estimativa de término e a curva do
mAP por época. Usa apenas a biblioteca padrão:

```bash
python src/treino/monitor_web.py    # http://127.0.0.1:8765
```

### Particionamento

A unidade do particionamento é a carta, não a imagem: frente e verso da mesma
carta pertencem sempre à mesma dobra, por serem a mesma peça física. A
estratificação é feita pelo número de faces com defeito, e cartas de mesma
ilustração caem sempre na mesma dobra. Cada carta é avaliada exatamente uma
vez, na dobra a que pertence. Cerca de 20% das cartas de treino de cada rodada
formam a validação interna, que escolhe a época, o limiar de confiança do F1 e,
na primeira dobra, a variante de realce.

O `generate_folds.py` confere o resultado por conta própria e falha se qualquer
invariante for violada.

## Resultados principais

Os detalhes estão na Seção 3.4 e no Capítulo 4 da monografia.

- O aumento de dados melhorou o detector nas duas representações: com imagens
  RGB, o F1 por carta médio passou de 0,332 para 0,376 (Configurações 1 e 2).
  Abaixo de 8 pixels, a contagem de defeitos encontrados foi menor com aumento
  de dados nas duas representações, com diferença significativa só nas imagens
  RGB, em análise exploratória, feita depois dos resultados.
- O realce de micro-relevo não superou as imagens RGB. Sem aumento de
  dados, uma representação de controle, com a luminosidade sem realce, separou os
  dois efeitos: trocar os canais pelos filtros piorou o F1 por carta nas duas
  referências de anotação, e a perda pela retirada da cor só apareceu com a
  anotação completa.
- A cabeça P2 não trouxe ganho: o F1 por carta foi menor, e a diferença na
  revocação dos defeitos menores que 8 pixels não se manteve com limiar de
  confiança comum.
- Nas cinco dobras, a entrada de 1280 pixels superou a de 640: F1 por carta médio
  de 0,376 contra 0,270, maior em 161 das 206 cartas, em análise sem hipótese
  formal. A comparação YOLOv8m × YOLOv8s, em uma dobra, foi inconclusiva.
- A inferência por fatias encontrou mais defeitos abaixo de 8 pixels (75 contra 15
  de 517), mas o F1 por carta médio caiu de 0,379 para 0,355.
- Na curva de aprendizado, nas cinco dobras, o mAP@0,5 das 206 cartas cresceu a
  cada acréscimo de cartas (0,167, 0,198, 0,214 e 0,230), com ganhos menores
  depois do primeiro.
- A melhor configuração encontrou cerca de 29% dos defeitos anotados
  (revocação 0,285, precisão 0,413). O detector ainda não é suficiente para a
  graduação automática.

## Análises e reprodução dos números do TCC

Os números do texto saem dos scripts abaixo, a partir das predições de cada
rodada. Todos rodam de qualquer pasta e gravam em `resultados/` (as saídas
de `src/analise/` ficam em `resultados/analises/`). Nos nomes de arquivo as
dobras vão de 0 a 4; no texto, de 1 a 5 (dobra 0 = Dobra 1, dobra 4 = Dobra 5).

| Script | O que gera |
|---|---|
| `src/treino/compare_paired.py` | Recall por faixa de tamanho e Wilcoxon por carta (F1 e recall), com p bruto e ajustado por Holm: `resultados/comparacao_pareada_206_rev.csv` |
| `src/treino/pick_variant.py` | Escolha do operador de realce pela validação interna da dobra 0: `resultados/selecao_realce_206.csv` |
| `src/analise/reevaluate_human_only.py` | Métricas sem as caixas sugeridas pelo modelo (só anotação humana), com Holm |
| `src/analise/interaction_test.py` | Teste de interação aumento × realce por carta |
| `src/analise/resolution_wilcoxon.py` | 1280 × 640 e yolov8s × yolov8m, por carta, na dobra 4 (Dobra 5 no texto) |
| `src/analise/reevaluate_resolution_control.py` | 1280 × 640 nas cinco dobras e representação de controle, por carta, nas duas referências de anotação: `saida_resolucao_controle_0.5.txt` |
| `src/analise/reevaluate_without_diffuse.py` | Treino sem as nove cartas de desgaste difuso, por carta, nas 197 cartas restantes: `saida_sem_difuso_0.5.txt` |
| `src/analise/sahi_eval.py` | Inferência por fatias contra a imagem inteira: `saida_sahi.txt` e predições em `fatias/` (da imagem inteira e da combinação escolhida; das outras duas fica o F1 na validação interna, em `saida_sahi.txt`) |
| `src/analise/sahi_small_only.py` | Variação das fatias que só acrescenta as caixas pequenas (exploratória, não citada no texto) |
| `src/analise/learning_curve_5folds.py` | Curva de aprendizado nas cinco dobras: mAP@0,5 das 206 cartas com intervalos (bootstrap por carta, 2.000 repetições) e comparações por carta: `saida_curva_5dobras.txt` |
| `src/analise/learning_curve_bootstrap*.py` | Versão inicial da curva, em uma dobra por vez (dobras 2 e 4) |
| `src/analise/h4_small_recall.py` | Revocação dos defeitos menores que 8 px, com e sem a cabeça P2 |
| `src/analise/h4_small_recall_human_only.py` | Revocação por faixa de tamanho das quatro configurações e da P2, nas duas referências de anotação, e o controle com limiar comum (P2 e aumento de dados) e o Wilcoxon por carta abaixo de 8 px do aumento de dados: `saida_h4_dois_gabaritos.txt`, `saida_h4_limiar_comum.txt`, `saida_h4_aumento_por_carta.txt` |
| `src/analise/recall_by_face.py` | Revocação da Configuração 2 por face da carta (frente e verso): `saida_recall_por_face.txt` |
| `src/analise/error_summary.py` | Cartas com ao menos um defeito encontrado e métricas sem as cartas de desgaste difuso: `saida_resumo_erros.txt` |
| `src/analise/box_origin.py` | Origem de cada caixa da anotação (original, sugerida ou manual), usada para a referência sem as caixas sugeridas: `origem_caixas.json`, `saida_box_origin.txt` |
| `src/analise/check_folds.py` | Conferência das dobras e da curva (aninhamento, grupos de mesma arte) |

Os lançadores das rodadas feitas depois do estudo fatorial ficam em `src/treino/`:

| Script | O que treina |
|---|---|
| `run_640_dobras.sh` | Configuração 2 a 640 pixels nas dobras 0 a 3 |
| `run_sem_difuso.sh` | Configuração 2 sem as nove cartas de desgaste difuso no treino, nas cinco dobras |
| `run_curva_dobras.sh` | Pontos de 25%, 50% e 75% da curva de aprendizado nas dobras 0, 1 e 3 |

As predições de cada rodada ficam em `resultados/campanha_206/predicoes/*.json.xz`
(descompactar em `Dataset_YOLO/runs_206_rev/<rodada>_avaliacao/predictions.json`).
Os scripts também leem os rótulos e o tamanho das imagens, que não são
versionados (material cedido por terceiro); com o conjunto em `Dataset_YOLO/`,
todos os números se refazem sem treinar de novo.

A exceção é `sahi_eval.py`, que também carrega os pesos `best.pt` de cada dobra, não
versionados pelo tamanho.

## Demonstração

`src/demo/app.py` abre uma página local que recebe a foto de uma carta, recorta a
carta e mostra as caixas de defeito da Configuração 2. Precisa dos pesos treinados
e do conjunto de dados, que não são versionados pelo tamanho. A página fica em
`http://127.0.0.1:8765`.

```bash
python src/demo/app.py
```

## Ambiente

```bash
python3 -m venv .venv
./.venv/bin/pip install torch==2.14.0+rocm7.2 torchvision==0.29.0+rocm7.2 --index-url https://download.pytorch.org/whl/rocm7.2
./.venv/bin/pip install -r requirements.txt
```

Para reproduzir exatamente o ambiente usado nos experimentos (Python 3.12),
instale a lista completa de versões fixadas, em vez do `requirements.txt`:

```bash
./.venv/bin/pip install -r requirements-lock-py312.txt --extra-index-url https://download.pytorch.org/whl/rocm7.2
```

Treinamento em AMD Radeon RX 9060 XT (gfx1200) com ROCm sob Linux. O ROCm não
treina no Windows.

**A versão do PyTorch importa.** Com `torch 2.10.0+rocm7.0` todo treino nesta
placa morre com `hipErrorIllegalAddress` antes de completar uma época, e o ponto
da falha muda a cada execução, ora no kernel de supressão não máxima, ora no
cálculo da perda. O sintoma é de corrupção de memória, não de um operador
defeituoso: não depende da resolução, da precisão nem do número de processos de
carga. A partir de `2.14.0+rocm7.2` o problema desaparece. Daí o índice
`rocm7.2` acima.

**Não instalar `albumentations`.** O pipeline de detecção do Ultralytics
instancia um bloco `Albumentations` que, se o pacote estiver presente, aplica
`Blur`, `MedianBlur`, `ToGray` e `CLAHE` a 1% das imagens de todas as
configurações. O último contaminaria a linha de base RGB com o tratamento que a
Configuração 3 investiga.

## Compilar as monografias

```bash
cd tcc2
pdflatex tcc2.tex && bibtex tcc2 && pdflatex tcc2.tex && pdflatex tcc2.tex
```

Requer TeX Live com `abntex2`:

```bash
sudo apt install texlive-latex-recommended texlive-latex-extra \
  texlive-fonts-recommended texlive-lang-portuguese texlive-publishers \
  texlive-pictures texlive-bibtex-extra
```

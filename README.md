# Detecção de Defeitos em Cartas Pokémon TCG com YOLOv8

Trabalho de Conclusão de Curso em Ciência da Computação, Centro Universitário
SENAC Santo Amaro.

**Autor:** Matheus da Silva Marini
**Orientador:** Prof. Afonso César Lelis Brandão
**Entrega:** segundo semestre de 2026

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

Sobre a configuração vencedora (2), uma ablação avalia um componente por vez:
a cabeça de detecção adicional P2 e a resolução de entrada (640 contra 1280
pixels). Completam o estudo uma curva de aprendizado e um teste de capacidade
(yolov8m). A avaliação usa validação cruzada estratificada em cinco dobras, fixas
entre as configurações, e comparação por teste de Wilcoxon pareado com a carta
como unidade de análise (206 pares), com correção de Holm.

## Conjunto de dados

206 cartas fornecidas por empresa brasileira especializada em graduação,
totalizando 412 imagens (frente e verso) e 4.808 instâncias anotadas em classe
única, `defeito`, pelo autor, com apoio nos laudos de graduação. Parte das caixas
(892) foi sugerida por um modelo treinado previamente e validada uma a uma pelo
autor; os resultados também são conferidos sem essas caixas. Uma campanha
anterior, com 82 cartas, ficou como testes preliminares (`resultados/campanha_82/`).

O conjunto **não é versionado** neste repositório, por seu volume (6,5 GB) e por
ser material cedido por terceiro.

## Estrutura

```
tcc1/     Monografia do TCC1, entregue e aprovada. Congelada.
tcc2/       Monografia do TCC2, em elaboração, mais os slides da defesa.
src/dados/  Preparação dos dados e particionamento.
src/treino/ Treinamento, campanha, comparação pareada e escolha do realce.
src/analise/ Análises complementares que geram números citados no TCC.
resultados/ Resumo de cada rodada, predições compactadas e saídas das análises.
```

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

## Treinamento

`src/treino/train_fold.py` treina uma das quatro configurações do fatorial em
uma dobra e avalia na dobra retida:

```bash
python src/treino/train_fold.py --config 1 --fold 0 --imgsz 1280 --batch 4
python src/treino/train_fold.py --config 1 --fold 0 --dry-run   # só mostra o que faria
```

O `--dry-run` imprime os hiperparâmetros de aumento de dados já resolvidos, para
conferência direta contra a Tabela 4 da monografia. A avaliação da dobra retida
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

```bash
python src/dados/run_pipeline.py --pool pool_206_rev     # etapas 2 e 3 sobre as 412 imagens
python src/dados/generate_folds.py --pool pool_206_rev   # particionamento, com conferência automática
```

Resultado da preparação (conjunto de 206 cartas): 412 de 412 imagens
processadas, nenhuma anotação descartada nem cortada. A ocupação média da carta
no quadro passa de 86,5% para 93,8%. Dobras com 39, 39, 38, 45 e 45 cartas.

### Particionamento

A unidade do particionamento é a carta, não a imagem: frente e verso da mesma
carta pertencem sempre à mesma dobra, por serem a mesma peça física. A
estratificação é feita pelo número de faces com defeito, e cartas de mesma
ilustração caem sempre na mesma dobra. Cada carta é avaliada exatamente uma
vez, na dobra a que pertence. Cerca de 20% das cartas de treino de cada rodada
formam a validação interna, que escolhe a época, o limiar de confiança do F1 e,
na dobra 0, o operador de realce.

O `generate_folds.py` confere o resultado por conta própria e falha se qualquer
invariante for violada.

## Análises e reprodução dos números do TCC

Os números do texto saem dos scripts abaixo, a partir das predições de cada
rodada. Todos rodam de qualquer pasta e gravam em `resultados/`.

| Script | O que gera |
|---|---|
| `src/treino/compare_paired.py` | Recall por faixa de tamanho e Wilcoxon por carta (F1 e recall), com p bruto e ajustado por Holm: `resultados/comparacao_pareada_206_rev.csv` |
| `src/treino/pick_variant.py` | Escolha do operador de realce pela validação interna da dobra 0: `resultados/selecao_realce_206.csv` |
| `src/analise/reevaluate_human_only.py` | Métricas sem as caixas sugeridas pelo modelo (só anotação humana), com Holm |
| `src/analise/interaction_test.py` | Teste de interação aumento × realce por carta |
| `src/analise/resolution_wilcoxon.py` | 1280 × 640 e yolov8s × yolov8m, por carta, na dobra 4 |
| `src/analise/learning_curve_bootstrap*.py` | Intervalos de confiança da curva de aprendizado (bootstrap por carta, 2.000 repetições) |
| `src/analise/h4_small_recall.py` | Revocação dos defeitos menores que 8 px, com e sem a cabeça P2 |
| `src/analise/check_folds.py` | Conferência das dobras e da curva (aninhamento, grupos de mesma arte) |

As predições de cada rodada ficam em `resultados/campanha_206/predicoes/*.json.xz`
(descompactar em `Dataset_YOLO/runs_206_rev/<rodada>_avaliacao/predictions.json`).
Os scripts também leem os rótulos e o tamanho das imagens, que não são
versionados (material cedido por terceiro); com o conjunto em `Dataset_YOLO/`,
todos os números se refazem sem treinar de novo.

## Ambiente

```bash
python3 -m venv .venv
./.venv/bin/pip install torch==2.14.0+rocm7.2 torchvision==0.29.0+rocm7.2 --index-url https://download.pytorch.org/whl/rocm7.2
./.venv/bin/pip install -r requirements.txt
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

Requer TeX Live com `abntex2` e `pgfgantt`:

```bash
sudo apt install texlive-latex-recommended texlive-latex-extra \
  texlive-fonts-recommended texlive-lang-portuguese texlive-publishers \
  texlive-pictures texlive-bibtex-extra
```

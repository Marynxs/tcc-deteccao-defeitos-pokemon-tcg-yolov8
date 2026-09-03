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

O desenho é um estudo de ablação fatorial 2x2, combinando duas variáveis
independentes:

| | Representação RGB | Realce de superfície |
|---|---|---|
| **Sem aumento de dados** | Configuração 1 | Configuração 3 |
| **Com aumento de dados** | Configuração 2 | Configuração 4 |

A avaliação usa validação cruzada estratificada em cinco dobras, com folds fixos
entre as quatro configurações, e comparação por teste de Wilcoxon pareado tendo
a carta como unidade de análise.

## Conjunto de dados

82 cartas fornecidas por empresa brasileira especializada em graduação,
totalizando 164 imagens (frente e verso) e 1.038 instâncias anotadas em classe
única, `defeito`. As anotações derivam dos registros de avaliadores
profissionais.

O conjunto **não é versionado** neste repositório, por seu volume (6,5 GB) e por
ser material cedido por terceiro.

## Estrutura

```
tcc1/     Monografia do TCC1, entregue e aprovada. Congelada.
tcc2/     Monografia do TCC2, em elaboração, mais os slides da defesa.
src/      Código: preparação dos dados e treinamento.
```

## Pipeline de dados

Os scripts em `src/dados/` executam nesta ordem, sempre a partir da raiz do
repositório:

| Ordem | Script | O que faz |
|---|---|---|
| 1 | `organize_cards.py` | Padroniza a nomenclatura das imagens de origem e gera a planilha de mapeamento. |
| 2 | `card_detector.py` | Detecta a carta na foto. Modelo de fundo por z-score em Lab normalizado pelo desvio robusto, com evidência de textura e validação pela proporção física 63x88 mm. |
| 3 | `crop_and_adjust.py` | Corta pela caixa detectada e recalcula todas as anotações YOLO para o novo enquadramento. Importa a detecção da etapa anterior, de modo que as duas etapas enxergam a mesma caixa. |
| 4 | `run_pipeline.py` | Executa as etapas 2 e 3 sobre o conjunto completo de 164 imagens. |
| 5 | `generate_folds.py` | Gera as cinco dobras por carta, estratificadas, e grava os arquivos de partição. |

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

```bash
python src/dados/run_pipeline.py      # etapas 2 e 3 sobre as 164 imagens
python src/dados/generate_folds.py     # particionamento, com conferência automática
```

Resultado da preparação: 164 de 164 imagens processadas, 1.038 de 1.038
anotações preservadas, nenhuma descartada. A ocupação média da carta no quadro
passa de 82,4% para 93,7%.

### Particionamento

A unidade do particionamento é a carta, não a imagem: frente e verso da mesma
carta pertencem sempre à mesma dobra, por serem a mesma peça física. A
estratificação é feita pelo número de faces com defeito, de modo que as cinco
cartas sem defeito em nenhuma face fiquem distribuídas uma por dobra. Cada carta
é avaliada exatamente uma vez, na dobra a que pertence.

O `generate_folds.py` confere o resultado por conta própria e falha se qualquer
invariante for violada.

## Ambiente

```bash
python3 -m venv .venv
./.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
./.venv/bin/pip install ultralytics opencv-python numpy pillow
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

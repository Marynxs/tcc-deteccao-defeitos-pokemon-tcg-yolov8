# Resultados das campanhas de treinamento

Arquivos pequenos de cada rodada, copiados de `Dataset_YOLO/runs*/` (que nao e
versionada por causa dos pesos e das imagens), e as saidas das analises. Sao os
numeros que entram no TCC.

- `campanha_206/`: campanha principal, 206 cartas. Fatorial 2x2 x 5 dobras
  (`cfg1` a `cfg4`), ablacao da cabeca P2 (`cfg2_*_p2`), selecao do realce na
  dobra 0 (`cfg3_fold0_1280_realce_{a,b,d,e}`), curva de aprendizado
  (`*_curva25/50/75`), resolucao 640 (`cfg2_fold4_640`) e capacidade
  (`cfg2_fold4_1280_m`). Por rodada: `resumo.json` (metricas na dobra retida,
  F1 com o limiar da validacao interna), `results.csv` (curva epoca a epoca),
  `args.yaml` (hiperparametros exatos).
- `campanha_206/predicoes/`: predicoes caixa a caixa de cada rodada na dobra
  retida (`predictions.json` compactado com xz).
- `comparacao_pareada_206_rev.csv`: Wilcoxon por carta (F1 e recall), p bruto e
  ajustado por Holm sobre a familia de pares (2x1, 4x3, 1x3, 2x4, 2xP2). Gerado
  por `src/treino/compare_paired.py`; o texto do mesmo script fica em
  `comparacao_pareada_206_rev.txt`.
- `selecao_realce_206.csv`: escolha do operador de realce pela validacao interna
  (`src/treino/pick_variant.py`).
- `analises/`: saidas de `src/analise/` (sem as caixas sugeridas, interacao,
  resolucao e capacidade, curva com bootstrap, H4, conferencia das dobras) e a
  origem de cada caixa (`origem_caixas.json`: original, sugerida ou manual).
- `campanha_82/`: testes preliminares com 82 cartas (fatorial 2x2 x 5 dobras),
  mais a rodada `cfg2_fold1_1696`. `folds.json` e a particao usada.
- `teste_externo_82_em_124/`: os 20 modelos da campanha_82 avaliados nas 248
  imagens das 124 cartas novas.
- `folds_206.json`: particao das 206 cartas, com os grupos de mesma arte.

Pesos (`best.pt`) e imagens ficam fora do repositorio; ver `Dataset_YOLO/runs*/`.

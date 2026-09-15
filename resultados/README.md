# Resultados das campanhas de treinamento

Arquivos pequenos de cada rodada, copiados de `Dataset_YOLO/runs*/` (que nao e
versionada por causa dos pesos e das predicoes). Sao os numeros que entram no
TCC.

- `campanha_82/`: fatorial 2x2 x 5 dobras sobre as 82 cartas (testes
  preliminares), mais a rodada `cfg2_fold1_1696`. Por rodada: `resumo.json`
  (metricas na dobra retida), `results.csv` (curva epoca a epoca),
  `args.yaml` (hiperparametros exatos). `folds.json` e a particao usada.
- `teste_externo_82_em_124/`: os 20 modelos da campanha_82 avaliados nas 248
  imagens das 124 cartas novas.
- `folds_206.json`: particao das 206 cartas, com os grupos de mesma arte.

Pesos (`best.pt`) e predicoes caixa a caixa (`predictions.json`) ficam fora do
repositorio; ver `Dataset_YOLO/runs*/`.

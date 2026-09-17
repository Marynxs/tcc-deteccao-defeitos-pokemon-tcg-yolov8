# Revisão assistida da anotação (15/09/2026)

Motivo: a análise de erros (`docs/analise_erros_cfg2.md`) mostrou que a maior
parte dos "falsos positivos de fundo" são marcas reais sem anotação, e que
marcas iguais aparecem ora anotadas, ora não. Esta passada corrige a
consistência; não corrige completude (o que os modelos não veem continua
fora).

## Como as propostas foram geradas

`src/ferramentas/propose_missing.py`, modelos `best.pt` da cfg 2 (RGB com
aumento) da campanha das 82, imgsz 1280, conf 0,05 por modelo, aumento de
teste ligado.

- **124 cartas novas (083 a 206):** comitê dos 5 modelos; caixas fundidas por
  média ponderada; voto = soma das confianças / 5; precisa de 2 modelos.
- **82 cartas antigas:** SÓ o modelo da dobra em que a carta ficou retida
  (nunca a viu). Os outros quatro treinaram com ela e com a anotação dela e
  aprenderam a chamar de fundo as marcas não anotadas. Voto = confiança dele.
  Medido: com o comitê saíam 615 propostas nas antigas; com o retido só, 778.
- Sobrevive só o que não encosta em anotação existente (IoU < 0,1, centros
  fora um do outro).

Saída: `Dataset_YOLO/revisao_206/candidatos.csv` (uma linha por proposta,
coluna `decisao` para o autor) e `revisao_206/imagens/<carta>_<face>.png`
(carta com propostas numeradas + recortes ampliados com voto).

## Números

1.189 propostas em 306 das 412 imagens. 721 no verso, 468 na frente.

| escala | ≥ 0,10 | ≥ 0,15 | ≥ 0,20 | ≥ 0,30 | ≥ 0,40 | ≥ 0,50 |
|---|---|---|---|---|---|---|
| antigas (confiança do retido) | 778 | | 392 | 194 | 98 | 52 |
| novas (voto do comitê) | 411 | 208 | 107 | 24 | 6 | |

As escalas não são comparáveis entre si: nas antigas é a confiança de um
modelo só; nas novas é a soma de cinco dividida por cinco.

## Protocolo (Xu2022, Levy2021, Arjmandi2026, ver referencias_novas.md)

1. Critério escrito ANTES de começar, em três linhas: sujeira solta conta?
   ponto de tinta de fábrica conta? padrão impresso nunca conta. Aplicar
   também às anotações antigas.
2. Ordem: antigas ≥ 0,30 (194) e novas ≥ 0,20 (107) primeiro, para fixar o
   critério. Depois antigas ≥ 0,20 e novas ≥ 0,15. Abaixo disso só se a taxa
   de aceitação continuar alta.
3. Decisão no CSV: `s` aceita; qualquer outra coisa rejeita. "Não sei" fica
   em branco e vai para uma lista de decisão posterior, não vira anotação.
4. **Amostra de controle às cegas:** 20 imagens revisadas SEM olhar as
   propostas, marcando o que o autor vê; depois comparar com as propostas.
   Mede o que o autor acha e o modelo não, e o quanto a proposta influencia.
5. Registrar: aceitas, rejeitadas, adicionadas por conta própria, e a AP
   entre as propostas e o rótulo final (o Xu2022 reporta 33).
6. `python src/ferramentas/apply_review.py` gera `pool_206_rev`,
   `dataset_206_rev` e `folds_206_rev` com a MESMA partição de `folds_206`.
   Originais intactos. Treinar com `--pool pool_206_rev`.

## Riscos, medidos na literatura

- Viés de confirmação: com sugestão na tela, o recall do que a sugestão não
  cobria caiu de 64% para 31% e os anotadores não perceberam (Levy2021).
- Rótulo ajustado a partir de proposta fica menos preciso que do zero
  (Arjmandi2026), embora mais consistente entre pessoas.
- 30% de anotações omitidas custam 5 pontos de mAP em VOC (Wu2019); sem
  medição em defeito minúsculo.

O que a passada NÃO faz: achar os riscos fracos que os modelos perdem (70%
dos defeitos anotados). Isso exige passada manual, concentrada nas molduras
dos versos.

## Resultado da revisão (16/09/2026)

Revisão feita pelo autor em `review_app.py` durante 15 e 16/09/2026, sobre as
1.189 propostas iniciais mais 117 propostas regeradas nas seis cartas
ladrilhadas (ver abaixo). Números finais, medidos em `revisao_206/`:

- propostas do modelo: **892 aceitas, 414 rejeitadas** (motivos: 188 arte,
  92 fora da carta, 65 nada visível, 4 sujeira, restante sem motivo ou aninhada)
- caixas desenhadas à mão pelo autor: **963**
- anotações originais: **156 removidas, 3 editadas**
- total: 3.109 → **4.808 anotações** (`pool_206_rev`, mesma partição de folds)
- **15 imagens marcadas como "desgaste difuso"** (`superficie.csv`): as doze
  faces das cartas 047, 073, 074, 075, 076 e 077, mais 016 frente, 072 frente
  e 108 verso.

### Critério de anotação (escrito pelo autor com o orquestrador, 16/09/2026)

É o texto que o TCC deve citar como protocolo. Vale para toda caixa, antiga ou nova.

1. **Uma marca contínua, uma caixa**, justa à marca. Ponto de sujeira de 5 px
   ganha caixa de 5 px; mancha de 300 px ganha caixa de 300 px; vinco com
   várias linhas contínuas ganha uma caixa cobrindo o vinco. O tamanho não é
   critério; a continuidade é.
2. **Marcas separadas por fundo bom são caixas separadas.** Seis pontos pretos
   com arte intacta entre eles são seis caixas, não um retângulo que os
   contenha. Pontinhos que se tocam formando um borrão são uma marca.
3. **Borda desgastada é uma faixa por trecho contínuo de desgaste**, do ponto
   onde começa ao ponto onde termina, com a largura da faixa esbranquiçada.
   Não se divide a borda em quadrados em sequência. Canto gasto contido na
   faixa não recebe caixa própria.
4. **Desgaste difuso** (véu de micro-riscos ou micro-pontos sem começo nem
   fim, típico de verso azul muito manuseado e de holográficas) **não recebe
   caixa**. É registrado por imagem em `superficie.csv` (tecla G) e declarado
   no texto como limitação da anotação por caixa.
5. **Padrão de impressão não é defeito**: cristais e brilhos do foil, pontos
   de luz nos olhos das ilustrações, textura do holo. Risco em holográfica é
   defeito quando atravessa ou interrompe o desenho; se acompanha o desenho, é
   o desenho.
6. Caixa cujo conteúdo é o fundo da foto (fora da carta) é erro e sai.

### O que aconteceu com as cartas ladrilhadas

Seis cartas (047, 073, 074, 075, 076, 077) tinham, já na anotação original do
Roboflow, o padrão de **ladrilho**: retângulos grandes, colados uns nos
outros, cobrindo a região gasta em vez de cada marca. O autor, ao revisar,
continuou o mesmo padrão nelas (40 a 102 caixas manuais por imagem). O
`audit_review.py` detectou isso por geometria (bordas compartilhadas em dois
ou mais lados; 431 caixas em grade, metade delas originais). Correção
aplicada em 16/09: 144 originais e 219 manuais em grade removidas; os
ladrilhos de borda substituídos por uma faixa por lado (45 faixas, 8 depois
retiradas por já existir faixa do autor); 117 propostas novas geradas
nessas doze imagens contra o rótulo limpo; e o autor reanotou as doze faces
pelo critério acima (205 caixas novas, mediana da dimensão menor 18 px @1280,
contra 13 px do restante do dataset).

Consequência para o texto: o ladrilhado na anotação original é um achado
sobre a **qualidade da anotação do conjunto de 82 cartas** que alimentou os
testes preliminares, e explica parte dos falsos positivos de fundo em verso
azul. Entra na limitação da anotação e na justificativa da revisão.

### Verificação automática de consistência (audit_review.py)

Duas famílias de alerta, nenhuma corrige nada: regras de forma (caixa acima
do p99 das originais, sobreposição IoU ≥ 0,3, grade, centro a menos de 30 px
da borda do quadro) e discordância com o modelo que não treinou com a carta.
O `pesquisador` verificou o respaldo (lote 16/09 em `referencias_novas.md`,
NÃO para o TCC): a parte "modelo aponta o que falta" é o *Confident Learning*
(Northcutt 2021) e o *ObjectLab* (Tkachenko 2023); a parte "modelo não vê a
caixa, logo ela é suspeita" é contraindicada por esses mesmos artigos e por
Schubert 2024, e foi rebaixada a informação. Os alertas de forma foram
revistos pelo autor: ao fim, sobreposições resolvidas (1 par mantido como
duas marcas), 68 caixas junto da borda do quadro conferidas como carta.

### Pendências que ficaram

- Amostra de controle às cegas: o autor reabriu as cartas 001 a 005 e
  considerou o critério estável. Não foi medida concordância numérica; se o
  texto precisar de número, refazer com 20 imagens e H ligado.
- AP entre propostas e rótulo final (Xu2022, prova de que o rótulo não virou
  cópia do modelo): calcular sobre `pool_206_rev` antes da campanha.
- Métrica secundária tolerante a deslocamento no `train_fold.py`.

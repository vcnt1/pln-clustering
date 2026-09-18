# ADR-0008: Abordagens de modelo — A (TF-IDF) primeiro, C (embeddings congelados) depois

- **Status:** aceito
- **Data:** 2026-09-18
- **Princípios tocados:** P3, P4
- **Substitui:** ADR-003 e ADR-004 de [`mood-ml/specs/00-decisoes.md`](../mood-ml/specs/00-decisoes.md)
- **Depende de:** [ADR-0001](ADR-0001-escala-do-humor.md) (escala contínua, regressão), [ADR-0002](ADR-0002-origem-do-ground-truth.md) (rótulo por mensagem), [ADR-0007](ADR-0007-humor-por-conversa.md) (janela de 30 mensagens da conversa)
- **Referência:** `data-structure/data-model.md`, seções 4.3, 4.4, 6 (T3) e 9; `README.md`, "Arquitetura do modelo"

---

## Contexto

A arquitetura do modelo e o formato de features (T3) seguiam em aberto no README e no
modelo de dados. O registro do mood-ml os fechou localmente:

1. **ADR-003 (mood-ml)** define como o modelo consome a janela da ADR-0007: dois
   blocos, a mensagem disparadora e as mensagens anteriores da conversa. A versão
   original usava só a disparadora (`history_window = 1`) e foi corrigida em
   2026-09-18 para seguir a ADR-0007.
2. **ADR-004 (mood-ml)** fixou TF-IDF + Ridge e descartou embeddings, o que impede
   comparar as duas famílias de representação do texto.

A arquitetura do modelo afeta o manifesto, o dataset, as dependências e o orçamento
de latência. Por isso pertence ao registro canônico, e não a uma decisão local do
mood-ml.

Foram escolhidas duas abordagens para o projeto: **A**, com vetorização clássica, e
**C**, com transformer pré-treinado **sem fine-tuning**. O orçamento de latência
continua em aberto.

## Decisão

### Pipeline comum

`T1 → T2 → T3 → Ridge`, com saída recortada para `[-1, 1]` (ADR-0001). O baseline
obrigatório é `DummyRegressor(strategy="mean")`. As duas abordagens diferem **apenas na
vetorização de T3**.

### Entrada de T3: dois blocos

T3 recebe o `history` (1 a 30 mensagens `customer` da conversa, em ordem crescente,
ADR-0007), aplica T1 e T2 a cada item e produz dois blocos:

- **texto** (`text_clean: str`): o último item do `history`, a mensagem disparadora,
  que carrega o rótulo (ADR-0002);
- **contexto** (`context_clean: list[str]`): os itens anteriores, até 29, em ordem
  cronológica. É uma lista vazia no início de toda conversa (`len(history) == 1`).

O alvo continua sendo o humor da mensagem disparadora. O contexto ajuda a interpretá-la,
mas não muda o que é rotulado.

No treino, o mood-ml reconstrói o `history` de cada exemplo a partir do corpus com a
mesma regra da API: só mensagens `customer`, da mesma conversa, anteriores à
disparadora, no máximo 30 itens com ela. Os dois lados chamam a mesma
`extract_features` (`mood-ml/transform/features.py`).

`manifest.history_window = 30` e `history_scope = "conversation"` nas duas abordagens.

### Abordagem A — implementada primeiro

- Bloco texto: `FeatureUnion(tfidf_palavra[1-2], tfidf_char_wb[2-5])`.
- Bloco contexto: itens concatenados em um único texto → `tfidf_palavra[1-2]`.
- Os blocos são combinados por `ColumnTransformer`, com o peso do bloco contexto em
  `configs/pipeline.yaml`.
- `algorithm = "tfidf-ridge"`.

### Abordagem C — implementada depois

- Encoder de sentenças multilíngue, pré-treinado e **congelado**. Só a cabeça Ridge é
  treinada.
- Bloco texto = embedding da mensagem disparadora. Bloco contexto = média dos embeddings
  das mensagens anteriores (vetor nulo quando não há contexto).
- O modelo de encoder específico é fixado na spec de T3 quando C começar.
- `algorithm = "embeddings-ridge"`.

### Coexistência e comparação

- As duas abordagens vivem no `main`, selecionadas por configuração
  (`configs/pipeline.yaml`). Cada treino gera uma `model_version` própria.
- `datasets/*` guarda o **texto** após T1 + T2 (`text_clean` e `context_clean`), nunca
  vetores. A vetorização vive dentro do artefato serializado, garantindo a paridade
  exigida por P4.
- `feature_spec_version = "fs-1"` versiona T1, T2 e a montagem dos blocos, que são
  comuns às duas abordagens. Por isso A e C treinam sobre o mesmo `dataset_id`. A
  vetorização é identificada por `algorithm` e viaja dentro do artefato.
- A comparação usa o mesmo `dataset_id`, o split por `customer_id` e as métricas de
  regressão da ADR-0001 (MAE, RMSE e Spearman), além da latência p95 de inferência.
- A versão ativa é definida em ADR posterior, com base na spec de avaliação.

## Alternativas consideradas

### Abordagem B (Word2Vec/GloVe + LSTM/GRU)

**Descartada.** Exige treinar a rede sequencial do zero, o que não se sustenta com o
corpus sintético do MVP, e tem custo intermediário sem o contexto profundo de C.

### Abordagem C com fine-tuning

**Descartada para o MVP.** Exige GPU e dados rotulados reais em volume. Fica como
evolução em escala.

### Só a mensagem disparadora (versão original da ADR-003 do mood-ml)

**Descartada.** Conflita com a ADR-0007 e deixa de usar a janela que a API já monta.

### Só a janela concatenada, com a disparadora dentro

**Descartada.** Com até 30 mensagens de peso igual, a disparadora, que é a dona do
rótulo, se dilui no texto.

### Contexto como string única no dataset

**Descartada.** Serve à abordagem A, mas impede a média por mensagem da abordagem C sem
um separador frágil. A lista preserva as mensagens, e cada abordagem as agrega dentro
do próprio artefato.

### Classificação multiclasse nos 5 níveis

**Descartada.** Ignora a ordem entre os níveis e contraria a escala contínua da ADR-0001.

### Uma branch por abordagem

**Descartada.** O código comum diverge, e a comparação deixa de acontecer sobre o mesmo
commit.

## Consequências

**Facilita**

- Experimento controlado: só a vetorização do texto muda entre A e C.
- A entrega um modelo funcional cedo, com latência de milissegundos em CPU, e serve de
  baseline para C.
- A arquitetura do modelo passa a ter uma única fonte de verdade, no registro canônico.

**Dificulta**

- **A janela é montada em dois lugares:** pela API na trilha online e pelo mood-ml a
  partir do corpus no treino. As duas implementações precisam de um teste de contrato
  que garanta o mesmo resultado; caso contrário, P4 é violado de forma silenciosa.
- O corpus tem conversas de até 20 mensagens (DC-R15 do contrato `dc-1`), então o treino
  nunca vê janelas cheias. Contextos maiores em produção ficam fora da distribuição de
  treino. Esse limite deve ser declarado no TCC.
- C adiciona dependências pesadas (torch, sentence-transformers) e maior latência em CPU.
  Elas ficam em um arquivo de requirements separado, instalado só quando C for usado.
- Com ground truth sintético (ADR-0002), as duas abordagens podem saturar as métricas, e
  a diferença entre elas pode ser pouco informativa. Esse limite deve ser declarado no
  TCC.
- Sem fine-tuning, C pode não superar A no domínio de hotelaria em PT-BR. Esse resultado
  é válido e deve ser reportado.

**Passa a ser proibido**

- Mudar a montagem dos blocos, o tamanho ou o escopo da janela sem nova
  `feature_spec_version` e nova `model_version` (P4).
- Guardar vetores em `datasets/*`.

**Em aberto, adiado deliberadamente**

- Encoder específico de C (spec de T3, quando C começar).
- Versão ativa do modelo e orçamento de latência (spec de avaliação e ADR posterior).
- Ponderação por recência dentro do bloco contexto (já em aberto na ADR-0007).

## Documentos alinhados

- `data-structure/data-model.md`: T3 em `transform/features.py`; colunas
  `text_clean`, `context_clean` e `persona` em `datasets/*`; `algorithm` no manifesto;
  "Formato de features" movido para 9.1.
- `README.md`: "Arquitetura do modelo" movida para "Decisões Fechadas".
- `mood-ml/specs/00-decisoes.md`: ADR-003 e ADR-004 marcadas como substituídas por esta
  ADR.
- `mood-ml/specs/01-dataset-contract.md` e `mood-ml/transform/features.py`: referências
  à ADR-003 passam a seguir esta ADR.

# 00 — Decisões de Arquitetura (ADRs) do mood-ml

Registro das decisões que destravam o primeiro treino. Cada ADR fecha um item "em aberto" do [data-model.md](../../data-structure/data-model.md) (seções 6 e 9).

Formato: Contexto → Decisão → Consequências → Alternativas. Uma ADR aceita só muda por uma nova ADR que a substitua. O texto da ADR antiga não é editado.

**Precedência.** Estas ADRs especializam as ADRs gerais do projeto ([decisions/](../../decisions/)) e não podem contradizê-las. Em conflito, vale a ADR geral.

> **Correção (2026-09-18):** as ADR-001, 003 e 004 foram corrigidas para seguir a [ADR-0001](../../decisions/ADR-0001-escala-do-humor.md) (T6 fora do escopo, métricas só de regressão) e a [ADR-0007](../../decisions/ADR-0007-humor-por-conversa.md) (janela de 30 mensagens da conversa). A versão anterior contradizia as ADRs gerais por erro de redação, e não por uma decisão.

| ADR | Tema | Status | Data |
|---|---|---|---|
| 001 | Escala do humor e rótulos de treino | Aceita (corrigida) | 2026-09-18 |
| 002 | Grão do rótulo | Aceita | 2026-09-18 |
| 003 | Consumo da janela de contexto (T5) | Substituída por [ADR-0008](../../decisions/ADR-0008-abordagens-de-modelo.md) | 2026-09-18 |
| 004 | Arquitetura do modelo e formato de features (T3) | Substituída por [ADR-0008](../../decisions/ADR-0008-abordagens-de-modelo.md) | 2026-09-18 |

> **Substituição (2026-09-18):** a arquitetura do modelo e o formato de features passaram ao registro canônico, na [ADR-0008](../../decisions/ADR-0008-abordagens-de-modelo.md). Ela mantém os dois blocos da ADR-003 e a abordagem TF-IDF + Ridge da ADR-004 (abordagem A), guarda o contexto como lista e acrescenta a abordagem C (embeddings congelados). As ADR-003 e 004 abaixo ficam só como registro histórico.

---

## ADR-001 — Escala `-1 to 1` (ADR-0001) com rótulos de treino em 5 níveis

**Contexto.** A escala foi fixada pela [ADR-0001](../../decisions/ADR-0001-escala-do-humor.md): contínua `-1 to 1`, problema de regressão, `mood_label` nulo e T6 fora do escopo do MVP. Falta definir como o dataset sintético rotula as mensagens, porque gerar rótulos contínuos arbitrários produz ruído e inconsistência.

**Decisão.**
- `scale = "-1 to 1"` para todas as versões v1.x do modelo (ADR-0001).
- Rótulo de treino discreto em 5 níveis: `-1.0`, `-0.5`, `0.0`, `0.5`, `1.0`. São pontos da escala contínua, e não classes.
- Saída do modelo contínua (regressão), recortada para [-1, 1].
- **Sem T6:** `mood_label = null` no `InferResponse` e `manifest.mood_labels = null`. O frontend continua mapeando score para emoji.
- Métricas só de regressão: MAE (métrica principal), RMSE e Spearman.

**Consequências.**
- O contrato `mood_scores` não muda (`score` DOUBLE + `scale`).
- Os níveis discretos simplificam a rubrica do gerador (spec 01, §4) sem transformar o problema em classificação.
- Métricas de classificação (acurácia, macro-F1) não se aplicam (ADR-0001).

**Alternativas.** Somente 3 níveis (perde intensidade, que é o valor do produto). Rótulo contínuo livre (difícil de gerar de forma consistente).

---

## ADR-002 — Grão do rótulo = mensagem de cliente

**Contexto.** O label set suporta `message` e `conversation` (data-model 4.3). A inferência online é disparada por mensagem de cliente (data-model 5).

**Decisão.** `target_type = "message"`. Só mensagens `role = customer` recebem rótulo. Mensagens `agent` têm rótulo nulo.

**Consequências.**
- Treino e inferência ficam no mesmo grão.
- O humor da conversa ou do cliente continua sendo derivado pelas views `v_*_mood_latest`, e não rotulado.
- Mensagens `agent` permanecem no corpus para uso futuro, mas não entram na janela de contexto (ADR-0007, ADR-0008).

**Alternativas.** Rótulo por conversa: não ensina a variação mensagem a mensagem e reduz o número de exemplos em ~10×.

---

## ADR-003 — Consumo da janela de contexto (T5) de 30 mensagens da conversa

> **Substituída pela [ADR-0008](../../decisions/ADR-0008-abordagens-de-modelo.md).**

**Contexto.** A [ADR-0007](../../decisions/ADR-0007-humor-por-conversa.md) fixou a janela: as 30 mensagens `customer` mais recentes da conversa, em ordem crescente, com a disparadora como último item. `manifest.history_window = 30` e `history_scope = "conversation"` são conferidos no carregamento do modelo. Um modelo que declarasse 30 e usasse só a última mensagem burlaria essa verificação.

**Decisão.**
- O modelo consome a janela inteira (1 a 30 itens), dividida em dois blocos:
  - **texto:** o último item de `history` (a mensagem disparadora);
  - **contexto:** os itens anteriores (até 29), concatenados em ordem crescente.
- O alvo continua sendo o humor da **mensagem disparadora** (`target_type = "message"`, ADR-0002). O contexto ajuda a interpretá-la, mas não muda o que é rotulado.
- No treino, a janela é reconstruída a partir do corpus com a mesma regra da API: só mensagens `customer`, da mesma conversa, anteriores à disparadora, no máximo 29.
- `manifest.history_window = 30` e `manifest.history_scope = "conversation"`.

**Consequências.**
- `len(history) == 1` é o início de toda conversa: o contexto fica vazio. É caso de teste obrigatório no treino e na inferência.
- A spec 01 continua válida: o rótulo segue **inferível pelo próprio texto** (DC-R12), porque mede o humor da mensagem.
- O corpus tem conversas de até 20 mensagens (DC-R15), então o treino nunca vê janelas cheias. Contextos maiores em produção ficam fora da distribuição de treino. Isso é uma limitação a declarar no TCC.
- Mudar o tamanho, o escopo ou a divisão em blocos exige nova `feature_spec_version` e nova `model_version` (P4).

**Alternativas.** Concatenar a janela inteira em uma string (dilui a mensagem disparadora, dona do rótulo). Média de vetores com decaimento por recência (a ponderação por recência ficou em aberto na ADR-0007).

---

## ADR-004 — TF-IDF + Ridge em `sklearn.Pipeline`; dataset guarda texto limpo

> **Substituída pela [ADR-0008](../../decisions/ADR-0008-abordagens-de-modelo.md).**

**Contexto.** O formato de `features` (T3) e o algoritmo estavam em aberto. O MVP precisa de treino rápido, latência baixa e paridade treino/inferência (data-model 6, invariante 1).

**Decisão.**
- Modelo, com os dois blocos da ADR-003:
  ```
  Pipeline([
    ColumnTransformer([
      ("text",    FeatureUnion(tfidf_palavra[1-2], tfidf_char_wb[2-5]), "text_clean"),
      ("context", tfidf_palavra[1-2],                                  "context_clean"),
    ], transformer_weights={"context": <peso em configs/pipeline.yaml>}),
    Ridge,
  ])
  ```
- Baseline obrigatório: `DummyRegressor(strategy="mean")`.
- `datasets/*` guarda as colunas `text_clean` e `context_clean` (texto após T1 + T2), **não** vetores. `context_clean` é uma string vazia quando não há contexto. A vetorização vive dentro do artefato serializado (`model.joblib`).
- `feature_spec_version = "fs-1"` identifica a versão de T1 + T2 e da montagem dos blocos (T3).

**Consequências.**
- O vocabulário é ajustado apenas no split de treino, o que evita vazamento.
- Na inferência, o mesmo artefato vetoriza, o que garante a paridade treino/inferência.
- Dependências novas: `scikit-learn`, `joblib`, `pyarrow`.
- Latência esperada: milissegundos por mensagem na CPU.

**Alternativas.** `LogisticRegression` multiclasse: ignora a ordem dos níveis. Embeddings multilíngues: dependência pesada e ganho não comprovado com corpus sintético. BERTimbau fine-tuned: exige volume real (ver evolução em escala).

# 00 — Decisões de Arquitetura (ADRs) do mood-ml

Registro das decisões que destravam o primeiro treino. Cada ADR fecha um item "em aberto" do [data-model.md](../../data-structure/data-model.md) (seções 6 e 9).

Formato: Contexto → Decisão → Consequências → Alternativas. Uma ADR aceita só muda por uma nova ADR que a substitua. O texto da ADR antiga não é editado.

| ADR | Tema | Status | Data |
|---|---|---|---|
| 001 | Escala do humor | Aceita | 2026-09-18 |
| 002 | Grão do rótulo | Aceita | 2026-09-18 |
| 003 | Janela de contexto (T5) | Aceita | 2026-09-18 |
| 004 | Arquitetura do modelo e formato de features (T3) | Aceita | 2026-09-18 |

---

## ADR-001 — Escala `-1 to 1` com rótulos em 5 níveis

**Contexto.** A escala não estava fixada (data-model 3.5). O frontend já normaliza `-1 to 1`. Gerar rótulos contínuos arbitrários no dataset sintético produz ruído e inconsistência.

**Decisão.**
- `scale = "-1 to 1"` para todas as versões v1.x do modelo.
- Rótulo de treino discreto em 5 níveis: `-1.0`, `-0.5`, `0.0`, `0.5`, `1.0`.
- Saída do modelo contínua, recortada para [-1, 1].
- T6 (score → `mood_label`) por nível mais próximo:

| `mood_label` | Nível | Faixa de score |
|---|---|---|
| `muito_negativo` | -1.0 | [-1.00, -0.75) |
| `negativo` | -0.5 | [-0.75, -0.25) |
| `neutro` | 0.0 | [-0.25, 0.25] |
| `positivo` | 0.5 | (0.25, 0.75] |
| `muito_positivo` | 1.0 | (0.75, 1.00] |

**Consequências.**
- O contrato `mood_scores` não muda (`score` DOUBLE + `scale`).
- `manifest.mood_labels` declara a tabela acima. T6 passa a existir no mood-ml, e o frontend pode usar `mood_label` em vez de mapear sozinho.
- Permite duas famílias de métrica: de regressão (MAE, Spearman) e de classificação (macro-F1 após T6).

**Alternativas.** Escala `0 to 1` (menos intuitiva para "negativo"). Somente 3 classes (perde intensidade, que é o valor do produto). Rótulo contínuo livre (difícil de gerar de forma consistente).

---

## ADR-002 — Grão do rótulo = mensagem de cliente

**Contexto.** O label set suporta `message` e `conversation` (data-model 4.3). A inferência online é disparada por mensagem de cliente (data-model 5).

**Decisão.** `target_type = "message"`. Só mensagens `role = customer` recebem rótulo. Mensagens `agent` têm rótulo nulo.

**Consequências.**
- Treino e inferência ficam no mesmo grão.
- O humor da conversa ou do cliente continua sendo derivado pelas views `v_*_mood_latest`, e não rotulado.
- Mensagens `agent` permanecem no corpus para uso futuro como contexto (ADR-003).

**Alternativas.** Rótulo por conversa: não ensina a variação mensagem a mensagem e reduz o número de exemplos em ~10×.

---

## ADR-003 — Janela de contexto (T5) = 1 na v1

**Contexto.** O `InferRequest.history` transporta uma janela de tamanho indefinido.

**Decisão.** O modelo v1 usa somente o último item de `history`, que é a mensagem disparadora. `manifest.history_window = 1`.

**Consequências.**
- O contrato HTTP não muda: a API pode enviar mais itens, e o mood-ml ignora os excedentes.
- Consequência para o dataset: o rótulo de cada mensagem deve ser **inferível pelo próprio texto** (ver spec 01, regra DC-R12).
- A v2 testa janela > 1 contra este baseline, com uma nova `feature_spec_version`.

**Alternativas.** Janela N desde a v1: mais complexa, sem baseline para provar ganho.

---

## ADR-004 — TF-IDF + Ridge em `sklearn.Pipeline`; dataset guarda texto limpo

**Contexto.** O formato de `features` (T3) e o algoritmo estavam em aberto. O MVP precisa de treino rápido, latência baixa e paridade treino/inferência (data-model 6, invariante 1).

**Decisão.**
- Modelo: `Pipeline([FeatureUnion(tfidf_palavra[1-2], tfidf_char_wb[2-5]), Ridge])`.
- Baseline obrigatório: `DummyRegressor(strategy="mean")`.
- `datasets/*` guarda a coluna `text_clean` (texto após T1 + T2), **não** vetores. A vetorização vive dentro do artefato serializado (`model.joblib`).
- `feature_spec_version = "fs-1"` identifica a versão de T1 + T2.

**Consequências.**
- O vocabulário é ajustado apenas no split de treino, o que evita vazamento.
- Na inferência, o mesmo artefato vetoriza, o que garante a paridade treino/inferência.
- Dependências novas: `scikit-learn`, `joblib`, `pyarrow`.
- Latência esperada: milissegundos por mensagem na CPU.

**Alternativas.** `LogisticRegression` multiclasse: ignora a ordem dos níveis. Embeddings multilíngues: dependência pesada e ganho não comprovado com corpus sintético. BERTimbau fine-tuned: exige volume real (ver evolução em escala).

# 06 — Treino, Vetorização e Avaliação (camada `train/` + `evaluate/`)

**Status:** Aceita · **Versão da spec:** `tn-1` · **Data:** 2026-09-20
**Implementa:** [data-model.md §4.4](../../data-structure/data-model.md) (campos `algorithm`, `metrics`, `feature_spec_version` do manifesto) — este documento diz **como** o candidato é ajustado e **como** decide-se se ele merece virar uma `model_version`.
**Depende de:** [ADR-0001](../../decisions/ADR-0001-escala-do-humor.md) (regressão, saída recortada `[-1, 1]`, sem T6), [ADR-0002](../../decisions/ADR-0002-origem-do-ground-truth.md) (métricas sobre corpus sintético são otimistas), [ADR-0008](../../decisions/ADR-0008-abordagens-de-modelo.md) (pipeline `T1→T2→T3→Ridge`, abordagem A primeiro); [constitution.md](../../decisions/constitution.md) P3, P4, P5; [05-split.md](05-split.md) (portão do dataset)
**Consumido por:** `registry/registry.py` (spec 07) — só promove um candidato a `model_version` se `eval.json.gate.passed == true`
**Implementado em:** [train/train.py](../train/train.py), [evaluate/metrics.py](../evaluate/metrics.py)

---

## 1. Objetivo geral

Dois scripts, um fluxo: **ajustar um candidato e decidir, com um critério objetivo, se ele é bom o bastante para existir como versão de modelo.**

1. **`train/train.py`** — lê `text_clean`/`context_clean` do split `train`, vetoriza (TF-IDF, abordagem A da ADR-0008), ajusta o baseline obrigatório (`DummyRegressor`) e o candidato (`Ridge`) com hiperparâmetros vindos de `configs/pipeline.yaml`, e persiste os dois modelos ajustados — ainda **não avaliados, não registrados**.
2. **`evaluate/metrics.py`** — lê esses dois modelos e o split `test`, calcula MAE (métrica principal), RMSE e Spearman para os dois, aplica o *quality gate* (candidato precisa bater o baseline por uma margem mínima) e grava o veredito em `eval.json`.

Só depois disso — e só se o *gate* passar — é que `registry/registry.py` (spec 07) atribui um `model_version` e promove o candidato a artefato oficial. Este documento termina exatamente onde o registro começa: o produto de `evaluate/metrics.py` é uma **recomendação objetiva**, não uma publicação.

## 2. Arquitetura e fluxo de dados

### 2.1 Posição no fluxo — e por que existe um estágio intermediário

```
data/datasets/<dataset_id>/{train,validation,test}.parquet   (aprovado, spec 05)
       │
       ▼
train.train  ──►  models/_staging/<dataset_id>/<algorithm>/<fp>/
                       baseline.joblib
                       candidate.joblib
                       train_manifest.json
       │
       ▼
evaluate.metrics  ──►  models/_staging/<dataset_id>/<algorithm>/<fp>/
                            eval.json   (gate.passed: true | false)
       │
       ▼ (só se gate.passed == true)
registry.registry (spec 07)  ──►  models/<model_version>/{model.joblib, manifest.json}
```

`pipeline.py` (spec 09) expõe `train`, `evaluate` e `register` como **subcomandos independentes** — é possível rodar um sem os outros dois. Isso significa que `train.train` precisa persistir algo em disco para `evaluate.metrics` ler depois, em outra invocação de processo. Mas esse algo não pode viver em `models/<model_version>/`: essa pasta é reservada pela `data-model.md §4.4` para modelos **já aprovados**, com um `model_version` que **nunca é reutilizado**. Um candidato que ainda não passou pelo *quality gate* não tem — e não deveria ter — um `model_version`.

Por isso este documento introduz `models/_staging/`, um diretório que a `data-model.md` não documenta porque não existia até agora: é o espaço de candidatos **não publicados**, com identidade própria (§3.3), consumido só por `evaluate.metrics` e, se o *gate* passar, por `registry.registry`.

### 2.2 Passo a passo lógico — `train.train`

| Fase | Nome | O que faz |
|---|---|---|
| F0 | Resolução da entrada | Recebe `dataset_id` (ou caminho) e a configuração de `configs/pipeline.yaml` |
| F1 | Portão | Confirma que `data/datasets/<dataset_id>/dataset.json` existe e está íntegro (spec 05) |
| F2 | Identidade e idempotência | Calcula o *fingerprint* do treino (§3.3); se já existe staging completo com o mesmo *fingerprint*, pula direto para o log de encerramento |
| F3 | Carga | Lê `train.parquet` (e `validation.parquet`, só para diagnóstico — §3.4) |
| F4 | Vetorização + fit | Monta o `Pipeline` (§3.5), ajusta baseline e candidato sobre `train` |
| F5 | Sanity check | Confirma que as previsões sobre o próprio `train` são números finitos (§7) |
| F6 | Emissão | Grava `baseline.joblib`, `candidate.joblib`, `train_manifest.json`, atômicos |

### 2.3 Passo a passo lógico — `evaluate.metrics`

| Fase | Nome | O que faz |
|---|---|---|
| F0 | Resolução da entrada | Recebe a mesma identidade de staging que `train.train` produziu |
| F1 | Portão | Confirma que os três artefatos de staging existem e estão íntegros |
| F2 | Carga | Lê `test.parquet` e os dois modelos (`joblib.load`) |
| F3 | Métricas | MAE, RMSE, Spearman para candidato e baseline; MAE do candidato por `persona`, MAE do candidato para exemplos sem contexto (`context_clean = []`), e latência p95 da predição do candidato sobre o `test` (§3.2a) |
| F4 | *Quality gate* | Compara MAE do candidato contra o limiar relativo ao baseline (§3.4) |
| F5 | Emissão | Grava `eval.json` (sempre, gate tendo passado ou não); encerra com o *exit code* correspondente |

## 3. Requisitos técnicos e premissas

### 3.1 Conexões e autenticação

**Não há.** Mesma razão das specs 02–05: tudo é arquivo local. `train.train` depende de `data/datasets/<dataset_id>/`; `evaluate.metrics` depende do staging que `train.train` produziu e de `data/datasets/<dataset_id>/test.parquet`.

```bash
python -m train.train <dataset_id> [--config configs/pipeline.yaml] [--staging-dir models/_staging]
python -m evaluate.metrics <dataset_id> [--config configs/pipeline.yaml] [--staging-dir models/_staging]
```

Runtime: Python 3.12. Nenhuma dependência nova além de `scikit-learn`, `joblib`, `scipy` (Spearman), já em `requirements.txt`. Dependências de `torch`/`sentence-transformers` (abordagem C) ficam **fora deste documento** (ADR-0008: "implementada depois", requirements separado).

Invocação pelo orquestrador (spec 09): `make train DATASET=<dataset_id>`, `make evaluate DATASET=<dataset_id>`, e `python -m pipeline train|evaluate --config configs/pipeline.yaml`.

### 3.2 Formato dos dados

| | Entrada | Saída (`train`) | Saída (`evaluate`) |
|---|---|---|---|
| Formato | Parquet (`datasets/*`) | joblib + JSON | JSON |
| Caminhos | `data/datasets/<dataset_id>/{train,validation,test}.parquet` | `models/_staging/<dataset_id>/<algorithm>/<fp>/{baseline,candidate}.joblib`, `train_manifest.json` | `models/_staging/.../eval.json` |
| Esquema | `data-model.md §4.3` (spec 05) | ver §3.3 | ver §3.4 |

`train_manifest.json`:

```json
{
  "fingerprint": "b7a1...9f0c",
  "dataset_id": "ds-2026-09-20-a",
  "dataset_sha256": { "train": "...", "validation": "...", "test": "..." },
  "algorithm": "tfidf-ridge",
  "feature_spec_version": "fs-1",
  "hyperparameters": { "ridge_alpha": 1.0, "context_weight": 0.5, "seed": 42, "...": "..." },
  "row_counts": { "train": 1680, "validation": 361, "test": 362 },
  "validation_mae_candidate": 0.184,
  "code_commit": "<git sha>",
  "trained_at": "2026-09-20T15:02:11Z",
  "duration_ms": 4210
}
```

`eval.json`:

```json
{
  "fingerprint": "b7a1...9f0c",
  "dataset_id": "ds-2026-09-20-a",
  "test_rows": 362,
  "metrics": {
    "candidate": { "mae": 0.171, "rmse": 0.229, "spearman": 0.812 },
    "baseline":  { "mae": 0.401, "rmse": 0.452, "spearman": 0.0 }
  },
  "mae_by_persona": { "objetivo": 0.165, "ansioso": 0.183 },
  "context_breakdown": {
    "with_context_rows": 300, "with_context_mae": 0.168,
    "no_context_rows": 62, "no_context_mae": 0.201
  },
  "latency": { "p50_ms": 3.1, "p95_ms": 6.4, "sample_size": 362 },
  "gate": {
    "threshold_ratio": 0.9,
    "candidate_mae": 0.171,
    "baseline_mae": 0.401,
    "required_max_mae": 0.361,
    "passed": true
  },
  "evaluated_at": "2026-09-20T15:04:47Z",
  "duration_ms": 980
}
```

### 3.2a Métricas adicionais: MAE sem contexto e latência p95

Duas métricas que `evaluate.metrics` DEVE calcular além de MAE/RMSE/Spearman globais e por `persona`:

**MAE por presença de contexto.** O split `test` tem exemplos com `context_clean = []` — o início de toda conversa (ADR-0007) — e exemplos com contexto não vazio. Reportar o MAE separado para os dois grupos responde a uma pergunta metodológica que o número agregado esconde: *o modelo depende do contexto para acertar, ou funciona igualmente bem no primeiro turno de uma conversa nova?* Como toda conversa nova passa por `len(history) == 1` (ADR-0007, spec 04 §2.2), um MAE muito pior no grupo sem contexto seria um sinal de que o modelo aprendeu a se apoiar demais no bloco `context`, o que é relevante tanto para o TCC quanto para decidir o peso do bloco contexto em `configs/pipeline.yaml` (`train.context.weight`).

**Latência p95 offline.** Diferente da latência medida em produção pela spec 08 (round-trip HTTP completo), esta é uma medição **isolada do modelo**: para cada linha do split `test`, simula uma inferência (`extract_features` + `Pipeline.predict` sobre uma única linha, o mesmo caminho que `infer/predict.py` percorre por requisição — nunca em lote, que mascararia o custo por chamada), mede o tempo decorrido, e reporta p50/p95 sobre a amostra inteira. É o primeiro ponto do pipeline em que a meta de **p95 < 200ms** (spec 08, D3) pode ser verificada — antes de qualquer deploy, ainda em avaliação offline. Uma abordagem C (embeddings) que já estourasse esse número aqui não precisaria nem chegar a ser registrada para essa limitação ficar documentada.

### 3.3 Estratégia de carga e idempotência

**Identidade sem escolha do operador.** Ao contrário de `corpus_id` e `dataset_id` (specs 02 e 05), a identidade de um treino não é dada por ninguém — é **derivada** do que entra nele:

```
fingerprint = sha256(canonical_json({
    "dataset_id": ...,
    "dataset_sha256": {"train": ..., "validation": ..., "test": ...},   # de dataset.json
    "algorithm": "tfidf-ridge",
    "hyperparameters": {...},   # tudo que está em configs/pipeline.yaml, seção train
    "feature_spec_version": "fs-1",
}))[:16]
```

O diretório de staging é `models/_staging/<dataset_id>/<algorithm>/<fingerprint>/`. Como o caminho **é** a identidade, e a identidade **é** uma função do conteúdo, o problema que `dataset_id` teve (mesmo nome, conteúdos diferentes, exigindo um mecanismo de conflito com *exit* 4) **não existe aqui**: dois treinos com entradas diferentes caem em diretórios diferentes automaticamente. Não há como colidir.

| Situação | Comportamento | Exit |
|---|---|---|
| Staging inexistente | Treina e grava | 0 |
| Staging completo (3 arquivos) com o mesmo *fingerprint* | **No-op idempotente** — não relê os dados, não refita. Loga `already_trained` | 0 |
| Staging incompleto (execução anterior interrompida) | Refita do zero e sobrescreve | 0 |
| Portão reprovado (F1) | Aborta antes de qualquer leitura | 3 |

`evaluate.metrics` segue o mesmo raciocínio sobre o mesmo *fingerprint*: recalcula, compara com o `eval.json` existente se houver, e sobrescreve de forma determinística quando as entradas não mudaram (§7, EV-R10).

**Por que isso exige determinismo total do fit, não só do split.** A spec 05 já estabeleceu que `GroupShuffleSplit` com seed fixa é determinístico. Aqui a exigência se estende ao ajuste do modelo: `TfidfVectorizer` não tem aleatoriedade própria, mas o solver do `Ridge` pode ter (`'sag'`/`'saga'` usam amostragem aleatória internamente). Para que o *fingerprint* continue sendo uma garantia real de reprodutibilidade, e não uma promessa quebrada pelo solver, esta spec exige explicitamente que qualquer componente estocástico receba a `seed` do config (TR-R07/§7) — na prática, fixando o solver do `Ridge` para um determinístico (`'cholesky'`, `'lsqr'` ou `'sparse_cg'`, nenhum dos quais precisa de `random_state`) ou, se algum dia `'sag'`/`'saga'` for necessário por desempenho em escala maior, propagando a seed do config para ele.

### 3.4 Estratégia de hiperparâmetros

**Fit único, sem busca.** Todos os hiperparâmetros vêm de `configs/pipeline.yaml`, nenhum fixo no código — mas não há laço de busca (`GridSearchCV` ou equivalente) sobre o split `validation`. Ajustar um hiperparâmetro é editar o YAML e rodar `train.train` de novo; o *fingerprint* muda automaticamente (§3.3), então cada tentativa vira seu próprio diretório de staging, sem sobrescrever a anterior.

```yaml
train:
  algorithm: tfidf-ridge     # único valor implementado nesta spec; "embeddings-ridge" reservado (ADR-0008, abordagem C)
  seed: 42
  ridge:
    alpha: 1.0
    solver: cholesky         # determinístico; ver §3.3
  tfidf_word:
    ngram_range: [1, 2]
  tfidf_char:
    ngram_range: [2, 5]
    analyzer: char_wb
  context:
    ngram_range: [1, 2]
    weight: 0.5               # transformer_weights do bloco contexto (ADR-0008)
evaluate:
  quality_gate:
    max_mae_ratio: 0.9        # candidato precisa de MAE <= 0.9 x MAE do baseline
```

**Por que sem busca automática, apesar da `validation.parquet` existir.** Com o volume do MVP (milhares de exemplos, não milhões) e o prazo de um TCC individual, uma grade de hiperparâmetros multiplicaria o tempo de execução e a superfície de teste sem evidência de que o ganho compensa — o ponto fraco do MVP é a qualidade do gerador sintético (ADR-0002), não a escolha fina de `alpha`. **O split `validation` não fica sem uso**: `train.train` calcula e loga o MAE do candidato sobre `validation` como diagnóstico (`train_manifest.json.validation_mae_candidate`), útil para o operador perceber *overfitting* grosseiro entre tentativas manuais de hiperparâmetro, sem que nenhuma decisão automática dependa desse número. Uma busca automática fica registrada como evolução possível (§10), não implementada agora.

### 3.5 Tokenização, TF-IDF e vetorização

Arquitetura fixada pela ADR-0008, "abordagem A":

```
ColumnTransformer(
    transformers=[
        ("text", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1,2), analyzer="word")),
            ("char", TfidfVectorizer(ngram_range=(2,5), analyzer="char_wb")),
        ]), "text_clean"),
        ("context", Pipeline([
            ("join", FunctionTransformer(join_context_list)),
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), analyzer="word")),
        ]), "context_clean"),
    ],
    transformer_weights={"context": <configs/pipeline.yaml: train.context.weight>},
)
```

seguido de `Ridge(alpha=..., solver=...)`. O baseline é `DummyRegressor(strategy="mean")`, ajustado sobre o mesmo `y` (`label_score`), sem usar `X`.

**`join_context_list` vive dentro do `Pipeline` serializado, não em código externo.** `context_clean` chega como lista de strings (spec 04/05); `TfidfVectorizer` espera uma coluna de strings escalares. A junção (`" ".join(lista)`, string vazia se a lista for vazia) precisa acontecer **igualmente** no treino e na inferência online — e a forma mais segura de garantir isso não é replicar a mesma função em dois módulos (o erro clássico que quebra P4), é fazer a junção ser **parte do artefato**: um `FunctionTransformer` dentro do `ColumnTransformer`, que viaja dentro do `.joblib` e roda automaticamente sempre que alguém chama `.predict()`, em qualquer trilha. `infer/predict.py` (spec 08) não precisa saber que essa junção existe.

**Vocabulário ajustado só no `train`.** As duas instâncias de `TfidfVectorizer` (e a interna ao bloco contexto) chamam `.fit()` **exclusivamente** sobre o split `train` — nunca `validation`, nunca `test`. É a mesma disciplina já registrada na ADR-004 histórica (`00-decisoes.md`): vazamento de vocabulário infla artificialmente as métricas de validação/teste, e o `Pipeline` do scikit-learn já impõe isso por construção (`.fit()` uma vez sobre `train`, `.transform()`/`.predict()` sobre os demais) — o requisito aqui é **não violar essa garantia manualmente**, por exemplo ajustando um `TfidfVectorizer` solto fora do `Pipeline` antes de montar o `ColumnTransformer`.

**Saída recortada, e por um único ponto.** A ADR-0001 exige a saída em `[-1.0, 1.0]`. O `Ridge` em si não garante isso (é uma regressão linear, pode extrapolar). O recorte não é um passo do `Pipeline` — é uma função separada, `clip_score(raw: float) -> float`, definida em `evaluate/metrics.py` e reimportada por `infer/predict.py` (spec 08). Fica em `evaluate/` porque é a **avaliação** que primeiro precisa dela (métricas devem refletir o que o modelo realmente devolveria em produção, não o valor bruto do `Ridge`), e porque colocá-la dentro do `Pipeline` exigiria uma classe customizada não-trivial de serializar com `joblib` sem risco de quebrar entre versões do `scikit-learn`. Uma função pura de uma linha, importada nos dois lugares, é a solução mais simples que ainda respeita P4.

### 3.6 Volumetria estimada e frequência de execução

Herda a volumetria da spec 05: até ~28.000 linhas de `train` (70% do teto de 40.000), ~6.000 de `validation` e `test` cada.

| Etapa | Custo estimado no teto |
|---|---|
| `TfidfVectorizer.fit_transform` (dois blocos) | Segundos, para vocabulário de dezenas de milhares de exemplos curtos |
| `Ridge.fit` | Sub-segundo (solver fechado, poucas *features* esparsas) |
| `DummyRegressor.fit` | Instantâneo |
| `evaluate.metrics` (predição + métricas sobre `test`) | Sub-segundo |

Orçamento de desempenho: `train.train` em menos de 1 minuto no corpus esperado, poucos minutos no teto de 10×. `evaluate.metrics`, segundos em qualquer caso. Nenhum dos dois se aproxima de precisar de processamento distribuído ou GPU (reservado para a abordagem C, ADR-0008).

**Frequência.** Sob demanda, depois de `split` bem-sucedido. Diferente das specs 02–05, aqui a cadência **não é estritamente sequencial em uma direção só**: o operador pode rodar `train.train` várias vezes com hiperparâmetros diferentes sobre o mesmo `dataset_id` (cada um em seu próprio staging, §3.3) antes de rodar `evaluate.metrics` sobre o que parecer mais promissor.

## 4. Tratamento de erros e resiliência

### 4.1 Taxonomia

- **Erro de portão** — dataset (para `train`) ou staging de treino (para `evaluate`) ausente ou incompleto. *Exit* 3 nos dois scripts.
- **Erro de integridade interna** (`train`) — o fit produziu algo não-finito. É bug de configuração ou de dado, não do código em si necessariamente, mas precisa parar antes de persistir. *Exit* 4.
- **Reprovação do *quality gate*** (`evaluate`) — **não é um erro**, é um resultado de negócio válido: o candidato não é bom o bastante. *Exit* 5, mas `eval.json` é gravado normalmente.
- **Erro de uso** — CLI mal chamada. *Exit* 2.
- **Erro de ambiente** — I/O. *Exit* 1.

### 4.2 Política de retentativa

**Sobre o fit e a avaliação: nenhuma.** Determinísticos por construção (§3.3). Reexecutar sobre a mesma entrada e configuração dá o mesmo resultado — inclusive a mesma reprovação de *gate*, se for o caso.

**Sobre I/O: 3 tentativas, backoff 1s/2s/4s.** Mesmo padrão das specs 02–05, aplicado à leitura dos `.parquet`, à leitura/escrita dos `.joblib` e dos `.json`.

**Gravação atômica, em ordem.** Em `train.train`: `baseline.joblib`, depois `candidate.joblib`, depois `train_manifest.json` (marcador de sucesso — sua presença implica os dois modelos gravados). Em `evaluate.metrics`: só `eval.json`, sempre por `.tmp` + `os.replace()`.

### 4.3 *Exit codes*

**`train.train`**

| Exit | Significado |
|---|---|
| 0 | Treinado (ou reaproveitado de forma idêntica) |
| 2 | Erro de uso da CLI |
| 3 | Portão bloqueado — `dataset_id` sem `dataset.json` íntegro |
| 4 | Sanity check falhou — previsão não-finita sobre o próprio `train` |
| 1 | Erro interno ou de ambiente |

**`evaluate.metrics`**

| Exit | Significado |
|---|---|
| 0 | Avaliado, *gate* aprovado |
| 2 | Erro de uso da CLI |
| 3 | Portão bloqueado — staging de treino ausente ou incompleto |
| 5 | *Quality gate* reprovado — `eval.json` gravado, candidato não segue para o registro |
| 1 | Erro interno ou de ambiente |

`pipeline.py all` (spec 09) trata os *exit* 3, 4 e 5 como parada da esteira — nenhum deles chega a `register`, mas só o 4 é, de fato, um defeito a corrigir no código. O 5 é informação: o experimento rodou corretamente e não foi bom o bastante.

### 4.4 Alertas e notificações

**Nenhum canal externo**, mesma razão das specs anteriores. Uma nuance: a reprovação do *gate* (*exit* 5) é o tipo de evento que, num pipeline de produção com retrainings agendados, justificaria um alerta de verdade (silenciosamente nunca promover um modelo pior é o comportamento certo, mas alguém deveria saber que o retrain falhou em melhorar). No MVP, sob demanda e sem plantão, essa notificação é o próprio *exit code* visto por quem disparou o comando — registrado aqui como ponto a revisitar se o projeto ganhar um agendador (mesma condição de revisão já usada nas specs 02–05).

## 5. Logs e observabilidade

### 5.1 Padrão

Mesmo formato das specs anteriores: JSON por linha em `stderr`, campos `ts`, `level`, `event`, `run_id`, mais `dataset_id` e `fingerprint`.

### 5.2 Eventos — `train.train`

| Nível | Evento | Carga |
|---|---|---|
| INFO | `train_started` | `dataset_id`, `algorithm` |
| INFO | `gate_checked` | `dataset_json_path`, `ok: true` |
| INFO | `fingerprint_computed` | `fingerprint`, `reused: bool` |
| INFO | `already_trained` | `fingerprint` — no-op, encerra em 0 |
| INFO | `data_loaded` | `train_rows`, `validation_rows` |
| INFO | `fit_finished` | `model` (`baseline`\|`candidate`), `duration_ms` |
| INFO | `validation_diagnostic` | `mae_candidate_on_validation` (informativo, não decide nada — §3.4) |
| INFO | `sanity_check_passed` | `checks: ["finite_predictions"]` |
| INFO | `train_finished` | `fingerprint`, `staging_path`, `duration_ms` |
| WARNING | `io_retry` | `attempt`, `operation`, `errno` |
| ERROR | `gate_blocked` | `reason` |
| ERROR | `config_invalid` | `reason`, `detail` |
| ERROR | `sanity_check_failed` | `model`, `detail` |
| ERROR | `io_failed` / `internal_error` | `operation`, `errno` / `exc_type` e *traceback* |

### 5.3 Eventos — `evaluate.metrics`

| Nível | Evento | Carga |
|---|---|---|
| INFO | `evaluate_started` | `dataset_id`, `fingerprint` |
| INFO | `staging_loaded` | `baseline_path`, `candidate_path` |
| INFO | `metrics_computed` | `mae_candidate`, `mae_baseline`, `rmse_candidate`, `spearman_candidate`, `test_rows` |
| INFO | `mae_by_persona_computed` | `personas`, `min_mae`, `max_mae` |
| INFO | `mae_by_context_computed` | `with_context_mae`, `no_context_mae`, `no_context_rows` |
| INFO | `latency_measured` | `p50_ms`, `p95_ms`, `sample_size` |
| INFO | `quality_gate_passed` | `candidate_mae`, `required_max_mae` |
| INFO | `evaluate_finished` | `duration_ms`, `eval_json_path` |
| WARNING | `io_retry` | `attempt`, `operation`, `errno` |
| ERROR | `gate_blocked` | `reason` |
| ERROR | `quality_gate_failed` | `candidate_mae`, `baseline_mae`, `required_max_mae` |
| ERROR | `io_failed` / `internal_error` | `operation`, `errno` / `exc_type` e *traceback* |

`quality_gate_failed` é ERROR no sentido de "chame atenção para isto no log", não no sentido de "algo quebrou" — a distinção fica registrada explicitamente aqui para não ser mal-lida por quem grepar logs por nível.

### 5.4 Observabilidade

- *este candidato veio de qual dataset e configuração, exatamente?* → `fingerprint`, reproduzível a partir de `dataset_id` + `configs/pipeline.yaml` (§3.3);
- *o modelo está decorando o vocabulário do split de treino?* → `validation_diagnostic` de `train.train`, comparável ao MAE de teste de `evaluate.metrics` — uma diferença grande entre os dois é sinal de *overfitting*, mesmo sem busca automática de hiperparâmetro;
- *o desempenho varia muito entre personas?* → `mae_by_persona` em `eval.json`, a mesma preocupação de viés que perpassa as specs 01 e 03;
- *o modelo funciona no primeiro turno de uma conversa nova?* → `context_breakdown` em `eval.json` — um `no_context_mae` muito pior que `with_context_mae` é sinal de dependência excessiva do bloco contexto;
- *este modelo é viável para a meta de latência online (spec 08, p95 < 200ms)?* → `latency.p95_ms` em `eval.json`, medido offline, antes de qualquer registro ou deploy.

## 6. Segurança e conformidade

### 6.1 Nenhum texto bruto chega a esta camada

`train.train` e `evaluate.metrics` só leem `text_clean`/`context_clean` de `datasets/*` — já processados por T1+T2 (spec 04) antes de qualquer coisa. O vocabulário que o `TfidfVectorizer` aprende, e que fica serializado dentro de `candidate.joblib`, **não pode conter PII não mascarada**, porque nunca viu PII não mascarada: viu `<CPF>`, `<TEL>`, `<EMAIL>` no lugar. Isso é uma garantia estrutural herdada da spec 04, verificada aqui em profundidade (TN-R16, §7) como defesa contra uma regressão silenciosa em T2.

`eval.json` e `train_manifest.json` carregam só números agregados, hiperparâmetros e IDs — nenhum texto, mascarado ou não.

### 6.2 Base LGPD/GDPR

Mesma base das specs 02–05: corpus sintético, sem titular real. `model.joblib` — mesmo ainda em staging — é o artefato mais fácil de copiar e menos óbvio de auditar visualmente (é binário), por isso a checagem de vocabulário (TN-R16) importa mais aqui do que em qualquer camada anterior: é o primeiro lugar onde um vazamento de PII ficaria **codificado dentro de um artefato binário**, em vez de visível num arquivo de texto ou Parquet inspecionável.

### 6.3 P1

Nenhuma referência a `duckdb` ou `.duckdb` em `train/` ou `evaluate/` — mesma verificação de *lint* das specs anteriores.

## 7. Requisitos verificáveis

**`train.train` — Portão e identidade**
- **TN-R01** QUANDO `data/datasets/<dataset_id>/dataset.json` não existir ou estiver incompleto (JSON ilegível ou sem `row_counts.test.rows`), o script DEVE abortar com *exit* 3, sem treinar.
- **TN-R02** A identidade do treino (caminho de staging) DEVE ser derivada deterministicamente de `dataset_id`, hashes dos três `.parquet`, `algorithm`, hiperparâmetros e `feature_spec_version` — nunca escolhida pelo operador.
- **TN-R03** QUANDO o staging existir completo com o mesmo *fingerprint*, o script DEVE pular o treino (no-op idempotente), *exit* 0.
- **TN-R04** QUANDO o staging existir incompleto, o script DEVE re-treinar e sobrescrever.

**Hiperparâmetros**
- **TN-R05** Todo hiperparâmetro DEVE vir de `configs/pipeline.yaml`; nenhum valor de hiperparâmetro DEVE estar fixo no código.
- **TN-R06** O script DEVE NÃO realizar busca de hiperparâmetros — um único fit por execução.
- **TN-R07** Qualquer componente estocástico do pipeline DEVE receber a `seed` do config; solvers que não aceitam determinismo explícito DEVEM NÃO ser usados.
- **TN-R17** QUANDO `train.algorithm` no config não estiver entre os algoritmos implementados, o script DEVE abortar com *exit* 2, sem ler os `.parquet`.

**Vetorização**
- **TN-R08** O bloco texto DEVE ser `FeatureUnion` de TF-IDF palavra (1-2) e TF-IDF `char_wb` (2-5) sobre `text_clean`.
- **TN-R09** O bloco contexto DEVE juntar `context_clean` em uma string (espaço como separador, vazia se lista vazia) **dentro do `Pipeline` serializado**, e então aplicar TF-IDF palavra (1-2).
- **TN-R10** O vocabulário de ambos os blocos DEVE ser ajustado exclusivamente sobre o split `train`.
- **TN-R11** A combinação dos blocos DEVE usar `ColumnTransformer` com o peso do bloco contexto lido do config.

**Fit e persistência**
- **TN-R12** O script DEVE ajustar `DummyRegressor(strategy="mean")` (baseline) e o `Pipeline` completo (candidato) sobre o mesmo split `train`.
- **TN-R13** O script DEVE persistir os dois modelos via `joblib` e um `train_manifest.json` com hiperparâmetros, `dataset_id`, `algorithm`, `feature_spec_version`, contagens de linhas, *fingerprint*, `code_commit` e timestamps.
- **TN-R14** QUANDO alguma previsão sobre o próprio split `train` não for um float finito, o script DEVE abortar sem persistir, *exit* 4.

**`evaluate.metrics` — Portão**
- **EV-R01** QUANDO o staging de treino não existir completo para a identidade informada, o script DEVE abortar com *exit* 3.

**Métricas**
- **EV-R02** O script DEVE calcular MAE, RMSE e Spearman do candidato e do baseline exclusivamente sobre o split `test`.
- **EV-R03** As previsões usadas em toda métrica DEVEM passar por `clip_score` antes do cálculo.
- **EV-R04** O script DEVE calcular MAE do candidato por `persona` sobre o split `test`.
- **EV-R05** O script DEVE calcular MAE do candidato separado por presença de contexto: exemplos com `context_clean = []` e exemplos com `context_clean` não vazio (§3.2a).
- **EV-R06** O script DEVE medir a latência de predição do candidato item a item (nunca em lote) sobre o split `test` e reportar p50/p95 (§3.2a).
- **EV-R07** O script DEVE NÃO calcular métricas de classificação (acurácia, macro-F1) — fora de escopo (ADR-0001, T6 inexistente).

**Quality gate**
- **EV-R08** QUANDO `mae_candidato > threshold_ratio × mae_baseline`, o script DEVE gravar `eval.json` com `gate.passed = false` e encerrar com *exit* 5.
- **EV-R09** QUANDO o *gate* passar, o script DEVE gravar `eval.json` com `gate.passed = true` e encerrar com *exit* 0.
- **EV-R10** Duas execuções sobre o mesmo staging e o mesmo split `test` DEVEM produzir o mesmo `eval.json`, exceto os campos de tempo e de latência medida (§3.2a: a latência em si não é determinística, só a métrica que ela reporta é recalculada a cada execução).

**Segurança**
- **TN-R15 / EV-R11** Nenhum dos dois módulos DEVE importar `duckdb` nem referenciar caminho `.duckdb` (P1).
- **TN-R16** O vocabulário aprendido pelos `TfidfVectorizer` DEVE NÃO conter token que pareça CPF/telefone/e-mail não mascarado — verificado por teste.
- **EV-R12** `eval.json` DEVE conter apenas métricas agregadas, hiperparâmetros e IDs — nenhum `text_clean`/`context_clean`.

### 7.1 Catálogo de erros

| Código | Requisito | Script |
|---|---|---|
| `TN_R01_GATE_DATASET_NOT_OK` | TN-R01 | train |
| `TN_R14_NON_FINITE_PREDICTION` | TN-R14 | train |
| `TN_R17_UNSUPPORTED_ALGORITHM` | TN-R17 | train |
| `TN_IO_FAILED` | — (I/O) | train |
| `EV_R01_GATE_STAGING_NOT_OK` | EV-R01 | evaluate |
| `EV_R08_QUALITY_GATE_FAILED` | EV-R08 | evaluate |
| `EV_IO_FAILED` | — (I/O) | evaluate |
| `INTERNAL_ERROR` | — | qualquer |

## 8. Decisões deste documento

| # | Decisão | Motivo | Consequência assumida |
|---|---|---|---|
| D1 | **Escopo:** treino e avaliação num só documento | Replica a numeração de componente já usada no projeto (`06-train-evaluate`); o *quality gate* não é interpretável isolado do fit que o precede | Documento mais longo que as specs 02–05; mitigado por seções claramente separadas por script |
| D2 | **`models/_staging/`** — diretório novo, não previsto na `data-model.md` | `train.train` e `evaluate.metrics` são subcomandos independentes (spec 09); o candidato precisa de um lugar para existir entre o fit e o registro | A `data-model.md §4` deveria ganhar uma nota sobre este diretório — pendência a levar para fora do `mood-ml` (mesmo padrão da spec 03 §10 com `persona` em `datasets/*`) |
| D3 | **Identidade do treino é conteúdo-endereçada** (*fingerprint*), não escolhida pelo operador | Elimina a classe inteira de conflito que `corpus_id`/`dataset_id` precisaram resolver com *exit* 4 — aqui, entradas diferentes não podem colidir por construção | Um `fingerprint` não é memorável; o operador navega por `dataset_id` + config, não por um nome à mão |
| D4 | **Sem busca de hiperparâmetros** | Volume do MVP e prazo de TCC não justificam a complexidade; `validation` cumpre papel diagnóstico, não decisório | Ajuste de hiperparâmetro é manual (editar YAML, rerodar); registrado como evolução possível, não implementado |
| D5 | **`clip_score` mora em `evaluate/metrics.py`**, importada por `infer/predict.py` | Evita uma classe customizada dentro do `Pipeline` serializado (risco de quebra entre versões do scikit-learn); função pura de uma linha é mais simples e igualmente segura para P4 | `evaluate/` passa a ter uma dependência de import por `infer/` (spec 08) — documentado para não surpreender quando a spec 08 for escrita |

## 9. Critérios de aceite

- **CA-01** Dataset aprovado, staging inédito → `train.train` *exit* 0, `baseline.joblib`, `candidate.joblib`, `train_manifest.json` gravados com o esquema da §3.2.
- **CA-02** Reexecução de `train.train` sobre o mesmo `dataset_id` e config → *exit* 0, `already_trained`, nenhum arquivo reescrito (verificado por `mtime`).
- **CA-03** `train.train` com hiperparâmetro alterado no config → *fingerprint* diferente, staging novo, staging anterior preservado intacto.
- **CA-04** `evaluate.metrics` sobre um staging completo e um candidato competente (MAE bem abaixo do baseline num *dataset* de teste sintético) → *exit* 0, `gate.passed: true`.
- **CA-05** `evaluate.metrics` sobre um candidato deliberadamente ruim (ex.: `Ridge(alpha)` absurdo no config, ou um teste que injeta um modelo degenerado) → *exit* 5, `eval.json` gravado com `gate.passed: false`.
- **CA-06** Dataset sem `dataset.json` íntegro → `train.train` *exit* 3, nada gravado em `models/_staging/`.
- **CA-07** Staging incompleto/ausente → `evaluate.metrics` *exit* 3, nenhum `eval.json` gravado.
- **CA-08** Teste de paridade: `clip_score` aplicado a um valor fora de `[-1, 1]` (ex.: `1.4`, `-1.2`) devolve `1.0`/`-1.0`; dentro do intervalo, devolve o valor inalterado.
- **CA-09** Teste de vazamento de vocabulário: um `dataset_id` de teste com PII deliberadamente não mascarada num exemplo (para simular uma regressão em T2) faz o teste de TN-R16 falhar — prova de que o teste realmente detecta o problema que deveria detectar.
- **CA-10** `mae_by_persona` em `eval.json` tem uma chave para cada `persona` presente no split `test`, e a soma ponderada aproxima o `mae` geral do candidato.
- **CA-11** `context_breakdown.with_context_rows + context_breakdown.no_context_rows == test_rows`; num `dataset_id` de teste onde toda conversa tem só uma mensagem `customer` (todos os exemplos sem contexto), `with_context_rows == 0` e o cálculo não levanta divisão por zero.
- **CA-12** `latency.p95_ms` é medido item a item (não em lote): um teste substitui `Pipeline.predict` por uma versão artificialmente lenta e confirma que a medição reflete o atraso por chamada, não um tempo agregado de lote.
- **CA-13** Teste de P1: nenhuma ocorrência de `duckdb`/`.duckdb` em `train/` ou `evaluate/`.

## 10. Fora de escopo

- **Registro e promoção do modelo** (`registry/registry.py`, `model_version`, `active.json`) — spec 07. Este documento produz o insumo (`eval.json` aprovado), não o consome.
- **Abordagem C (embeddings congelados)** — ADR-0008, "implementada depois". `algorithm: "embeddings-ridge"` é um valor reservado no config, não implementado.
- **Busca automática de hiperparâmetros** (`GridSearchCV` ou equivalente) — decisão explícita (D4). Pode ser revisitada se o projeto crescer além do escopo de TCC.
- **T6 (score → `mood_label`) e métricas de classificação** — fora de escopo por ADR-0001 (`mood_label = null`, sem T6). Métricas são só de regressão (MAE, RMSE, Spearman).
- **Agendamento de retreino e alerta de regressão de qualidade** — mencionado em §4.4 como ponto a revisitar, não implementado no MVP.
- **Orquestração, CI e alvos de Makefile.** Spec 09.

## 11. Checklist de implementação

- [ ] `train.train` — CLI, F1 portão sobre `dataset.json` (TN-R01)
- [ ] F2 — cálculo do *fingerprint* e decisão de reaproveitar/re-treinar (TN-R02 a TN-R04)
- [ ] F3 — carga de `train.parquet`/`validation.parquet`
- [ ] F4 — `ColumnTransformer` com `FunctionTransformer` de junção de contexto embutido; `Ridge` com solver determinístico; `DummyRegressor` (TN-R08 a TN-R12)
- [ ] F5 — sanity check de previsões finitas (TN-R14)
- [ ] F6 — escrita atômica de `baseline.joblib`, `candidate.joblib`, `train_manifest.json`, nesta ordem (TN-R13)
- [ ] `evaluate.metrics` — CLI, F1 portão sobre o staging (EV-R01)
- [ ] `clip_score` em `evaluate/metrics.py` (§3.5, D5)
- [ ] F3 — MAE/RMSE/Spearman de candidato e baseline sobre `test`; MAE por `persona` (EV-R02 a EV-R04)
- [ ] F3 — MAE por presença de contexto e medição de latência p95 item a item (EV-R05, EV-R06, §3.2a)
- [ ] F4 — *quality gate* e decisão de *exit code* (EV-R08, EV-R09)
- [ ] F5 — escrita atômica de `eval.json` (EV-R10)
- [ ] Teste de vazamento de vocabulário (TN-R16, CA-09)
- [ ] Retentativa de I/O com backoff 1s/2s/4s nos dois scripts
- [ ] Log estruturado JSON por evento nos dois scripts (§5)
- [ ] Fixtures: dataset aprovado, dataset não aprovado, staging completo/incompleto, candidato bom/ruim para CA-04/CA-05
- [ ] Testes `tests/unit/` e `tests/contract/` cobrindo CA-01 a CA-11

# 07 — Registro de Versão e Promoção (camada `registry/`)

**Status:** Proposta · **Versão da spec:** `rg-1` · **Data:** 2026-09-20
**Implementa:** [data-model.md §4.4](../../data-structure/data-model.md) (`models/<model_version>/manifest.json`) e fecha a decisão "em aberto" de `models/active.json` (`data-model.md §9.2`).
**Depende de:** [ADR-0001](../../decisions/ADR-0001-escala-do-humor.md) (`mood_labels = null`, sem T6), [ADR-0007](../../decisions/ADR-0007-humor-por-conversa.md) (`history_window = 30`, `history_scope = "conversation"`), [ADR-0008](../../decisions/ADR-0008-abordagens-de-modelo.md) (`algorithm`); [constitution.md](../../decisions/constitution.md) P3, P4, P5; [06-train-evaluate.md](06-train-evaluate.md) (staging e `eval.json` de origem)
**Consumido por:** `infer/predict.py` (spec 08) — lê `active.json` na subida do serviço, depois `models/<model_version>/manifest.json` e `model.joblib`
**Implementado em:** [registry/registry.py](../registry/registry.py)

---

## 1. Objetivo geral

Dois comandos, duas responsabilidades deliberadamente separadas — a mesma separação já fixada como decisão de projeto: *"promover para `active.json` é um comando explícito, nunca automático"*:

1. **`registry.register`** — pega um candidato que **já passou** pelo *quality gate* da spec 06 (`eval.json.gate.passed == true`) e o transforma num artefato permanente: atribui um `model_version` que nunca será reutilizado, monta o `manifest.json` completo (`data-model.md §4.4`), copia o modelo para `models/<model_version>/model.joblib`. Depois disso, essa versão existe para sempre — registrada não é o mesmo que ativa.
2. **`registry.promote`** — muda qual `model_version` o serviço de inferência deve carregar, sobrescrevendo `models/active.json`. É a única ação desta camada com efeito imediato em produção (o próximo `restart` do `mood-api`/`mood-ml` de inferência passa a servir outra versão), e por isso é a única que **nunca** acontece como consequência automática de `register`.

O que esta spec **não decide** é *qual* candidato merece ser registrado — isso já foi decidido pela spec 06 (o *quality gate*). Esta camada decide **como um candidato aprovado vira um artefato citável e imutável**, e **como um operador troca deliberadamente qual versão está no ar**.

## 2. Arquitetura e fluxo de dados

### 2.1 Posição no fluxo

```
models/_staging/<dataset_id>/<algorithm>/<fingerprint>/
    baseline.joblib  candidate.joblib  train_manifest.json  eval.json   (gate.passed: true, spec 06)
       │
       ▼
registry.register  ──►  models/<model_version>/
                             model.joblib        (cópia de candidate.joblib)
                             manifest.json        (data-model §4.4 + training_fingerprint)
       │
       │  (mais tarde, decisão humana separada)
       ▼
registry.promote  ──►  models/active.json        (ponteiro mutável: {"model_version", "promoted_at"})
       │
       ▼
infer/predict.py (spec 08) — lê active.json na subida do serviço
```

### 2.2 Passo a passo lógico — `registry.register`

| Fase | Nome | O que faz |
|---|---|---|
| F0 | Resolução da entrada | Recebe `dataset_id` + *fingerprint* (identidade do staging, spec 06) |
| F1 | Portão | Confirma que `eval.json` existe e tem `gate.passed == true` |
| F2 | Busca de idempotência | Varre `models/mood-*/manifest.json` procurando um `training_fingerprint` igual ao deste candidato |
| F3a | (achou) | Loga `already_registered` com o `model_version` encontrado; encerra em 0, sem escrever nada |
| F3b | (não achou) | Minta um `model_version` novo (§3.3) |
| F4 | Montagem do manifesto | Lê `eval.json`, `train_manifest.json` e `dataset.json` (spec 05) para preencher os campos de `data-model.md §4.4` |
| F5 | Emissão | Copia `candidate.joblib` → `model.joblib`; grava `manifest.json`; loga o resumo; encerra |

### 2.3 Passo a passo lógico — `registry.promote`

| Fase | Nome | O que faz |
|---|---|---|
| F0 | Resolução da entrada | Recebe `model_version` alvo (obrigatório, explícito) |
| F1 | Portão | Confirma que `models/<model_version>/manifest.json` existe e é válido |
| F2 | Já ativo? | Se `active.json.model_version` já for o alvo, encerra em 0 sem reescrever (idempotência) |
| F3 | Comparação de métricas | Se existir `active.json`, compara `metrics.mae` do alvo com o da versão atualmente ativa (§3.3) |
| F4a | (piorou, sem `--force`) | Emite *warning*, aborta sem alterar `active.json` |
| F4b | (não piorou, ou `--force`) | Prossegue |
| F5 | Emissão | Grava `active.json` = `{"model_version": ..., "promoted_at": ...}`, atômico; loga; encerra |

### 2.4 Por que `register` nunca falha "por causa do dado"

Ao contrário das specs 02–06, esta camada não tem um portão que rejeita conteúdo malformado vindo de fora: o `candidate.joblib` e o `eval.json` já passaram por toda a cadeia de verificação das specs anteriores. O único jeito de `register` falhar por uma causa que não seja bug ou ambiente é o candidato **não ter passado no *quality gate*** — e isso não é "dado ruim", é uma informação legítima (§4.1).

## 3. Requisitos técnicos e premissas

### 3.1 Conexões e autenticação

**Não há.** Mesma razão de todas as camadas anteriores — arquivo local. `registry.register` depende do staging de um treino específico (spec 06); `registry.promote` depende apenas de `models/<model_version>/` já registrados.

```bash
python -m registry.register <dataset_id> <fingerprint> [--models-dir models]
python -m registry.promote <model_version> [--force] [--models-dir models]
```

Runtime: Python 3.12, sem dependência nova (usa `joblib`, `json`, `shutil` da própria stdlib/`requirements.txt` já existentes).

Invocação pelo orquestrador (spec 09): `make register DATASET=<dataset_id> FINGERPRINT=<fp>`, `make promote VERSION=<model_version>`, e `python -m pipeline register|promote --config configs/pipeline.yaml`.

### 3.2 Formato dos dados

| | Entrada (`register`) | Saída (`register`) | Entrada (`promote`) | Saída (`promote`) |
|---|---|---|---|---|
| Formato | JSON (`eval.json`, `train_manifest.json`, `dataset.json`) + joblib | JSON + joblib | JSON (`manifest.json`, `active.json`) | JSON |
| Caminhos | `models/_staging/.../{eval,train_manifest}.json`, `data/datasets/<dataset_id>/dataset.json` | `models/<model_version>/{manifest.json,model.joblib}` | `models/<model_version>/manifest.json`, `models/active.json` | `models/active.json` |

**`manifest.json`** — exatamente o esquema de `data-model.md §4.4`, mais um campo que este documento introduz:

```json
{
  "model_version": "mood-2026.09.0",
  "scale": "-1 to 1",
  "mood_labels": null,
  "trained_at": "2026-09-20T15:02:11Z",
  "dataset_id": "ds-2026-09-20-a",
  "label_set_id": "ls-syn-2026-09-18-a",
  "feature_spec_version": "fs-1",
  "history_window": 30,
  "history_scope": "conversation",
  "algorithm": "tfidf-ridge",
  "metrics": { "mae": 0.171, "rmse": 0.229, "spearman": 0.812 },
  "code_commit": "<git sha>",
  "training_fingerprint": "b7a1...9f0c"
}
```

`training_fingerprint` (o mesmo *fingerprint* da spec 06, §3.3 daquele documento) **não está** na `data-model.md §4.4` original — é o que torna a busca de idempotência da §3.3 abaixo possível sem um índice separado. Fica registrado como extensão pendente de levar para a `data-model.md`, no mesmo padrão já usado pela spec 03 (`persona` em `datasets/*`) e pela spec 06 (`models/_staging/`).

**Correção em relação a documentação anterior do projeto:** `history_window` é **30**, `history_scope` é **`"conversation"`** — não `1`. Uma nota de resumo anterior (*"history_window = 1. Usa só o último item de history"*) descrevia a decisão original da ADR-0003, substituída pela ADR-0007 em 2026-09-18. Como ADRs têm precedência sobre resumos de projeto, o manifesto segue a ADR-0007, e este documento fecha essa divergência para a camada de registro.

**`active.json`** — esquema mínimo, decidido por este documento (a `data-model.md` deixava em aberto):

```json
{ "model_version": "mood-2026.09.0", "promoted_at": "2026-09-20T16:00:00Z" }
```

Deliberadamente **não** duplica `algorithm`, `metrics` ou qualquer outro campo do `manifest.json` — quem consome `active.json` (`infer/predict.py`, spec 08) usa o `model_version` só para localizar e carregar o `manifest.json` correspondente. Duas cópias da mesma informação divergiriam com o tempo; `active.json` é um ponteiro, não um resumo.

### 3.3 Estratégia de carga e idempotência

**`register` é idempotente por *fingerprint*, sem índice separado.** Em vez de manter um arquivo à parte mapeando *fingerprint* → `model_version` (mais um artefato para manter sincronizado com a verdade), o script varre os `manifest.json` já existentes em `models/mood-*/` procurando um `training_fingerprint` igual ao do candidato atual. Nos volumes desta camada (§3.4 — dezenas de versões, não milhares), uma varredura é mais barata de manter correta do que um índice que poderia divergir da fonte.

| Situação | Comportamento | Exit |
|---|---|---|
| Nenhum `manifest.json` com este *fingerprint* | Minta `model_version` novo, registra | 0 |
| Já existe um `manifest.json` com este *fingerprint* | **No-op idempotente** — devolve o `model_version` existente, não cria outro | 0 |
| `eval.json` não existe ou está incompleto | Aborta, não registra | 3 |
| `eval.json` existe, `gate.passed == false` | Aborta, não registra — candidato existe mas foi reprovado | 6 |

**Numeração de `model_version` (`mood-AAAA.MM.N`).** `N` é sequencial dentro do mês corrente: o script lê os `model_version` existentes com o prefixo `mood-<ano>.<mês>.`, toma o maior `N` encontrado, usa `N + 1` (ou `0` se nenhum existir naquele mês). Não há mecanismo de concorrência — o MVP assume um único operador rodando `register` de cada vez, e isso é declarado como premissa (§10), não implementado como trava.

**`promote` é idempotente por comparação direta.** Se `active.json.model_version` já é o alvo, é um no-op — sem *fingerprint* aqui, porque a identidade é simplesmente "qual `model_version` está apontado", e essa pergunta já tem resposta direta sem precisar de hash.

**`active.json` é um ponteiro mutável, não um artefato versionado.** Ao contrário de tudo que as specs 02–07 produziram até aqui, `active.json` **é sobrescrito a cada promoção**, sem manter histórico das promoções anteriores — decisão explícita do usuário. Isso não compromete a rastreabilidade do que foi treinado: cada `model_version` continua imutável e seu `manifest.json` nunca muda; o que se perde é só a resposta a "qual versão estava ativa antes desta, e quando trocou" depois do fato. Como `mood-ml/models/` está fora do git (P5), essa pergunta não tem uma segunda fonte de verdade a consultar — é uma limitação assumida, registrada explicitamente aqui (§8) para não ser descoberta por acidente depois.

### 3.4 Volumetria estimada e frequência de execução

A camada de menor volume do pipeline inteiro — não processa mensagens, processa **decisões de experimento**:

| Grandeza | Ordem de grandeza esperada (vida útil de um TCC) |
|---|---|
| Chamadas de `register` | Dezenas — uma por candidato que passou no *quality gate* |
| `model_version` existentes simultaneamente | Poucas dezenas no máximo |
| Chamadas de `promote` | Poucas unidades — só quando o operador decide trocar a versão ativa |
| Tamanho de `model.joblib` (abordagem A) | Pequeno — vocabulário TF-IDF + coeficientes do Ridge, tipicamente < 10 MB no volume da spec 05 |

Orçamento de desempenho: a varredura de idempotência (§3.3) sobre dezenas de `manifest.json` é da ordem de milissegundos. A cópia de `model.joblib` é limitada pelo tamanho do arquivo, não pela lógica do script.

**Frequência.** `register`: sob demanda, imediatamente depois de um `evaluate.metrics` com *gate* aprovado — parte natural de `make all`. `promote`: **nunca** automático (decisão de projeto), disparado manualmente pelo operador quando decide, por avaliação própria, que uma versão deve substituir a ativa. Não há cadência a definir para `promote` além de "quando alguém decidir".

## 4. Tratamento de erros e resiliência

### 4.1 Taxonomia

- **Erro de portão** — staging incompleto (`register`) ou `model_version` inexistente (`promote`). *Exit* 3 nos dois.
- **Candidato reprovado** (`register`) — não é erro, é informação: o *quality gate* já disse não. *Exit* 6.
- **Regressão de métrica sem confirmação** (`promote`) — também não é erro técnico, é uma barreira de segurança que espera confirmação explícita. *Exit* 7.
- **Erro de uso** — CLI mal chamada. *Exit* 2.
- **Erro de ambiente** — I/O (cópia de arquivo, escrita de JSON). *Exit* 1.

### 4.2 Política de retentativa

**Sobre a lógica de registro/promoção: nenhuma.** Determinística — a varredura de *fingerprint*, a numeração de `model_version` e a comparação de métricas dão o mesmo resultado toda vez, para a mesma entrada.

**Sobre I/O: 3 tentativas, backoff 1s/2s/4s.** Mesmo padrão de todas as specs anteriores, aplicado à leitura de `eval.json`/`train_manifest.json`/`dataset.json`/`manifest.json`, à cópia de `candidate.joblib` → `model.joblib`, e à escrita de `manifest.json`/`active.json`.

**Gravação atômica, em ordem — `register`.** `model.joblib` primeiro (cópia via arquivo temporário + `os.replace()`), `manifest.json` por último. `manifest.json` é o marcador de sucesso: sua presença implica o modelo copiado corretamente. Se a cópia falhar e for retentada, nada em `models/<model_version>/` fica pela metade visível a quem procura por `manifest.json`.

**Gravação atômica — `promote`.** Só `active.json`, via `.tmp` + `os.replace()`. Não há ordem a definir porque só há um arquivo.

### 4.3 *Exit codes*

**`registry.register`**

| Exit | Significado |
|---|---|
| 0 | Registrado (ou já registrado, mesmo *fingerprint*) |
| 2 | Erro de uso da CLI |
| 3 | Portão bloqueado — staging sem `eval.json` completo |
| 6 | Candidato existe, mas `gate.passed == false` — não registrável |
| 1 | Erro interno ou de ambiente |

**`registry.promote`**

| Exit | Significado |
|---|---|
| 0 | Promovido (ou já era a versão ativa) |
| 2 | Erro de uso da CLI |
| 3 | Portão bloqueado — `model_version` alvo sem `manifest.json` válido |
| 7 | Regressão de métrica sem `--force` — promoção bloqueada, aguardando confirmação explícita |
| 1 | Erro interno ou de ambiente |

### 4.4 Alertas e notificações

**Nenhum canal externo**, mesma razão de todas as specs anteriores — MVP offline, sob demanda, sem plantão. A única nuance nova: `promote` **é** a ação desta pipeline inteira com efeito mais imediato em produção (o próximo *restart* do serviço de inferência passa a servir outra versão), o que a tornaria a candidata mais natural a um alerta de verdade ("versão X foi promovida às HH:MM por fulano") num ambiente de produção real. Fica registrado como ponto a revisitar — mesma condição das specs 02–06 (migração para execução não supervisionada ou múltiplos operadores).

## 5. Logs e observabilidade

### 5.1 Padrão

Mesmo formato de todas as specs anteriores: JSON por linha em `stderr`, campos `ts`, `level`, `event`, `run_id`.

### 5.2 Eventos — `registry.register`

| Nível | Evento | Carga |
|---|---|---|
| INFO | `register_started` | `dataset_id`, `fingerprint` |
| INFO | `gate_checked` | `eval_json_path`, `passed: true` |
| INFO | `fingerprint_scan_completed` | `manifests_scanned`, `match_found: bool` |
| INFO | `already_registered` | `model_version` — no-op, encerra em 0 |
| INFO | `model_version_minted` | `model_version`, `month_sequence` |
| INFO | `manifest_written` | `model_version`, `algorithm`, `metrics` |
| INFO | `register_finished` | `model_version`, `duration_ms` |
| WARNING | `io_retry` | `attempt`, `operation`, `errno` |
| ERROR | `gate_blocked` | `reason` (`staging_incomplete`) |
| ERROR | `candidate_rejected` | `fingerprint`, `candidate_mae`, `required_max_mae` |
| ERROR | `io_failed` / `internal_error` | `operation`, `errno` / `exc_type` e *traceback* |

### 5.3 Eventos — `registry.promote`

| Nível | Evento | Carga |
|---|---|---|
| INFO | `promote_started` | `target_model_version` |
| INFO | `already_active` | `model_version` — no-op, encerra em 0 |
| INFO | `metric_comparison` | `target_mae`, `active_mae`, `regression: bool` |
| INFO | `active_json_written` | `previous_model_version`, `new_model_version`, `promoted_at` |
| INFO | `promote_finished` | `model_version`, `duration_ms` |
| WARNING | `io_retry` | `attempt`, `operation`, `errno` |
| WARNING | `promotion_blocked_regression` | `target_mae`, `active_mae` — sem `--force` |
| ERROR | `gate_blocked` | `reason` (`model_version_not_found`) |
| ERROR | `io_failed` / `internal_error` | `operation`, `errno` / `exc_type` e *traceback* |

`promotion_blocked_regression` é WARNING, não ERROR — o script não quebrou, só recusou agir sem confirmação, exatamente como pedido (§3.3, decisão do usuário).

### 5.4 Observabilidade

- *este `model_version` veio de qual candidato, exatamente?* → `training_fingerprint` no `manifest.json`, o mesmo que aparece em `train_manifest.json` e `eval.json` da spec 06 — rastreável ponta a ponta;
- *qual versão está no ar agora?* → `models/active.json.model_version`;
- *essa promoção piorou alguma coisa?* → `metric_comparison` no log de `promote`, único lugar onde essa comparação fica registrada (não persiste em arquivo, §3.3).

## 6. Segurança e conformidade

### 6.1 Nada de novo entra nesta camada

`registry.register` copia um `.joblib` já verificado (spec 06, TN-R16 — vocabulário sem PII) e lê apenas JSONs de metadados (`eval.json`, `train_manifest.json`, `dataset.json`) que, por design das specs 05 e 06, nunca carregam `text`. `manifest.json` e `active.json` resultantes contêm só IDs, números e timestamps — a mesma garantia estrutural que vem se repetindo desde a spec 03: um campo que a camada nunca lê para gravação não pode vazar por ela.

### 6.2 Base LGPD/GDPR

Mesma base de todas as specs anteriores: corpus sintético, sem titular real. Esta camada em particular não introduz nenhuma superfície nova de PII — só rearranja identificadores e metadados que já passaram por todas as verificações anteriores.

### 6.3 P1, e P2 por analogia

Nenhuma referência a `duckdb`/`.duckdb` em `registry/` — mesma verificação de *lint*. A imutabilidade de `model_version` (uma vez criado, seu `manifest.json` nunca muda) é o mesmo princípio de P2 (*append-only*) aplicado a um artefato de arquivo: o histórico de versões treinadas nunca é reescrito, só `active.json` — o ponteiro de estado atual — é mutável, exatamente como `v_customer_mood_latest` deriva do histórico *append-only* de `mood_scores` no banco operacional.

## 7. Requisitos verificáveis

**`register` — Portão**
- **RG-R01** QUANDO o staging identificado não tiver `eval.json` completo, o script DEVE abortar com *exit* 3, sem registrar.
- **RG-R02** QUANDO `eval.json` existir com `gate.passed == false`, o script DEVE abortar com *exit* 6, sem registrar.

**Idempotência e identidade**
- **RG-R03** O script DEVE varrer `models/mood-*/manifest.json` procurando `training_fingerprint` igual ao do candidato, antes de mintar qualquer `model_version` novo.
- **RG-R04** QUANDO encontrar correspondência, o script DEVE devolver o `model_version` existente sem criar outro, *exit* 0.
- **RG-R05** QUANDO não encontrar, o script DEVE mintar `model_version` no formato `mood-AAAA.MM.N`, com `N` = maior `N` existente no mês corrente `+ 1`, ou `0` se nenhum existir.
- **RG-R06** Um `model_version` já mintado DEVE NUNCA ser reatribuído a um *fingerprint* diferente do que o originou.

**Montagem do manifesto**
- **RG-R07** `manifest.json` DEVE conter exatamente os campos de `data-model.md §4.4` mais `training_fingerprint`.
- **RG-R08** `history_window` DEVE ser `30` e `history_scope` DEVE ser `"conversation"` (ADR-0007) — nunca `1` ou outro valor.
- **RG-R09** `mood_labels` DEVE ser `null` (ADR-0001).
- **RG-R10** `metrics` DEVE vir de `eval.json.metrics.candidate`, nunca recalculado por esta camada.
- **RG-R11** `label_set_id` e `feature_spec_version` DEVEM vir de `dataset.json` do `dataset_id` correspondente.
- **RG-R12** `code_commit` DEVE vir de `train_manifest.json`.

**Persistência**
- **RG-R13** O script DEVE copiar `candidate.joblib` para `model.joblib` sem alterar o arquivo de origem em staging.
- **RG-R14** `manifest.json` DEVE ser gravado atomicamente, depois da cópia do modelo.
- **RG-R15** `models/<model_version>/` DEVE NUNCA ser sobrescrito depois de criado.

**`promote` — Portão**
- **PM-R01** QUANDO o `model_version` alvo não tiver `manifest.json` válido, o script DEVE abortar com *exit* 3.

**Comparação de métricas**
- **PM-R02** QUANDO existir `active.json`, o script DEVE comparar `metrics.mae` do alvo com o da versão atualmente ativa antes de promover.
- **PM-R03** QUANDO o alvo tiver MAE pior (maior) que o ativo e `--force` não for passado, o script DEVE abortar com *exit* 7, sem alterar `active.json`.
- **PM-R04** QUANDO `--force` for passado, ou o alvo não for pior, o script DEVE prosseguir com a promoção.
- **PM-R05** QUANDO não existir `active.json` (primeira promoção), o script DEVE prosseguir sem comparação.

**Idempotência**
- **PM-R06** QUANDO o `model_version` alvo já for o ativo, o script DEVE encerrar em 0 sem reescrever `active.json`.

**Persistência**
- **PM-R07** `active.json` DEVE conter no mínimo `model_version` e `promoted_at`.
- **PM-R08** `active.json` DEVE ser sobrescrito atomicamente a cada promoção e DEVE NÃO manter histórico de promoções anteriores.

**Segurança**
- **RG-R16 / PM-R09** Nenhum dos dois scripts DEVE importar `duckdb` nem referenciar caminho `.duckdb` (P1).
- **RG-R17** `manifest.json` e `active.json` DEVEM conter apenas IDs, métricas e timestamps — nenhum `text`.

### 7.1 Catálogo de erros

| Código | Requisito | Script |
|---|---|---|
| `RG_R01_GATE_STAGING_INCOMPLETE` | RG-R01 | register |
| `RG_R02_CANDIDATE_REJECTED` | RG-R02 | register |
| `RG_R15_IO_FAILED` | — (I/O) | register |
| `PM_R01_GATE_MODEL_NOT_FOUND` | PM-R01 | promote |
| `PM_R03_METRIC_REGRESSION_BLOCKED` | PM-R03 | promote |
| `PM_R09_IO_FAILED` | — (I/O) | promote |
| `INTERNAL_ERROR` | — | qualquer |

## 8. Decisões deste documento

| # | Decisão | Motivo | Consequência assumida |
|---|---|---|---|
| D1 | **Idempotência de `register` por varredura**, sem índice separado | Volume desta camada (dezenas de versões) torna uma varredura mais barata de manter correta do que um índice que pode divergir da verdade | Custo de varredura cresce linearmente com o número de `model_version` — irrelevante na escala do MVP, a revisitar se a camada crescer muito |
| D2 | **`active.json` sem histórico de promoções** | Decisão explícita do usuário: simplicidade sobre auditabilidade de longo prazo | Sem git rastreando `models/` (P5) e sem log, não há como reconstruir "o que estava ativo antes" depois do fato — limitação assumida, não uma lacuna esquecida |
| D3 | **`promote` avisa mas não bloqueia regressão de métrica**, exceto sem `--force` | Promoção é deliberação humana por design; um bloqueio automático converteria uma decisão de produto em um *quality gate* técnico, papel que já pertence à spec 06 | Um operador apressado pode usar `--force` para promover uma regressão deliberadamente — comportamento permitido de propósito, não uma falha de proteção |
| D4 | **`training_fingerprint` estende `manifest.json`** além do esquema original de `data-model.md §4.4` | É o que viabiliza D1 sem um índice à parte | `data-model.md` precisa ser atualizada para refletir o campo novo — pendência a levar para fora do `mood-ml`, mesmo padrão da spec 03 (`persona`) e da spec 06 (`models/_staging/`) |
| D5 | **`code_commit` vem de `train_manifest.json`**, não recapturado por `registry.register` | O que importa para reprodutibilidade é o commit que **treinou** o modelo, não o commit que o registrou — podem divergir se `registry.py` for atualizado depois do treino | Nenhuma — é a leitura correta de proveniência |

## 9. Critérios de aceite

- **CA-01** Staging com `eval.json.gate.passed == true`, *fingerprint* inédito → `register` *exit* 0, `models/<model_version>/{manifest.json,model.joblib}` gravados com o esquema da §3.2.
- **CA-02** Reexecução de `register` sobre o mesmo staging → *exit* 0, `already_registered`, nenhum `model_version` novo criado, nenhum arquivo reescrito.
- **CA-03** Staging com `eval.json.gate.passed == false` → `register` *exit* 6, nenhum `model_version` criado.
- **CA-04** Staging incompleto/inexistente → `register` *exit* 3, nenhum `model_version` criado.
- **CA-05** Dois candidatos de datasets/configurações diferentes, ambos aprovados → dois `model_version` distintos, `N` sequencial dentro do mesmo mês (`mood-2026.09.0`, `mood-2026.09.1`).
- **CA-06** `manifest.json` gerado tem `history_window == 30` e `history_scope == "conversation"` — nunca `1`.
- **CA-07** `promote` para um `model_version` existente sem `active.json` prévio → *exit* 0, `active.json` criado.
- **CA-08** `promote` para um `model_version` com MAE pior que o ativo, sem `--force` → *exit* 7, `active.json` inalterado.
- **CA-09** Mesmo cenário de CA-08, com `--force` → *exit* 0, `active.json` atualizado.
- **CA-10** `promote` repetido para o mesmo `model_version` já ativo → *exit* 0, `active.json` com `mtime` inalterado.
- **CA-11** `promote` para um `model_version` inexistente → *exit* 3, `active.json` inalterado.
- **CA-12** Teste de P1: nenhuma ocorrência de `duckdb`/`.duckdb` em `registry/`.
- **CA-13** Teste de P5: busca literal por qualquer `text` de mensagem em `manifest.json` e `active.json` não encontra ocorrência.

## 10. Fora de escopo

- **Carregamento do modelo pelo serviço de inferência** (leitura de `active.json`, verificação de `feature_spec_version`/`history_window`/`history_scope` no carregamento, resposta `503` se ausente) — spec 08.
- **Controle de concorrência entre chamadas simultâneas de `register`** — premissa assumida de operador único (§3.3); revisitar se o projeto ganhar múltiplos operadores ou automação concorrente.
- **Log/histórico de promoções** — decisão explícita do usuário (D2); pode ser adicionado depois sem quebrar o esquema atual de `active.json` (é aditivo).
- **Reversão automática de promoção** ("voltar para a versão anterior") — não implementada; sem histórico (D2), reverter exige o operador saber de cor qual era o `model_version` anterior.
- **Abordagem C (embeddings)** — o `algorithm` já é um campo do manifesto (ADR-0008); nenhuma lógica específica de C é tratada aqui.
- **Orquestração, CI e alvos de Makefile.** Spec 09.

## 11. Checklist de implementação

- [ ] `registry.register` — CLI, F1 portão sobre `eval.json` (RG-R01, RG-R02)
- [ ] F2 — varredura de `training_fingerprint` em `models/mood-*/manifest.json` (RG-R03, RG-R04)
- [ ] F3b — numeração sequencial de `model_version` por mês (RG-R05, RG-R06)
- [ ] F4 — montagem do `manifest.json` a partir de `eval.json`, `train_manifest.json`, `dataset.json` (RG-R07 a RG-R12)
- [ ] F5 — cópia atômica de `model.joblib` e escrita de `manifest.json`, nesta ordem (RG-R13 a RG-R15)
- [ ] `registry.promote` — CLI, F1 portão sobre `manifest.json` do alvo (PM-R01)
- [ ] F2 — checagem de já-ativo (PM-R06)
- [ ] F3 — comparação de métrica com a versão ativa e bloqueio sem `--force` (PM-R02 a PM-R05)
- [ ] F5 — escrita atômica de `active.json` (PM-R07, PM-R08)
- [ ] Retentativa de I/O com backoff 1s/2s/4s nos dois scripts
- [ ] Log estruturado JSON por evento nos dois scripts (§5)
- [ ] Fixtures: candidato aprovado/reprovado, `model_version` existente/inexistente, `active.json` presente/ausente, cenário de regressão de métrica
- [ ] Testes `tests/unit/` e `tests/contract/` cobrindo CA-01 a CA-13

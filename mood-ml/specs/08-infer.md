# 08 — Serviço de Inferência Online (camada `infer/`)

**Status:** Aceita · **Versão da spec:** `if-1` · **Data:** 2026-09-20
**Implementa:** [data-model.md §2.2](../../data-structure/data-model.md) (`POST /internal/v1/infer` — `InferRequest`/`InferResponse`).
**Depende de:** [ADR-0001](../../decisions/ADR-0001-escala-do-humor.md) (`scale`, `mood_label = null`, recorte `[-1, 1]`), [ADR-0007](../../decisions/ADR-0007-humor-por-conversa.md) (janela de 30 mensagens da conversa), [ADR-0004](../../decisions/ADR-0004-retentativa-e-quarentena.md) (sem retry no `mood-ml`, quarentena é do `mood-api`); [constitution.md](../../decisions/constitution.md) P1, P3, P4, P5; [04-transform.md](04-transform.md) (`extract_features`), [06-train-evaluate.md](06-train-evaluate.md) (`clip_score`), [07-registry.md](07-registry.md) (`active.json`, `manifest.json`)
**Consumido por:** `mood-api` (fora deste repositório de especificação — cliente HTTP interno)
**Implementado em:** [infer/predict.py](../infer/predict.py), [main.py](../main.py)

---

## 1. Objetivo geral

Um serviço HTTP (processo `mood-ml` separado do `mood-api`, README §"Arquitetura") que faz três coisas, nesta ordem de importância:

1. **Carrega o modelo ativo uma vez, na subida, e confere que ele é compatível com o código antes de aceitar qualquer requisição** — a verificação de P4 (`feature_spec_version`, `history_window`, `history_scope`) que a `data-model.md §4.4` exige "no carregamento do modelo".
2. **Aplica o contrato**: recebe `InferRequest`, valida contra `data-model.md §2.2` além do que a tipagem sozinha garante (papel das mensagens de `history`), e devolve `InferResponse` no formato exato que o `mood-api` espera.
3. **Nunca inventa um score.** Esta é a única camada do pipeline cujo erro tem efeito imediato e visível para um cliente de verdade (via `mood-api` → `chat-app`) — e é aqui que P3 (*"nunca gravar score de fallback"*) se torna uma regra literal de cada resposta HTTP: toda falha é `4xx`/`5xx` com `{"error_code", "detail"}`, nunca um score calculado sobre um erro engolido.

O que este documento **não** cobre: como o modelo foi treinado (specs 04–06) ou registrado (spec 07) — chega aqui como um fato consumado, `models/<model_version>/{manifest.json,model.joblib}` e `models/active.json` já prontos.

## 2. Arquitetura e fluxo de dados

### 2.1 Posição no fluxo

```
mood-api                                    mood-ml (processo separado, main.py)
   │ POST /internal/v1/infer                        │
   │  {request_id, customer_id, conversation_id,     │
   │   history: [...]}                               │
   ├─────────────────────────────────────────────────►
   │                                                  │  modelo já carregado em memória
   │                                                  │  (na subida, uma vez — §2.2)
   │                                                  ▼
   │                                        extract_features(history)  [spec 04]
   │                                                  │
   │                                                  ▼
   │                                        model.predict(...) → clip_score  [spec 06]
   │                                                  │
   │  {request_id, customer_id, conversation_id,      ▼
   │   trigger_message_id (= history[-1].message_id), InferResponse
   │   score, scale, mood_label: null, model_version,  │
   │   computed_at}                                    │
   ◄─────────────────────────────────────────────────┤
```

### 2.2 Subida do serviço — três desfechos possíveis

Diferente de toda spec anterior, aqui a "carga" não é de dado, é de **modelo**, e acontece uma vez por vida do processo, antes de qualquer requisição ser aceita:

| Situação | Desfecho |
|---|---|
| `models/active.json` não existe | O processo **sobe normalmente**, em **modo degradado**: nenhum modelo em memória. `GET /internal/v1/infer` responde `503 ml_unavailable` a qualquer chamada. É o estado legítimo de "ainda ninguém promoveu nada" (bootstrap do projeto, spec 07) |
| `active.json` existe, mas `model_version` referenciado não tem `manifest.json` legível ou `model.joblib` corrompido/ausente | O processo **recusa subir** — encerra com falha antes de aceitar conexões. É um estado de registro quebrado, não um "ainda não" legítimo |
| `active.json` existe, modelo carrega, mas `manifest.feature_spec_version`/`history_window`/`history_scope`/`algorithm` diverge do que **este código** implementa | O processo **recusa subir** (P4) — servir mesmo assim significaria calcular features com uma premissa que o modelo não foi treinado para ver, silenciosamente |
| `active.json` existe, modelo carrega, tudo compatível | Operação normal — modelo, manifesto e `model_version` ficam em memória para toda a vida do processo |

**Por que os dois primeiros casos de falha são diferentes um do outro, apesar de os dois "impedirem servir corretamente".** `active.json` ausente é uma fase esperada do ciclo de vida do projeto — o serviço deve poder subir mesmo assim, para que operação e observabilidade (`/healthz`, §3.6) funcionem enquanto ninguém promoveu nada. Um `active.json` que aponta para algo quebrado, ou para um modelo incompatível com o código, é sempre um erro de configuração — subir mesmo assim esconderia o problema até a primeira requisição falhar (ou, pior, até uma requisição *não* falhar, mas processar com a premissa errada).

**Sem recarregamento em runtime (decisão do usuário).** O modelo é lido uma única vez. Uma nova promoção (spec 07) só passa a valer depois de reiniciar o processo — o mesmo espírito de "`promote` é ação deliberada, nunca automática" da spec 07, estendido para "e só tem efeito depois de uma ação deliberada de reiniciar o serviço".

### 2.3 Passo a passo lógico — por requisição

| Passo | O que faz | Falha possível |
|---|---|---|
| 1 | Valida o corpo contra `InferRequest` (tipos, `history` com 1–30 itens) | `400 invalid_request` |
| 2 | QUANDO em modo degradado (§2.2), responde antes de qualquer outra checagem | `503 ml_unavailable` |
| 3 | Valida regra de negócio: todo item de `history` tem `role = "customer"` | `400 invalid_history` |
| 4 | `extract_features(history)` (spec 04) — monta `text_clean`/`context_clean` | `500 internal_error` (não deveria acontecer se o passo 3 já validou a forma; ver §4.1) |
| 5 | `model.predict(...)` sobre o `Pipeline` carregado | `500 internal_error` |
| 6 | `clip_score(raw)` (spec 06) | — (função total, não levanta) |
| 7 | Monta `InferResponse`: ecoa IDs do pedido, `score` recortado, `scale = "-1 to 1"`, `mood_label = null`, `model_version` do modelo carregado, `computed_at = now()` UTC | — |
| 8 | Loga a conclusão (latência, `model_version`) e devolve `200` | — |

Todo o passo 4–6 roda inteiramente em memória — nenhum I/O de arquivo ou rede no caminho de uma requisição (§3.2).

## 3. Requisitos técnicos e premissas

### 3.1 Conexões e autenticação

**Nenhuma nesta camada.** O serviço recebe conexões HTTP (não abre nenhuma) e não lê arquivo além da subida (§2.2). Autenticação entre `mood-api` e `mood-ml` está listada como "ainda não definida" no `README.md` — fora do escopo desta spec, que assume a chamada já chegou autorizada por quem estiver na frente dela (rede interna, *gateway*, o que vier a ser decidido).

```bash
uvicorn main:app --host 0.0.0.0 --port 8001   # exemplo; porta e orquestração ficam com a spec 09/deploy
```

Runtime: Python 3.12, `fastapi`, `uvicorn[standard]`, `pydantic`, `joblib`, `pandas` — todos já em `requirements.txt`.

### 3.2 Formato dos dados

**Só JSON sobre HTTP. Nenhum Parquet nesta camada** — é a única spec da série onde o formato de arquivo do restante do pipeline (`.parquet`, `.jsonl`) simplesmente não aparece: a entrada é o corpo de uma requisição, a saída é o corpo de uma resposta.

`InferRequest` e `InferResponse` são exatamente os esquemas de `data-model.md §2.2`, reproduzidos aqui só para referência rápida (a `data-model.md` é a fonte, este documento não a substitui):

| Campo (`InferRequest`) | Tipo | Regra adicional desta spec |
|---|---|---|
| `request_id` | string (UUID) | — |
| `customer_id`, `conversation_id` | string | — |
| `history` | lista de `HistoryMessage`, 1–30 itens | Todo item DEVE ter `role = "customer"`. O mood-api acrescenta a mensagem disparadora por último, mesmo com timestamp atrasado; não há `trigger_message_id` separado no pedido |

| Campo (`InferResponse`) | Tipo | Origem |
|---|---|---|
| `request_id`, `customer_id`, `conversation_id` | string | Eco do pedido |
| `trigger_message_id` | string | `history[-1].message_id` do pedido |
| `score` | number | `clip_score(model.predict(...))`, em `[-1.0, 1.0]` |
| `scale` | string | Sempre `"-1 to 1"` |
| `mood_label` | null | Sempre `null` (ADR-0001) |
| `model_version` | string | Do modelo carregado na subida — nunca `"untrained"` |
| `computed_at` | string ISO-8601 UTC | Instante da resposta |

**Erro** (qualquer `4xx`/`5xx`): `{"error_code": string, "detail": string}` — inclusive quando a violação é de validação estrutural do Pydantic, cujo corpo padrão do FastAPI (`{"detail": [...]}`) **não** segue esse formato e precisa ser reescrito por um manipulador de exceção dedicado (§8, D5).

### 3.3 Estratégia de carga e idempotência

"Carga", aqui, é a carga do **modelo em memória** (§2.2) — não há artefato de dado para versionar ou sobrescrever nesta camada; isso já aconteceu nas specs 02–07.

**Idempotência, redefinida mais uma vez.** Assim como em `transform/` (spec 04), idempotência aqui não significa "não processar a mesma requisição duas vezes" — significa **transparência referencial**: a mesma `history` submetida ao mesmo modelo carregado produz sempre o mesmo `score`, porque `extract_features` é pura (spec 04) e o `Pipeline` é determinístico (spec 06, `solver` fixado). O serviço **não deduplica** chamadas pelo `request_id` — se o `mood-api` reenviar a mesma requisição, o `mood-ml` calcula de novo, do zero. Isso é coerente com a ADR-0004: a retentativa de uma falha de inferência é **absorvida pela próxima mensagem do cliente**, nunca por reenvio da mesma requisição, então um mecanismo de deduplicação aqui resolveria um problema que o projeto decidiu não ter.

### 3.4 Volumetria estimada e frequência de execução

O `README.md` lista o orçamento de latência como decisão em aberto ("Metodologia e Avaliação"). Esta spec fecha essa lacuna:

**Meta: p95 < 200ms por requisição**, medido do recebimento do corpo ao envio da resposta (exclui rede entre `mood-api` e `mood-ml`, que é responsabilidade da spec de infraestrutura/deploy). Compatível com a estimativa da ADR-0008 para a abordagem A (TF-IDF + Ridge, "latência de milissegundos por mensagem na CPU"), com folga para o pior caso de janela cheia (30 itens de contexto). Esse número também vira o **critério objetivo de decisão entre a abordagem A e a C** quando C existir: se C não bater 200ms em CPU, essa é uma consequência a relatar no TCC, não um motivo para mudar o alvo depois do fato.

Não há volumetria de requisições por segundo a estimar — o `README.md` já registra que o projeto ainda não tem tráfego real (*"a implementação e a configuração reprodutível ainda não estão disponíveis"*). Esta spec é dimensionada para a janela máxima possível (30 mensagens de contexto, ADR-0007), não para um volume de chamadas.

## 4. Tratamento de erros e resiliência

### 4.1 Por que quase não há retry

**No caminho de requisição: nenhum.** Depois da subida, todo o trabalho de `extract_features` → `predict` → `clip_score` é CPU pura sobre dados em memória — não há I/O a falhar de forma transitória (mesmo raciocínio da spec 04, §4.1). Se uma requisição falha no passo 4 ou 5 (§2.3), falha de novo com a mesma entrada, sempre: é um erro determinístico, não um problema de rede ou disco a se resolver tentando de novo.

**Na subida: sim, para I/O.** Ler `active.json`, `manifest.json` e `model.joblib` é I/O de arquivo local, sujeito ao mesmo tipo de falha transitória das specs 02–07 (arquivo temporariamente bloqueado, por exemplo). 3 tentativas, *backoff* 1s/2s/4s, antes de decidir entre os desfechos da §2.2.

### 4.2 Catálogo de respostas de erro

| `error_code` | HTTP | Quando |
|---|---|---|
| `invalid_request` | 400 | Corpo não corresponde a `InferRequest` (Pydantic) |
| `invalid_history` | 400 | `history` viola regra de negócio: papel errado |
| `ml_unavailable` | 503 | Serviço em modo degradado (§2.2) |
| `internal_error` | 500 | Exceção não esperada em `extract_features`/`predict` |

**P3 é absoluto aqui: nenhuma dessas respostas carrega um `score`.** O corpo de erro nunca tem os campos de `InferResponse` — são esquemas completamente diferentes, e essa separação de tipo (não só de convenção) é o que torna impossível, por construção, devolver acidentalmente um score de reserva dentro de um corpo de erro.

### 4.3 Alertas e notificações

**Nenhum canal externo direto desta camada** — mesma razão de sempre (MVP sem plantão). A diferença em relação às specs 02–07: aqui quem primeiro vê uma falha não é um operador olhando um terminal, é o `mood-api`, de forma síncrona, a cada requisição. A ADR-0004 já decidiu o que ele faz com isso (grava `inference_failures`, mantém o último score válido, conta falhas consecutivas para quarentena) — esta spec só garante que o sinal que chega até lá é honesto (um `error_code` real, nunca um score disfarçado). Um alerta de verdade (crash-loop do processo, taxa de erro acima de um limiar) é responsabilidade de infraestrutura/observabilidade de produção, fora do escopo deste MVP.

## 5. Logs e observabilidade

### 5.1 Padrão

Mesmo formato das specs anteriores: JSON por linha em `stderr`/`stdout` do processo, `ts` ISO-8601 UTC, `level`, `event`. Diferença estrutural: os eventos desta camada são de **duas naturezas** — os de subida (uma vez por vida do processo) e os de requisição (um por chamada).

> Implementação: `common.log`; a retentativa de I/O da subida (IF-R19) usa `common.io.with_io_retry` ([spec 10](10-common.md)).

### 5.2 Eventos de subida

| Nível | Evento | Carga |
|---|---|---|
| INFO | `startup_started` | — |
| WARNING | `startup_degraded` | `reason: "active_json_missing"` — serviço sobe sem modelo |
| INFO | `model_loaded` | `model_version`, `algorithm`, `feature_spec_version`, `history_window`, `history_scope` |
| ERROR | `startup_refused` | `reason` (`registry_broken` \| `feature_spec_mismatch`), `expected`, `found` — processo encerra logo em seguida |
| WARNING | `io_retry` | `attempt`, `operation`, `errno` |

### 5.3 Eventos por requisição

| Nível | Evento | Carga |
|---|---|---|
| INFO | `request_received` | `request_id`, `customer_id`, `conversation_id`, `history_size` |
| INFO | `request_completed` | `request_id`, `model_version`, `score`, `latency_ms` |
| WARNING | `request_rejected` | `request_id`, `error_code`, `detail` — para `400`/`503` |
| ERROR | `request_failed` | `request_id`, `error_code: "internal_error"`, `exc_type`, *traceback* |

`score` no evento `request_completed` é um número, não texto — não conflita com P5. `history_size` é uma contagem, nunca o conteúdo de `history`.

### 5.4 O que nunca aparece em log, em hipótese alguma

`history[].text` — cru ou mascarado — **nunca é logado**, em nenhum nível, em nenhum evento. É a mesma regra literal de `data-model.md §8`: *"Logs de erro carregam `message_id`, nunca `text`"*. Diferente de `ingest/validate.py` (spec 02), que tinha um modo opt-in (`--review-sample`) para materializar texto sob controle, aqui **não existe** esse modo — é um serviço online, sem revisão manual no caminho, e qualquer exceção logada com o corpo completo da requisição (um erro comum de "log tudo para depurar") vazaria exatamente o que P5 proíbe. O manipulador de exceção (§8, D5) serializa só `request_id`, `error_code` e uma mensagem genérica — nunca `repr(request)` ou equivalente.

### 5.5 `GET /internal/v1/healthz`

Endpoint adicional desta spec (não faz parte do contrato `InferRequest`/`InferResponse` da `data-model.md`, mas serve à mesma necessidade de observabilidade que motivou os relatórios das specs 02–07):

```json
{ "status": "ok", "model_version": "mood-2026.09.0" }
```
ou, em modo degradado:
```json
{ "status": "degraded", "model_version": null }
```

Não substitui os relatórios de auditoria offline — é o equivalente, para um serviço de vida longa, de "existe um jeito barato de perguntar 'está tudo bem?' sem disparar uma inferência de verdade".

## 6. Segurança e conformidade

### 6.1 T2 roda de novo, sempre, sem exceção

`extract_features` (spec 04) aplica `mask_pii` a cada item de `history`, incondicionalmente — o mesmo raciocínio já registrado na spec 04 §2.3: a `data-model.md §2.2` deixa em aberto se o `mood-api` já mascara antes de montar `HistoryMessage.text`, e esta camada **não confia nisso**. Como `mask_pii` é idempotente (spec 04, TR-R09), aplicar de novo sobre um texto já mascarado é uma operação neutra. É a garantia de P5 mais importante desta spec, porque é a única onde texto de cliente **de verdade** (não sintético) algum dia vai passar, se o projeto avançar além do MVP.

### 6.2 Base LGPD/GDPR

Diferente das specs 02–07 (que operam só sobre corpus sintético), esta é a camada que **vai** processar dado real quando o `mood-api` estiver em produção — o próprio `README.md` lista isso como pendência ("Ética e Privacidade": base legal, mascaramento, retenção, ainda em aberto). Esta spec não resolve essas pendências (não são dela), mas garante que o mecanismo técnico de mascaramento (T2) já está no caminho obrigatório de qualquer texto que entre no modelo, **antes** dessas decisões de política serem tomadas — a infraestrutura de conformidade já existe, mesmo que a política que a rege ainda não esteja escrita.

### 6.3 P1 e P3

Nenhuma referência a `duckdb`/`.duckdb` em `infer/` — mesma verificação de *lint*. P3 é o princípio organizador desta spec inteira (§4.2): nenhuma resposta de erro carrega score, por construção de esquema, não por disciplina de código.

## 7. Requisitos verificáveis

**Subida**
- **IF-R01** Na subida, o processo DEVE tentar ler `models/active.json`.
- **IF-R02** QUANDO `active.json` não existir, o processo DEVE subir em modo degradado — nenhuma exceção, nenhum encerramento.
- **IF-R03** QUANDO `active.json` existir mas o `model_version` referenciado não tiver `manifest.json` legível ou `model.joblib` válido, o processo DEVE recusar a subida.
- **IF-R04** QUANDO `manifest.feature_spec_version`, `history_window`, `history_scope` ou `algorithm` divergirem do que o código desta versão implementa, o processo DEVE recusar a subida.
- **IF-R05** Um modelo carregado com sucesso DEVE permanecer em memória por toda a vida do processo, sem recarregar entre requisições nem observar mudanças em `active.json`.

**Contrato HTTP**
- **IF-R06** `POST /internal/v1/infer` DEVE validar o corpo contra `InferRequest`; violação estrutural DEVE devolver `400 invalid_request` no formato `{"error_code","detail"}` (nunca o corpo padrão do FastAPI).
- **IF-R07** QUANDO o serviço estiver em modo degradado, toda chamada a `/infer` DEVE devolver `503 ml_unavailable` antes de qualquer outra validação.
- **IF-R08** QUANDO algum item de `history` tiver `role != "customer"`, o endpoint DEVE devolver `400 invalid_history`.
- **IF-R09** QUANDO `extract_features` ou a predição levantarem exceção, o endpoint DEVE devolver `500 internal_error`.
- **IF-R10** O endpoint DEVE NUNCA devolver um `score` calculado a partir de um erro capturado, nem um valor padrão/neutro.
- **IF-R11** `score` DEVE passar por `clip_score` antes de compor a resposta.
- **IF-R12** `InferResponse` DEVE ecoar `request_id`, `customer_id`, `conversation_id` do pedido, sem alteração, e `trigger_message_id` DEVE ser `history[-1].message_id`.
- **IF-R13** `mood_label` DEVE ser sempre `null`.
- **IF-R14** `model_version` na resposta DEVE ser o do modelo carregado na subida — nunca um valor fixo no código, nunca `"untrained"`.

**Health**
- **IF-R15** `GET /internal/v1/healthz` DEVE devolver `status: "ok"` e o `model_version` carregado quando houver modelo, e `status: "degraded"` com `model_version: null` quando não.

**Idempotência**
- **IF-R16** Para a mesma `history` e o mesmo modelo carregado, chamadas repetidas DEVEM produzir o mesmo `score`.
- **IF-R17** O serviço DEVE NÃO deduplicar chamadas pelo `request_id`.

**Resiliência**
- **IF-R18** O caminho de requisição (passos 4–6, §2.3) DEVE NÃO realizar I/O de qualquer tipo.
- **IF-R19** A leitura de `active.json`/`manifest.json`/`model.joblib` na subida DEVE retentar 3 vezes com *backoff* 1s/2s/4s antes de decidir entre os desfechos da §2.2.

**Segurança**
- **IF-R20** Nenhum log DEVE conter `history[].text`, mascarado ou não, em nenhum nível, em nenhum evento.
- **IF-R21** `extract_features` DEVE ser chamada para toda requisição válida, independentemente de o texto já chegar mascarado.
- **IF-R22** O módulo `infer/` DEVE NÃO importar `duckdb` nem referenciar caminho `.duckdb` (P1).

### 7.1 Catálogo de erros

| Código | Requisito | HTTP |
|---|---|---|
| `IF_R06_INVALID_REQUEST` | IF-R06 | 400 |
| `IF_R08_INVALID_HISTORY` | IF-R08 | 400 |
| `IF_R07_ML_UNAVAILABLE` | IF-R07 | 503 |
| `IF_R09_INTERNAL_ERROR` | IF-R09 | 500 |
| `IF_R03_STARTUP_REGISTRY_BROKEN` | IF-R03 | — (processo não sobe) |
| `IF_R04_STARTUP_FEATURE_SPEC_MISMATCH` | IF-R04 | — (processo não sobe) |

## 8. Decisões deste documento

| # | Decisão | Motivo | Consequência assumida |
|---|---|---|---|
| D1 | **`active.json` ausente → sobe degradado; registro quebrado ou P4 divergente → recusa subir** | Ausente é um estado legítimo de bootstrap; os outros dois são configuração quebrada que não deveria ser servida silenciosamente | O primeiro `promote` (spec 07) é o que tira o serviço do modo degradado — nenhuma ação adicional necessária em `infer/` |
| D2 | **Sem *hot reload***; nova promoção exige reiniciar o processo | Evita um componente de concorrência (trocar modelo em memória com requisições em voo) que o MVP não precisa; coerente com "promoção é ação deliberada" já fixado na spec 07 | Uma promoção só tem efeito operacional depois de um restart manual — documentar isso no procedimento de deploy (fora desta spec) |
| D3 | **Meta de latência: p95 < 200ms**, fechando a pendência do `README.md` | Compatível com a estimativa da ADR-0008 para a abordagem A; vira critério objetivo de comparação com a abordagem C quando ela existir | `README.md` deveria ser atualizado para citar esta spec em vez de listar o item como aberto — pendência a levar para fora do `mood-ml`, mesmo padrão das specs 03/06/07 |
| D4 | **`GET /internal/v1/healthz`** — endpoint novo, fora do contrato `InferRequest`/`InferResponse` original | Distingue "processo no ar, sem modelo" de "processo no ar, servindo" sem precisar disparar uma inferência de teste | Mais um endpoint para a `data-model.md` eventualmente documentar (não é parte do contrato público com o `mood-api`, é operacional) |
| D5 | **Corpo de erro do FastAPI reescrito** para `{"error_code","detail"}` via manipulador de exceção dedicado | O padrão do FastAPI (`{"detail": [...]}` para erro de validação Pydantic) não seria distinguível de um erro de negócio pelo `mood-api`, e quebraria a uniformidade que P3 exige de toda resposta de erro | Todo erro de validação passa por um único ponto de serialização — reforça IF-R20 (evita vazar o corpo bruto da requisição num log de exceção não tratado) |

## 9. Critérios de aceite

- **CA-01** Modelo compatível carregado, requisição válida → `200`, `score` em `[-1.0, 1.0]`, `mood_label: null`, `model_version` correto.
- **CA-02** `active.json` ausente → processo sobe; `/infer` responde `503 ml_unavailable`; `/healthz` responde `status: "degraded"`.
- **CA-03** `active.json` aponta para `model_version` sem `manifest.json` → processo não conclui a subida (teste instancia o app e espera falha).
- **CA-04** `manifest.json` com `history_window: 1` (valor antigo/incompatível) → processo não conclui a subida.
- **CA-05** `history` com algum item `role: "agent"` → `400 invalid_history`.
- **CA-06** Corpo sem `history` → `400 invalid_request`, corpo `{"error_code","detail"}` (não o formato padrão do FastAPI).
- **CA-07** Duas chamadas idênticas (mesma `history`) → mesmo `score`, byte a byte no JSON de resposta exceto `computed_at`.
- **CA-08** `extract_features` forçada a levantar exceção (teste com *mock*) → `500 internal_error`, nunca um `score` no corpo.
- **CA-09** Medição local de latência sobre N requisições sintéticas com janela de 30 itens → p95 < 200ms.
- **CA-10** Bateria de requisições com PII sintética em `history[].text` → nenhuma ocorrência do texto bruto nos logs gerados.
- **CA-11** Teste de P1: nenhuma ocorrência de `duckdb`/`.duckdb` em `infer/`.

## 10. Fora de escopo

- **Quarentena e contagem de falhas consecutivas** — responsabilidade do `mood-api` (ADR-0004); `infer/` só garante que o sinal de falha é honesto.
- **Autenticação entre `mood-api` e `mood-ml`** — README lista como "ainda não definida"; fora desta spec.
- **Recarregamento em runtime (*hot reload*)** — decisão explícita (D2); um restart manual é o mecanismo do MVP.
- **Abordagem C (embeddings)** — o `algorithm` do manifesto já é conferido (IF-R04); nenhuma lógica específica de C é tratada aqui.
- **Rate limiting, autenticação de borda, TLS** — infraestrutura de deploy, fora do escopo de `mood-ml`.
- **Orquestração, CI e alvos de Makefile.** Spec 09.

## 11. Checklist de implementação

- [ ] `main.py` — `lifespan`/evento de subida do FastAPI chamando o carregamento do modelo (IF-R01 a IF-R05)
- [ ] Verificação de compatibilidade P4 contra constantes do código (`feature_spec_version`, `history_window`, `history_scope`, `algorithm` suportados)
- [ ] Retentativa de I/O na subida (IF-R19)
- [ ] `POST /internal/v1/infer` — validação estrutural (`InferRequest`) e de negócio (papel de cada item de `history`) (IF-R06 a IF-R09)
- [ ] Manipulador de exceção único para o formato `{"error_code","detail"}` (D5)
- [ ] Integração com `extract_features` (spec 04) e o `Pipeline` carregado; `clip_score` (spec 06) antes da resposta (IF-R09 a IF-R14)
- [ ] `GET /internal/v1/healthz` (IF-R15, D4)
- [ ] Log estruturado JSON por evento de subida e por requisição (§5), sem `history[].text` em nenhum caminho (IF-R20)
- [ ] Testes de contrato para os três desfechos de subida (CA-02 a CA-04)
- [ ] Teste de validação de negócio (CA-05)
- [ ] Teste de determinismo (CA-07) e de ausência de score em erro (CA-08)
- [ ] Teste de latência sintético (CA-09)
- [ ] Testes `tests/unit/` e `tests/contract/` cobrindo CA-01 a CA-11

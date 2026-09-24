# 02 — Ingestão e Validação do Corpus (camada `ingest/`)

**Status:** Aceita · **Versão da spec:** `ig-1` · **Data:** 2026-09-20
**Implementa:** [01-dataset-contract.md](01-dataset-contract.md) (`dc-1`) — este documento diz **como** validar o que aquele contrato **exige**.
**Depende de:** [constitution.md](../../decisions/constitution.md) P1, P4, P5; ADR-001, ADR-002 ([00-decisoes.md](00-decisoes.md)); [ADR-0004](../../decisions/ADR-0004-retentativa-e-quarentena.md) (analogia da política de retentativa); [data-model.md §4.1](../../data-structure/data-model.md)
**Consumido por:** [labels/build.py](../labels/build.py) (T4, spec 03), [transform/](../transform/) (T1–T3, spec 04), [pipeline.py](../pipeline.py) (spec 09)
**Implementado em:** [ingest/validate.py](../ingest/validate.py)

---

## 1. Objetivo geral

O corpus rotulado **chega pronto**, produzido por um gerador que vive fora deste pipeline (`dc-1` §8: o contrato cobra o resultado, não o processo). A camada `ingest/` é o **portão de entrada** desse arquivo: ela decide se o corpus entra ou não no pipeline de treino.

O script tem um único propósito: **dado um arquivo `.jsonl`, provar que ele cumpre o contrato `dc-1` e registrar essa prova**. A prova é o `validation_report.json` (CA-01 da spec 01), que carrega contagens, distribuições e o `sha256` do arquivo aprovado.

Três não-objetivos delimitam a camada:

- **Não transforma.** Limpeza (T1), mascaramento (T2) e features (T3) são de `transform/` (spec 04). O ingest lê e mede; nunca reescreve o dado.
- **Não gera.** [ingest/synthetic.py](../ingest/synthetic.py) está depreciado por decisão registrada. O ingest não chama LLM nem monta corpus.
- **Não toca banco.** P1 é absoluto: nenhum `import duckdb`, nenhum caminho `.duckdb` nesta camada. O único insumo é um arquivo em disco.

A consequência de projeto é que **uma etapa seguinte nunca decide sobre qualidade de dado**. Se `labels/build.py` está rodando, é porque o corpus já passou pelo portão.

## 2. Arquitetura e fluxo de dados

### 2.1 Posição no fluxo

```
[gerador externo]                      fora do pipeline, fora do repositório
       │  operador salva o arquivo à mão
       ▼
data/raw/synthetic/<corpus_id>.jsonl   fonte única da verdade, imutável
       │
       ▼
ingest.validate  ──►  data/reports/<corpus_id>/validation_report.json
       │                    (status, contagens, distribuições, sha256)
       │
       │ load_corpus() — leitor canônico, ordenado
       ▼
labels/build.py (T4) ──► transform/ (T1–T3) ──► train/
```

O ingest **não materializa dado normalizado**. O JSONL cru permanece a fonte única; a ordenação exigida pelo contrato é entregue em tempo de leitura (§2.3).

### 2.2 Passo a passo lógico

Sete fases, executadas em ordem. Uma fase só roda se a anterior não rejeitou o corpus.

| Fase | Nome | O que faz | Custo |
|---|---|---|---|
| F0 | Resolução da entrada | Valida o argumento da CLI: arquivo existe, é legível, extensão `.jsonl`, está sob `data/raw/synthetic/`, nome do arquivo igual ao `corpus_id` esperado | O(1) |
| F1 | Impressão digital | Calcula `sha256` em streaming (blocos de 64 KiB) e mede o tamanho em bytes | 1 passe de I/O |
| F2 | Decisão de idempotência | Compara o hash com o do report anterior, se houver. Pode encerrar aqui como no-op (§3.3) | O(1) |
| F3 | Esquema, linha a linha | Parse JSON, esquema da `dc-1` §3, tipos, limites, rótulo por papel. Acumula erros até o teto de 50 | 1 passe de I/O |
| F4 | Integridade entre linhas | Unicidade de `message_id`, prefixo `syn-`, `conversation_id`→`customer_id`, `customer_id`→`persona`, unicidade de `sent_at` na conversa | sobre os acumuladores de F3 |
| F5 | Composição e realismo | Contagens mínimas, distribuição de rótulos e personas, proporção de PII fictícia, tamanho de conversa, variação de humor | sobre os acumuladores de F3 |
| F6 | Emissão | Grava o report de forma atômica, loga o resumo e encerra com o *exit code* da §4.3 | O(1) |

**Por que F3 antes de F5.** Composição sobre corpus com esquema quebrado produz números que não significam nada — a distribuição de rótulos de um arquivo com 200 linhas ilegíveis é ruído. F5 é bloqueada por qualquer erro em F3 ou F4, e o report registra `composition: "not_evaluated"`.

**Por que o teto de 50 erros em F3.** Falhar na primeira linha inválida faria o operador corrigir um problema sistemático (ex.: `sent_at` sem sufixo `Z` em todo o arquivo) por 3000 iterações. Acumular tudo, por outro lado, gera um report inútil de megabytes. Cinquenta erros bastam para reconhecer um padrão.

### 2.3 O leitor canônico e o problema da ordenação

A `dc-1` §2 diz que as linhas ordenadas são *recomendadas, não obrigatórias*, porque "o ingest reordena". Com a saída restrita ao report, não existe artefato reordenado onde gravar esse resultado. A obrigação continua valendo, então ela é cumprida **em tempo de leitura**:

`ingest/validate.py` expõe `load_corpus(path) -> list[Message]`, que devolve as mensagens ordenadas por `(conversation_id, sent_at)`. `labels/` e `transform/` **DEVEM** consumir o corpus exclusivamente por essa função, nunca abrindo o `.jsonl` por conta própria.

É o mesmo raciocínio de P4 aplicado à ordenação: uma regra de leitura duplicada em dois módulos é uma regra que vai divergir. O custo assumido é reler e reordenar o arquivo a cada etapa — irrelevante na volumetria da §3.4 (§3.5 dimensiona).

**Interpretação de DC-R06.** O requisito fala em `sent_at` "repetido ou fora de ordem". Como a §2 do contrato admite arquivo desordenado, "fora de ordem" **não pode** significar ordem física das linhas: as duas regras se contradiriam. A leitura vigente é: dentro de uma conversa, o conjunto de `sent_at` **DEVE** ser estritamente crescente **após a ordenação**, o que equivale a exigir unicidade de `sent_at` por conversa. Arquivo desordenado gera *warning*, nunca rejeição.

## 3. Requisitos técnicos e premissas

### 3.1 Conexões e autenticação

**Não há.** A decisão de origem (§8) elimina a categoria: o gerador roda fora do pipeline e o operador salva o arquivo em `data/raw/synthetic/`. O ingest recebe um caminho de sistema de arquivos pela CLI.

Portanto, **não existem** no escopo desta camada: credenciais, variáveis de ambiente com segredo, chave de API, token, cliente HTTP, SDK de cloud ou *connection string*. A superfície de ataque e o custo de operação são os de ler um arquivo local.

A única autorização relevante é a de sistema de arquivos: o processo precisa de leitura em `data/raw/synthetic/` e escrita em `data/reports/`. Ambos os diretórios estão no `.gitignore` (P5).

**CLI e execução**

```bash
python -m ingest.validate <caminho-do-corpus.jsonl> \
    [--report-dir data/reports] \
    [--skip-composition]        # isenta DC-R09..R11 (CA-03 da spec 01)
    [--review-sample N]         # amostra para conferência manual de DC-R12
    [--log-format json|text]    # default: json
    [--log-level INFO|WARNING|ERROR]
```

Runtime: **Python 3.12** (fixado por [setup-mood-ml.sh](../../setup-mood-ml.sh)). Dependências: **somente biblioteca padrão** (`json`, `hashlib`, `re`, `pathlib`, `logging`, `argparse`, `datetime`, `os`). O ingest não importa `pandas`, `pyarrow` nem `scikit-learn` — validar um JSONL de poucos megabytes não justifica carregar um stack de dados, e a camada de entrada ser leve significa que ela falha rápido e por motivos óbvios.

Os detectores de PII (CPF, telefone, e-mail) usados para medir DC-R14 **DEVEM** ser importados de [transform/mask.py](../transform/mask.py), não reimplementados aqui. Medir e mascarar com definições diferentes é como um corpus passa em DC-R14 e depois atravessa T2 sem ser mascarado.

Invocação pelo orquestrador (spec 09): `make ingest CORPUS=<corpus_id>` e `python -m pipeline ingest --config configs/pipeline.yaml`.

### 3.1a API pública do módulo

A CLI é uma casca fina sobre uma função importável — é o que `labels/build.py` (spec 03) chamaria diretamente se um dia precisasse validar sem passar pelo processo `python -m`, e é o nome que o restante da documentação do projeto usa para se referir a esta etapa:

```python
def validate_corpus(path: str | Path, skip_composition: bool = False) -> ValidationReport: ...
```

`ValidationReport` é a contraparte em memória do esquema da §3.6 — mesmos campos (`status`, `source`, `counts`, `distributions`, `realism`, `errors`, `warnings`), como estrutura de dados (`dataclass`/`TypedDict`), não como arquivo. A função:

- **não** grava `validation_report.json` sozinha nem decide o *exit code* — isso é responsabilidade da CLI (F6, §2.2), que serializa o `ValidationReport` devolvido e traduz `status` em código de saída (§4.3);
- **não** aplica a decisão de idempotência (F2) por conta própria — recebe o hash já calculado pela CLI e o compara, mas quem decide "no-op vs. revalida vs. aborta por conflito" é a camada que chama, porque essa decisão depende do sistema de arquivos (existência de report anterior), não da validação em si;
- é o ponto de entrada que `tests/unit/` exercita diretamente, sem precisar disparar um subprocesso — mais rápido e mais fácil de depurar do que testar só pela CLI.

### 3.2 Formato dos dados

| | Entrada | Saída |
|---|---|---|
| Formato | JSON Lines (`.jsonl`) | JSON (UTF-8, indentado com 2 espaços, chaves ordenadas) |
| Codificação | UTF-8 **sem BOM**, obrigatório | UTF-8 |
| Caminho | `data/raw/synthetic/<corpus_id>.jsonl` | `data/reports/<corpus_id>/validation_report.json` |
| Esquema | `dc-1` §3 (9 campos) | §3.6 deste documento |
| Grão | 1 linha = 1 mensagem | 1 arquivo = 1 execução aprovada |

Regras de leitura: nova linha `\n` ou `\r\n` (tolerado, arquivo pode vir de Windows); última linha pode ou não terminar com nova linha; **linhas em branco são ignoradas** e contadas como *warning*; nenhum campo extra é aceito — chave fora da §3 é `DC_R01_SCHEMA_VIOLATION`, porque campo silenciosamente ignorado é como um contrato deriva.

Nem Parquet, nem CSV, nem tabela SQL entram nesta camada. Parquet aparece em `labels/` e `datasets/` (data-model §4.3); SQL não aparece nunca (P1).

### 3.3 Estratégia de carga e idempotência

A carga é **por corpus inteiro**, e o modelo mental correto não é *append* nem *overwrite*, mas **verificação com trava de conteúdo**. Cada `corpus_id` é uma identidade imutável, e o `sha256` é o que garante isso.

Isto segue a regra de arquitetura do projeto de que **artefatos são imutáveis**: um reprocessamento gera um novo ID, nunca sobrescreve um existente. Regra sem exceção — não há flag que a contorne.

A decisão de F2, comparando o hash do arquivo com o `source.sha256` do report anterior:

| Estado | Comportamento | Exit |
|---|---|---|
| Não existe report para o `corpus_id` | Validação completa. Grava o report | 0 ou 3 |
| Report existe, `status = ok`, hash **igual** | **No-op idempotente.** Não relê o arquivo, não reescreve o report. Loga `already_validated` | 0 |
| Report existe, `status = rejected`, hash **igual** | Revalida e regrava — mesmo conteúdo, verificado de novo (ex.: o validador ganhou uma regra nova). Não é o cenário que a imutabilidade protege, porque nenhum artefato posterior depende de um report `rejected` | 0 ou 3 |
| Report existe, hash **diferente** | **Aborta, sem exceção.** Não valida, não sobrescreve. Não há flag que force a escrita | 4 |

**Por que abortar em hash divergente é incondicional.** Sobrescrever faria um `corpus_id` significar dois conteúdos diferentes em dois momentos — e como `dataset.json` referencia o corpus por `source_ids` (data-model §4.3), um modelo já treinado ficaria apontando para uma linhagem que deixou de existir. A única saída correta é gerar um **novo `corpus_id`**. O contrato `dc-1` já prevê o mecanismo — o padrão `syn-AAAA-MM-DD-<letra>` reserva a letra final exatamente para isso. A mensagem do *exit* 4 instrui a criar o arquivo com a próxima letra (ou a data seguinte), nunca a reutilizar o mesmo `corpus_id`.

**Idempotência real.** Duas execuções com o mesmo arquivo produzem o mesmo veredito e o mesmo report byte a byte, exceto os campos de tempo (`started_at`, `finished_at`, `duration_ms`, `run_id`). Não há estado acumulado, contador, banco ou fila. A validação é uma função pura do arquivo mais as flags.

**Portão para as etapas seguintes (contrapartida de não materializar artefato).** `labels/build.py` **DEVE** abortar quando não existir report com `status = ok` cujo `sha256` coincida com o do arquivo atual. Sem essa trava, a decisão de "somente o report" permitiria treinar sobre um corpus nunca validado, ou sobre um arquivo editado depois da validação.

### 3.4 Volumetria estimada

Derivada dos pisos do contrato (DC-R09, DC-R15), e não de medição — o corpus ainda não existe:

| Grandeza | Piso do contrato | Esperado | Teto de projeto (10×) |
|---|---|---|---|
| Clientes | 150 | 150–400 | 4.000 |
| Conversas | 300 | 300–800 | 8.000 |
| Mensagens `customer` | 2.000 | 2.000–4.000 | 40.000 |
| Linhas totais | — | 2.400–6.500 | 65.000 |
| Tamanho do arquivo | — | **1–6 MB** | ~60 MB |

`text` tem no máximo 1000 caracteres (DC-R06 do esquema), o que limita a linha a ~1,2 KB.

**Orçamento de desempenho.** No corpus esperado: < 5 s de parede. No teto de 10×: < 30 s e < 300 MB de RSS. Isso é folgado de propósito, mas impõe duas decisões: o arquivo é lido **em streaming** (`for line in f`), nunca `read().splitlines()`; e os acumuladores guardam **agregados e IDs**, nunca `text` (o que também é exigência de P5 — §7).

### 3.5 Frequência de execução

**Sob demanda, manual.** Não há agendador, cron, *trigger* ou serviço. A execução acontece quando um corpus novo é salvo, quando o validador muda, ou como pré-requisito de `make labels` / `make train`.

Expectativa realista: poucas execuções por versão de corpus — dezenas durante o desenvolvimento do gerador, quando o ciclo "gera → valida → corrige a rubrica" é rápido, e praticamente nenhuma depois que o corpus estabiliza. É o padrão de um pipeline offline de TCC, e é o que justifica a ausência de infraestrutura de orquestração.

### 3.6 Esquema do `validation_report.json`

```json
{
  "report_version": "ig-1",
  "contract_version": "dc-1",
  "status": "ok",
  "run": {
    "run_id": "b7c1f0e4",
    "started_at": "2026-09-20T13:02:11Z",
    "finished_at": "2026-09-20T13:02:14Z",
    "duration_ms": 2870,
    "tool": "ingest.validate",
    "python": "3.12.3",
    "flags": { "skip_composition": false }
  },
  "source": {
    "corpus_id": "syn-2026-09-18-a",
    "path": "data/raw/synthetic/syn-2026-09-18-a.jsonl",
    "sha256": "9f2b...c41d",
    "bytes": 3145728,
    "lines_read": 4812,
    "lines_blank_skipped": 0
  },
  "counts": {
    "messages": 4812, "messages_customer": 2410, "messages_agent": 2402,
    "conversations": 612, "customers": 214, "personas": 5
  },
  "distributions": {
    "label_levels": { "-1.0": {"n": 402, "pct": 0.167}, "-0.5": {"n": 498, "pct": 0.207},
                      "0.0": {"n": 523, "pct": 0.217}, "0.5": {"n": 511, "pct": 0.212},
                      "1.0": {"n": 476, "pct": 0.197} },
    "personas": { "objetivo": {"customers": 46, "pct": 0.215} },
    "label_by_persona": { "objetivo": { "-1.0": 61 } },
    "conversation_length": { "min": 4, "p50": 8, "max": 19 }
  },
  "realism": {
    "pii_ratio": 0.092,
    "pii_by_kind": { "cpf": 74, "phone": 88, "email": 60 },
    "mood_variation_ratio": 0.41
  },
  "manual_checks": {
    "DC-R12": { "state": "pending", "sample_path": null },
    "DC-R13": { "state": "declared_by_location" }
  },
  "errors": [],
  "warnings": [
    { "code": "file_unordered", "detail": "1.204 linhas fora da ordem recomendada (dc-1 §2)" }
  ]
}
```

`status` ∈ `ok` | `rejected`. Cada item de `errors` tem `{ "code", "line", "message_id", "detail" }`, com `line` e `message_id` nulos nos erros que não são de linha (composição, I/O). `detail` **nunca** contém `text` (§7).

## 4. Tratamento de erros e resiliência

### 4.1 Taxonomia: erro de dado, de uso e de ambiente

A distinção governa tudo nesta seção:

- **Erro de dado** (corpus viola `dc-1`) — resultado legítimo e determinístico da execução, não uma falha do script. Não se repete, não se alerta: registra-se no report e devolve-se *exit* 3.
- **Erro de uso** (caminho errado, flag inválida, arquivo fora de `raw/synthetic/`) — *exit* 2, mensagem acionável, nenhum report gravado.
- **Erro de ambiente** (I/O, permissão, disco cheio, arquivo em uso) — a única categoria em que retentar pode mudar o resultado.

### 4.2 Política de retentativa

**Sobre validação: nenhuma retentativa, por construção.** É o mesmo argumento da [ADR-0004](../../decisions/ADR-0004-retentativa-e-quarentena.md): não se constrói máquina de retentativa para uma falha que a retentativa não resolve. A validação é determinística — reexecutar com o mesmo arquivo dá exatamente o mesmo veredito. Um corpus rejeitado precisa de um gerador corrigido, não de mais uma tentativa.

**Sobre I/O: 3 tentativas com backoff 1s / 2s / 4s.** Aplicável *apenas* a duas operações, e apenas para `OSError`/`PermissionError`:

1. abrir o corpus para leitura (F1, F3);
2. gravar o report (F6).

O caso real que isso cobre é doméstico e frequente: o arquivo ainda está sendo escrito pelo gerador, ou está aberto no editor, e o Windows nega o *lock*. Esgotadas as tentativas, *exit* 1 com `IG_R15_IO_FAILED`.

> Implementação: `common.io.with_io_retry` e `common.io.atomic_write` ([spec 10](10-common.md)).

**Gravação atômica.** O report é escrito em `validation_report.json.tmp` e movido com `os.replace()`, que é atômico no mesmo volume, inclusive em NTFS. Report truncado por interrupção no meio da escrita seria pior que report ausente: `labels/` leria um `status` parcial. Não existe estado intermediário a limpar — interrupção em qualquer fase deixa o report anterior intacto.

### 4.3 *Exit codes*

| Exit | Significado | Quem age |
|---|---|---|
| 0 | Corpus aprovado, ou no-op idempotente | ninguém |
| 2 | Erro de uso da CLI | operador corrige o comando |
| 3 | **Corpus rejeitado** — um ou mais `dc-1` violados | dono do gerador corrige o corpus |
| 4 | Conflito de hash para `corpus_id` existente | operador gera um `corpus_id` novo (dc-1 §3: `syn-AAAA-MM-DD-<letra>`) — nunca sobrescreve |
| 1 | Erro interno ou de ambiente | mantenedor do `mood-ml` |

O código 3 é separado do 1 de propósito: "o dado está errado" e "o validador quebrou" exigem pessoas e ações diferentes, e um *exit code* único obrigaria a ler log para saber qual é.

### 4.4 Alertas e notificações

**Nenhum canal externo.** Sem Slack, Teams, e-mail ou PagerDuty — decisão registrada na §8. O sinal de falha é o *exit code* diferente de zero, o log estruturado em `stderr` e o `status: "rejected"` no report, todos avaliados pelo operador que disparou a execução e está olhando o terminal.

O fundamento é proporcionalidade: alerta serve para quem **não** está olhando. Um pipeline offline, manual, sem SLA e sem plantão não tem esse destinatário — um webhook aqui só somaria um segredo para gerenciar e um caminho a mais por onde conteúdo de mensagem poderia escapar (P5).

**Quando isso deve ser revisto:** se o ingest passar a rodar em CI a cada PR, ou se a origem do corpus se tornar remota. Nos dois casos aparece um executor não-humano, e aí o canal de notificação deixa de ser decoração. Fica registrado como evolução consciente, não como lacuna.

## 5. Logs e observabilidade

### 5.1 Padrão

Log estruturado, **um objeto JSON por linha**, em `stderr` (`--log-format text` dá saída legível para uso interativo). `stdout` carrega apenas o caminho do report e uma linha de resumo, para que a saída seja encadeável em pipe sem ruído.

> Implementação: `common.log` ([spec 10](10-common.md)).

Campos obrigatórios em todo evento:

| Campo | Exemplo | Nota |
|---|---|---|
| `ts` | `2026-09-20T13:02:11.482Z` | ISO-8601 **UTC com sufixo `Z`**, igual ao `sent_at` do contrato |
| `level` | `INFO` | `INFO` \| `WARNING` \| `ERROR` |
| `event` | `phase_completed` | `snake_case`, vocabulário fechado |
| `run_id` | `b7c1f0e4` | correlaciona os eventos de uma execução e aparece no report |
| `corpus_id` | `syn-2026-09-18-a` | `null` antes de F0 concluir |

### 5.2 Eventos e níveis

| Nível | Evento | Carga |
|---|---|---|
| INFO | `ingest_started` | `path`, `flags` |
| INFO | `fingerprint_computed` | `sha256`, `bytes` |
| INFO | `already_validated` | `sha256` — no-op, encerra em 0 |
| INFO | `phase_completed` | `phase`, `records_processed`, `records_rejected`, `elapsed_ms` |
| INFO | `ingest_finished` | `status`, contagens finais, `duration_ms`, `report_path` |
| WARNING | `file_unordered` | `lines_out_of_order` |
| WARNING | `threshold_near_limit` | `rule`, `observed`, `limit` — ex.: nível de rótulo a 10,4% (DC-R10) |
| WARNING | `composition_skipped` | motivo (`--skip-composition`) |
| WARNING | `manual_check_pending` | `DC-R12` |
| WARNING | `io_retry` | `attempt`, `operation`, `errno` |
| ERROR | `validation_error` | `code`, `line`, `message_id`, `detail` (um por erro, teto de 50) |
| ERROR | `corpus_rejected` | `error_count`, `codes` (agregado) |
| ERROR | `io_failed` / `internal_error` | `operation`, `errno` / `exc_type` e *traceback* |

**`records_processed` e `records_rejected` são obrigatórios em toda conclusão de fase.** É a métrica que responde "processou o arquivo todo?" sem abrir o report — a pergunta que se faz primeiro quando um número parece estranho.

### 5.3 `threshold_near_limit`, e por que existe

DC-R10 e DC-R11 exigem ≥ 10%. Um nível a 10,2% **passa**, mas está a uma dúzia de mensagens de reprovar na próxima geração. Emitir *warning* na faixa de 10% a 12% (e simetricamente perto dos limites de DC-R14, 5%–15%) transforma uma reprovação futura em aviso presente. Não altera o veredito.

### 5.4 Observabilidade sem stack de observabilidade

Não há Prometheus, OpenTelemetry nem coletor — coerente com a §3.5. O que existe é o suficiente para responder às perguntas que realmente aparecem:

- *este corpus foi validado, e qual exatamente?* → `status` + `sha256` no report;
- *o que reprovou?* → `errors[]` com código, linha e `message_id`;
- *a distribuição mudou entre duas versões do corpus?* → `diff` de dois `validation_report.json` (motivo pelo qual o JSON é indentado e tem chaves ordenadas: para ser diffável);
- *quanto demorou e quanto processou?* → `duration_ms` e as contagens por fase.

## 6. Requisitos verificáveis

Formato da spec 01: **QUANDO** condição, o script **DEVE** ação. Cada item vira teste em `tests/unit/` ou `tests/contract/`.

**Entrada e portão**
- **IG-R01** QUANDO o caminho não existir, não for legível, não terminar em `.jsonl` ou estiver vazio, o script DEVE encerrar com *exit* 2 e um código `IG_R01_*` da §6.1, sem gravar report.
- **IG-R02** QUANDO o arquivo não estiver sob `data/raw/synthetic/`, o script DEVE recusar a execução (`IG_R02_PATH_OUTSIDE_SYNTHETIC`, *exit* 2). Impede que um snapshot de dado real (data-model §4.2) entre pela trilha sintética.
- **IG-R03** QUANDO o nome do arquivo divergir do `corpus_id` das linhas, ou o `corpus_id` variar entre linhas, o script DEVE rejeitar o corpus.
- **IG-R04** QUANDO o arquivo não for UTF-8 válido ou tiver BOM, o script DEVE rejeitar o corpus (`IG_R04_ENCODING_INVALID`).

**Contrato `dc-1`**
- **IG-R05** Para cada regra DC-R01 a DC-R11 e DC-R14 a DC-R16, o script DEVE emitir o código de erro da §6.1 e DEVE encerrar com *exit* 3.
- **IG-R06** QUANDO houver erro de esquema (F3) ou de integridade (F4), o script DEVE pular F5 e registrar `composition: "not_evaluated"`.
- **IG-R07** QUANDO `--skip-composition` for passado, o script DEVE isentar DC-R09 a DC-R11, DEVE manter todas as outras regras e DEVE registrar a flag no report e um *warning* no log (CA-03 da spec 01).
- **IG-R08** O script DEVE acumular no máximo 50 erros em F3 e, ao atingir o teto, DEVE registrar `errors_truncated: true`.
- **IG-R09** QUANDO as linhas estiverem fora da ordem recomendada, o script DEVE emitir *warning* e DEVE NÃO rejeitar o corpus (§2.3).

**Idempotência**
- **IG-R10** QUANDO o `sha256` for igual ao de um report com `status = ok`, o script DEVE encerrar em 0 sem reler o corpus e sem reescrever o report.
- **IG-R11** QUANDO o `sha256` divergir do report existente, o script DEVE encerrar com *exit* 4 sem sobrescrever nada, incondicionalmente — não existe flag que altere este comportamento.
- **IG-R12** Duas execuções sobre o mesmo arquivo DEVEM produzir reports idênticos, exceto os campos de tempo e `run_id`.

**Leitura pelas etapas seguintes**
- **IG-R13** `load_corpus()` DEVE devolver as mensagens ordenadas por `(conversation_id, sent_at)`.
- **IG-R14** `labels/build.py` DEVE abortar QUANDO não existir report `ok` com `sha256` igual ao do arquivo atual.

**Resiliência e segurança**
- **IG-R15** QUANDO a abertura do corpus ou a gravação do report falhar com `OSError`, o script DEVE retentar 3 vezes com backoff 1s/2s/4s antes de encerrar com *exit* 1.
- **IG-R16** O report DEVE ser gravado por `os.replace()` a partir de arquivo temporário.
- **IG-R17** Nenhum log e nenhum campo do report DEVE conter o valor de `text` ou qualquer trecho dele, inclusive nos erros de PII (P5).
- **IG-R18** O módulo `ingest/` DEVE NÃO importar `duckdb` nem referenciar caminho `.duckdb` (P1) — verificado por teste de *lint* sobre o pacote.
- **IG-R19** O módulo DEVE expor `validate_corpus(path, skip_composition=False) -> ValidationReport` como função pública, importável sem disparar a CLI (§3.1a); a CLI DEVE ser uma casca fina sobre essa função.

### 6.1 Catálogo de erros

Cada código é o contrato de asserção das fixtures exigidas por CA-02 da spec 01. Nome em `SCREAMING_SNAKE_CASE`, prefixado pela regra que o gera — o mesmo padrão de `DC_R02_DUPLICATE_MESSAGE_ID` usado no restante do projeto para citar requisito em código e commit.

| Código | Regra | Fase |
|---|---|---|
| `IG_R01_FILE_NOT_FOUND`, `IG_R01_NOT_READABLE`, `IG_R01_EMPTY_FILE` | IG-R01 | F0 |
| `IG_R02_PATH_OUTSIDE_SYNTHETIC` | IG-R02 | F0 |
| `IG_R03_FILENAME_MISMATCH`, `IG_R03_CORPUS_ID_INCONSISTENT` | IG-R03 | F0/F3 |
| `IG_R04_ENCODING_INVALID` | IG-R04 | F1 |
| `DC_R01_INVALID_JSON` | DC-R01 | F3 |
| `DC_R01_SCHEMA_VIOLATION` (campo ausente, tipo errado, extra, `text` vazio ou > 1000, `sent_at` malformado, `role` inválido, `persona` não-`snake_case`) | DC-R01 | F3 |
| `DC_R07_LABEL_MISSING_OR_INVALID` | DC-R07 | F3 |
| `DC_R08_AGENT_LABEL_NOT_NULL` | DC-R08 | F3 |
| `DC_R02_DUPLICATE_MESSAGE_ID` | DC-R02 | F4 |
| `DC_R03_ID_PREFIX_INVALID` | DC-R03 | F4 |
| `DC_R04_CONVERSATION_MULTI_CUSTOMER` | DC-R04 | F4 |
| `DC_R05_CUSTOMER_MULTI_PERSONA` | DC-R05 | F4 |
| `DC_R06_SENT_AT_DUPLICATE` | DC-R06 | F4 |
| `DC_R09_COMPOSITION_BELOW_MINIMUM` | DC-R09 | F5 |
| `DC_R10_LABEL_LEVEL_UNDERREPRESENTED` | DC-R10 | F5 |
| `DC_R11_PERSONA_UNDERREPRESENTED` | DC-R11 | F5 |
| `DC_R14_PII_RATIO_OUT_OF_RANGE` | DC-R14 | F5 |
| `DC_R15_CONVERSATION_LENGTH_INVALID`, `DC_R15_CONVERSATION_SINGLE_ROLE` | DC-R15 | F5 |
| `DC_R16_MOOD_VARIATION_INSUFFICIENT` | DC-R16 | F5 |
| `IG_R11_HASH_CONFLICT` | IG-R11 | F2 |
| `IG_R15_IO_FAILED` | IG-R15 | F1/F6 |
| `INTERNAL_ERROR` | — (não é violação de regra) | qualquer |

## 7. Segurança e conformidade

### 7.1 Base LGPD/GDPR desta camada

O corpus é **sintético por contrato** (DC-R13): não contém dado pessoal real, a PII presente é fictícia e existe só para exercitar o mascaramento (DC-R14). Em termos de LGPD, **não há titular**, e portanto não incidem base legal, finalidade, prazo de retenção ou direitos de titular sobre este artefato. O mesmo raciocínio *não* vale para `raw/snapshots/` (dado real, pós-MVP), o que é exatamente o que IG-R02 protege ao recusar arquivo fora de `raw/synthetic/`.

**O que o script pode e não pode provar.** DC-R13 é indecidível automaticamente: nenhuma regex distingue um CPF fictício de um real. O que o ingest faz é registrar o fundamento da presunção — `manual_checks.DC-R13: "declared_by_location"` — combinando três garantias estruturais: o arquivo está sob `raw/synthetic/`, todos os IDs têm prefixo `syn-` (DC-R03) e o `corpus_id` segue `syn-AAAA-MM-DD-<letra>`. Declarar a presunção é mais honesto que uma checagem que sugere garantia inexistente.

### 7.2 Mascaramento de PII

**O ingest não mascara.** T2 é de [transform/mask.py](../transform/mask.py) (spec 04), e o dado cru precisa continuar cru para que DC-R14 seja verificável. O ingest apenas **conta**: `pii_ratio` e `pii_by_kind` são agregados — quantidade por tipo, nunca o valor encontrado, nunca o `text`, nunca um trecho ao redor do casamento. Um erro de PII aponta `line` e `message_id`, e quem precisa ver o conteúdo abre o corpus.

Isso é P5 aplicado literalmente: *"logs de erro carregam `message_id`, nunca `text`"*. IG-R17 é a verificação.

### 7.3 Superfície e propriedades dos artefatos

| Artefato | Conteúdo | Pode ser compartilhado? |
|---|---|---|
| `data/raw/synthetic/*.jsonl` | texto das mensagens (sintético) | Não sai do `.gitignore`, por hábito e por uniformidade com a trilha real |
| `data/reports/**/validation_report.json` | contagens, distribuições, hashes, códigos de erro — **zero texto** | Sim. Serve de anexo de TCC e de evidência de qualidade sem expor corpus |
| log em `stderr` | eventos, códigos, IDs | Sim |

Duas propriedades reforçam a postura: a camada **não abre conexão de rede** (§3.1), logo não há exfiltração possível nem segredo a vazar; e o ingest **só escreve** em `data/reports/`, tratando o corpus como somente-leitura — não há caminho de código que altere ou apague o dado de entrada.

**`--review-sample N` (opcional, para DC-R12).** Grava `data/reports/<corpus_id>/review_sample.jsonl` com N mensagens `customer` amostradas por semente fixa, para a conferência manual de 50 mensagens que DC-R12 exige. É o **único** artefato do ingest que contém `text`, e por isso: é opt-in, nunca roda por padrão, só é permitido para corpus sob `raw/synthetic/` (IG-R02), fica no mesmo diretório ignorado pelo Git, e o caminho é registrado em `manual_checks.DC-R12.sample_path` para que a conferência tenha rastro. Em corpus real este recurso seria proibido por P5.

## 8. Decisões deste documento

Registradas para que a próxima leitura não reabra a discussão. Nenhuma exige ADR: são escolhas de implementação dentro do espaço que as ADRs vigentes deixam.

| # | Decisão | Motivo | Consequência assumida |
|---|---|---|---|
| D1 | **Origem:** arquivo local colocado à mão; o ingest só recebe um caminho | Elimina credenciais, rede e retry remoto de uma camada que não precisa deles | Ingestão depende de passo manual do operador, não auditável pelo pipeline |
| D2 | **Saída:** somente `validation_report.json`; nenhum artefato normalizado | Mantém o JSONL como fonte única e evita um artefato intermediário a versionar e invalidar | A ordenação volta a cada leitura, via `load_corpus()` (§2.3), e a trava de IG-R14 passa a ser essencial |
| D3 | **Carga:** *skip* se hash igual, *exit* 4 se divergir, sem exceção | Artefatos são imutáveis por regra do projeto: reprocessamento gera ID novo, nunca sobrescreve | Trocar o conteúdo de um corpus exige `corpus_id` novo — atrito deliberado, sem escape hatch |
| D4 | **Execução:** CLI local; sinal de falha por *exit code* e log | Sem plantão nem SLA, alerta externo não tem destinatário | Nenhuma verificação automática em PR; revisar se o ingest entrar em CI (§4.4) |

## 9. Critérios de aceite

- **CA-01** Corpus em conformidade → *exit* 0 e `validation_report.json` com `status: "ok"` e todos os campos da §3.6 preenchidos (equivale a CA-01 da spec 01).
- **CA-02** Para cada código da §6.1 ligado a uma regra `dc-1`, existe fixture em `tests/fixtures/` que faz o validador falhar **com aquele código**, e não apenas falhar (satisfaz CA-02 da spec 01).
- **CA-03** `tests/fixtures/sample.jsonl` (~50 linhas) passa com `--skip-composition` e reprova sem a flag, com `composition_below_minimum` (CA-03 da spec 01).
- **CA-04** Reexecução sobre arquivo aprovado encerra em 0 sem reescrever o report — verificado por `mtime` inalterado.
- **CA-05** Arquivo alterado após aprovação encerra em 4, sempre, sem exceção — não existe flag que reverta esse resultado. O report existente permanece byte a byte inalterado.
- **CA-06** Teste de privacidade: um corpus com PII fictícia em toda mensagem gera report e log **sem nenhuma ocorrência** dos valores de `text` (busca literal por amostras conhecidas).
- **CA-07** Teste de P1: nenhuma ocorrência de `duckdb` ou `.duckdb` em `ingest/`.
- **CA-08** Corpus embaralhado (mesmo conteúdo, linhas em ordem aleatória) produz o mesmo veredito, as mesmas contagens e *warning* `file_unordered`; `load_corpus()` devolve a mesma sequência ordenada nos dois casos.
- **CA-09** Orçamento da §3.4: corpus sintético de 65.000 linhas valida em < 30 s e < 300 MB de RSS.
- **CA-10** `validate_corpus(path)` é chamável diretamente de um teste `tests/unit/`, sem subprocesso, e devolve um `ValidationReport` cujo conteúdo serializado é idêntico ao `validation_report.json` que a CLI grava para a mesma entrada.

## 10. Fora de escopo

- **Geração do corpus.** Fica com o gerador externo (`dc-1` §8). [ingest/synthetic.py](../ingest/synthetic.py) permanece depreciado.
- **Ingestão de snapshots reais** (`raw/snapshots/`, data-model §4.2). Pós-MVP, com contrato e spec próprios — e com um capítulo de LGPD que a §7.1 deste documento explicitamente não cobre.
- **Transformações T1–T3 e rótulo T4.** Specs 03 e 04.
- **Orquestração, CI e alvos de Makefile.** Spec 09; aqui só se declara a interface de invocação.
- **Conferência manual de DC-R12.** O ingest fornece a amostra e o rastro; o protocolo de anotação é da v2 (`dc-1` §8).
- **Retentativa em fila, quarentena de corpus e saída de quarentena.** A [ADR-0004](../../decisions/ADR-0004-retentativa-e-quarentena.md) trata a trilha *online*; na trilha offline, corpus reprovado simplesmente não entra.

## 11. Checklist de implementação

Tarefas curtas, em ordem de dependência. Cada uma referencia os requisitos que fecha, para citar no commit (ex.: `feat(ingest): valida DC-R01..R08`).

- [ ] `validate_corpus(path, skip_composition=False) -> ValidationReport` — API pública do módulo, chamada pela CLI (IG-R19, §3.1a)
- [ ] CLI (`argparse`) e F0 — resolução de caminho, extensão, diretório (IG-R01, IG-R02)
- [ ] F1 — fingerprint (`sha256`, bytes) em streaming, sem carregar o arquivo inteiro em memória
- [ ] F2 — decisão de idempotência: *skip*, revalidação ou *exit* 4 (IG-R10, IG-R11, IG-R12)
- [ ] F3 — parser de esquema linha a linha com teto de 50 erros (DC-R01, DC-R07, DC-R08, IG-R03, IG-R04, IG-R08)
- [ ] F4 — integridade entre linhas: IDs, prefixo, unicidade (DC-R02 a DC-R06)
- [ ] F5 — composição, distribuição e realismo, com `--skip-composition` (DC-R09 a DC-R11, DC-R14 a DC-R16, IG-R06, IG-R07)
- [ ] F6 — emissão atômica do report e *exit code* (IG-R16, §4.3)
- [ ] `load_corpus()` — leitor canônico ordenado, para uso por `labels/` e `transform/` (IG-R13)
- [ ] Guardião em `labels/build.py`: aborta sem report `ok` correspondente (IG-R14) — coordenar com spec 03
- [ ] Retentativa de I/O com backoff 1s/2s/4s (IG-R15)
- [ ] Log estruturado JSON por evento, `stderr`, com `--log-format text` (§5)
- [ ] `--review-sample` para a amostra de conferência de DC-R12 (§7.3)
- [ ] Fixtures inválidas em `tests/fixtures/`, uma por código de erro da §6.1 (CA-02)
- [ ] Testes `tests/unit/` e `tests/contract/` cobrindo CA-01 a CA-09

# 03 — Construção do Label Set (camada `labels/`, T4)

**Status:** Aceita · **Versão da spec:** `lb-1` · **Data:** 2026-09-20
**Implementa:** [data-model.md §4.3](../../data-structure/data-model.md) (tabela `labels/<label_set_id>.parquet`) — este documento diz **como** T4 produz o que aquela seção descreve.
**Depende de:** ADR-0001 (escala), ADR-0002 (origem do ground truth, grão), ambas em [00-decisoes.md](00-decisoes.md) por remissão; [constitution.md](../../decisions/constitution.md) P1, P4, P5; [02-ingest-validate.md](02-ingest-validate.md) (portão de entrada, `load_corpus()`)
**Consumido por:** `transform/split.py` (T5, spec 05)
**Implementado em:** [labels/build.py](../labels/build.py)

---

## 1. Objetivo geral

Um corpus aprovado pelo ingest (spec 02) mistura mensagens de dois papéis, mas só uma parte dela é alvo de rótulo: as mensagens `role = customer`, cujo humor o modelo aprende a prever (ADR-0002, grão de mensagem). A camada `labels/` tem um propósito único: **separar esse subconjunto rotulável do corpus bruto e expressá-lo no formato de ground truth que o resto do pipeline consome** — `labels/<label_set_id>.parquet`, no esquema de `data-model.md §4.3`.

É uma **extração e remodelagem**, não uma geração de rótulo. O rótulo já existe — nasceu junto com o texto, escrito pelo gerador (ADR-0002: *"o rótulo nasce junto com o dado, e não é inferido depois"*). T4 apenas:

1. filtra as mensagens `customer` de um corpus já validado;
2. remodela cada uma para o esquema de `labels/` (`target_type`, `target_id`, `label_score`, `scale`, `label_source`, `annotator`, `labeled_at`);
3. grava o resultado com um identificador determinístico, `label_set_id = "ls-<corpus_id>"`.

Três não-objetivos delimitam a camada, no mesmo espírito da spec 02:

- **Não valida o contrato do corpus.** Isso já aconteceu no ingest. T4 confia no portão (§2.2, F1) e faz apenas verificação estrutural própria (§3.3), nunca reavalia DC-R01 a DC-R16.
- **Não limpa nem mascara texto.** `labels/` não carrega `text` — só ID e score. T1/T2 são de `transform/` (spec 04) e agem sobre um artefato diferente (`datasets/*`).
- **Não decide o split.** Treino/validação/teste é T5 (`transform/split.py`, spec 05), que junta `labels/` ao corpus por `customer_id`.

## 2. Arquitetura e fluxo de dados

### 2.1 Posição no fluxo

```
data/raw/synthetic/<corpus_id>.jsonl        (aprovado pelo ingest)
data/reports/<corpus_id>/validation_report.json   (status: ok)
       │
       ▼
labels.build  ──►  data/labels/<label_set_id>.parquet
       │              (label_set_id = "ls-<corpus_id>")
       │
       └──►         data/reports/<corpus_id>/label_report.json
                     (contagens, distribuição, sha256)
       │
       ▼
transform/split.py (T5) — junta labels/ ao corpus por customer_id
```

`labels/build.py` **não lê o corpus por conta própria**: usa `load_corpus()`, exportado por `ingest/validate.py` (spec 02, requisito IG-R13). É a mesma razão de lá — uma segunda implementação de leitura e ordenação divergiria da primeira com o tempo. `labels/` reaproveita também a função de *fingerprint* (`sha256` em streaming) do ingest, para o portão da §2.2 F1 usar exatamente o mesmo cálculo que gerou o hash no `validation_report.json`.

### 2.2 Passo a passo lógico

| Fase | Nome | O que faz |
|---|---|---|
| F0 | Resolução da entrada | Extrai `corpus_id` do nome do arquivo, localiza `data/reports/<corpus_id>/validation_report.json` |
| F1 | Portão de qualidade | Confirma `status = "ok"` no report **e** que o `sha256` recalculado do arquivo atual bate com `source.sha256` do report. Qualquer divergência aborta — nada é lido além disso |
| F2 | Carga ordenada | `load_corpus(path)` devolve as mensagens ordenadas por `(conversation_id, sent_at)` |
| F3 | Filtro e remodelagem | Mantém só `role = customer`; mapeia cada mensagem para uma linha de `labels/` (§3.2) |
| F4 | Sanity check | Confere contagem e domínio do resultado contra o que foi lido em F2/F3 (§3.3) |
| F5 | Emissão | Grava `<label_set_id>.parquet` e `label_report.json`, ambos atômicos; loga o resumo; encerra |

**Por que o portão é F1, antes de qualquer leitura de conteúdo.** Rodar T4 sobre um corpus não aprovado (ou alterado depois da aprovação) produziria um label set que parece válido mas descreve um contrato nunca verificado. A dependência de IG-R14 (spec 02) é honrada aqui como responsabilidade de quem consome, não de quem produziu o report: `labels/build.py` refaz o cálculo, não confia cegamente no nome do arquivo.

**Por que F4 existe apesar de ser "óbvio".** Filtrar por `role = customer` e mapear campo a campo é lógica simples o bastante para nunca falhar — e é exatamente por isso que, se falhar, é porque o código de T4 tem um defeito (filtro errado, mapeamento perdendo linha), não porque o dado é ruim. O portão de F1 já garante que o dado é bom; F4 garante que **T4 não o corrompeu**.

### 2.3 Determinismo como propriedade central

Diferente do ingest — onde o hash trava a *identidade* do corpus, mas o report em si carrega timestamps de execução — aqui o determinismo vai um passo além: **o `.parquet` de saída é byte-a-byte idêntico entre execuções**, não só "equivalente". Isso é possível porque `labeled_at` não é o instante em que o script rodou, é o `sent_at` da própria mensagem (§3.3) — o rótulo nasceu com a mensagem (ADR-0002), então sua marca de tempo é a da mensagem, não a da execução do pipeline. Só `label_report.json` carrega timestamps de execução (`run.started_at` etc.), pelo mesmo motivo que o report do ingest carrega.

Essa escolha é o que permite a estratégia de carga da §3.3: **sobrescrever sempre é seguro**, porque duas execuções sobre o mesmo corpus produzem exatamente o mesmo artefato.

## 3. Requisitos técnicos e premissas

### 3.1 Conexões e autenticação

**Não há**, pela mesma razão da spec 02: tudo é arquivo local. A única dependência nova em relação ao ingest é de **ordem de execução**, não de rede ou credencial — `labels/build.py` só produz saída válida depois que `ingest/validate.py` produziu um report `ok` para o mesmo arquivo. Essa dependência é verificada em tempo de execução (F1), não presumida.

```bash
python -m labels.build <caminho-do-corpus.jsonl> \
    [--report-dir data/reports] \
    [--labels-dir data/labels] \
    [--log-format json|text]    # default: json
    [--log-level INFO|WARNING|ERROR]
```

Runtime: Python 3.12, mesmo ambiente do ingest. Dependência nova em relação à spec 02: **`pyarrow`**, para escrever Parquet (já presente em `requirements.txt`). Nenhum outro pacote de dados — `labels/` não precisa de `pandas` para montar 8 colunas simples; `pyarrow.Table.from_pylist()` basta, e evita uma dependência que só entraria por conveniência.

Invocação pelo orquestrador (spec 09): `make labels CORPUS=<corpus_id>` e `python -m pipeline labels --config configs/pipeline.yaml`.

### 3.2 Formato dos dados

| | Entrada | Saída principal | Saída de auditoria |
|---|---|---|---|
| Formato | JSON Lines (via `load_corpus()`) | Parquet | JSON |
| Caminho | `data/raw/synthetic/<corpus_id>.jsonl` | `data/labels/<label_set_id>.parquet` | `data/reports/<corpus_id>/label_report.json` |
| Esquema | `dc-1` (spec 01) | data-model §4.3 | §3.6 deste documento |
| Grão | 1 linha = 1 mensagem (`customer` + `agent`) | 1 linha = 1 mensagem `customer` rotulada | 1 arquivo = 1 execução |

**Esquema de saída do Parquet**, exatamente as colunas de `data-model.md §4.3` — nenhuma a mais:

| Coluna | Tipo | Nulo? | Origem |
|---|---|---|---|
| `label_set_id` | string | não | `"ls-" + corpus_id` |
| `target_type` | string | não | constante `"message"` (ADR-0002) |
| `target_id` | string | não | `message_id` da mensagem |
| `label_score` | float64 | não | `generated_label` da mensagem |
| `scale` | string | não | constante `"-1 to 1"` (ADR-0001) |
| `label_source` | string | não | constante `"synthetic"` (ADR-0002) |
| `annotator` | string | sim | sempre `null` no MVP |
| `labeled_at` | string ISO-8601 UTC | não | `sent_at` da própria mensagem (§2.3) |

`persona` e `customer_id` **não aparecem aqui**, embora estejam disponíveis no corpus. Pertencem à junção que T5 faz em `datasets/*` (data-model §4.3, segunda tabela), não ao label set — que por design é **agnóstico à fonte do rótulo e ao cliente**, para que trocar `label_source` no futuro (`manual`, `csat`) não exija tocar neste esquema (ADR-0002).

Ordem das linhas: a mesma de `load_corpus()` — `(conversation_id, sent_at)` — preservada, nunca reordenada por `target_id` ou outro critério. Mantém o Parquet diffável contra execuções anteriores.

### 3.3 Estratégia de carga e idempotência

**Overwrite determinístico**, sem verificação de hash prévia e sem flag de confirmação. `label_set_id` é uma função pura de `corpus_id` (`"ls-" + corpus_id`), e `corpus_id` já está travado por conteúdo no ingest (spec 02, IG-R11): não existe cenário em que o mesmo `label_set_id` precise representar dois conteúdos diferentes, porque essa garantia já foi imposta uma vez, a montante. Reimpor a mesma trava aqui seria redundante — a existência de um `validation_report.json` com `status: ok` e hash coincidente **já é** a prova de imutabilidade de que T4 precisa.

| Situação | Comportamento | Exit |
|---|---|---|
| Portão aprova (F1 ok) | Constrói e **sobrescreve** `<label_set_id>.parquet` e `label_report.json`, sempre | 0 |
| Portão reprova (sem report `ok`, ou hash divergente) | Aborta antes de ler o corpus. Não escreve nada | 3 |
| Sanity check falha (F4) | Aborta depois de montar o label set em memória, mas **antes** de gravar qualquer arquivo | 4 |

**Por que overwrite é seguro aqui e não era no ingest.** No ingest, o hash era a única defesa contra um `corpus_id` mudar de conteúdo em silêncio — por isso divergência lá é fatal (spec 02 §3.3). Aqui, a entrada já passou por essa defesa antes de chegar; T4 é uma função determinística sobre um dado que **não pode mais mudar sob o mesmo nome**. Sobrescrever o resultado de uma função pura com o resultado da mesma função é uma operação sem risco de perda — o novo arquivo é idêntico ao antigo, exceto se o *código* de T4 mudou (ex.: bug corrigido), caso em que sobrescrever é exatamente o comportamento desejado.

**Idempotência real.** Duas execuções sobre o mesmo corpus aprovado produzem o mesmo `.parquet` byte a byte (§2.3). O `label_report.json` repete o padrão do ingest: idêntico exceto os campos de tempo da execução.

### 3.4 Volumetria estimada

Herda a volumetria do corpus (spec 02 §3.4), porque a saída é 1:1 com as mensagens `customer`:

| Grandeza | Piso do contrato | Esperado | Teto de projeto (10×) |
|---|---|---|---|
| Linhas do label set (= mensagens `customer`) | 2.000 | 2.000–4.000 | 40.000 |
| Tamanho do `.parquet` | — | **~150–400 KB** | ~3–4 MB |
| Tamanho do `label_report.json` | — | poucos KB | poucos KB |

O Parquet é bem menor que o JSONL de origem: oito colunas curtas, três delas constantes por arquivo inteiro (`label_set_id`, `scale`, `label_source`), o que a codificação em dicionário do Parquet comprime de forma eficiente. Orçamento de desempenho: < 3 s de parede no corpus esperado, < 15 s no teto de 10×, folgado o bastante para não exigir processamento em lotes.

### 3.5 Frequência de execução

**Sob demanda, imediatamente após um `ingest.validate` bem-sucedido** sobre o mesmo arquivo. Mesmo padrão da spec 02 (§3.5): sem agendador, disparado manualmente ou como parte de `make labels` / `make all`. Não há frequência própria a definir além de "sempre que o corpus de entrada mudar" — e como o portão (F1) impede rodar sobre corpus não aprovado, a cadência de `labels/build.py` está estruturalmente amarrada à do ingest.

### 3.6 Esquema do `label_report.json`

```json
{
  "report_version": "lb-1",
  "run": {
    "run_id": "a13e9f02",
    "started_at": "2026-09-20T14:10:03Z",
    "finished_at": "2026-09-20T14:10:04Z",
    "duration_ms": 812,
    "tool": "labels.build",
    "python": "3.12.3"
  },
  "source": {
    "corpus_id": "syn-2026-09-18-a",
    "corpus_path": "data/raw/synthetic/syn-2026-09-18-a.jsonl",
    "corpus_sha256": "9f2b...c41d",
    "ingest_report_path": "data/reports/syn-2026-09-18-a/validation_report.json"
  },
  "label_set": {
    "label_set_id": "ls-syn-2026-09-18-a",
    "path": "data/labels/ls-syn-2026-09-18-a.parquet",
    "sha256": "77ac...0e19",
    "rows": 2410
  },
  "counts": {
    "messages_read": 4812,
    "messages_customer": 2410,
    "messages_agent_excluded": 2402
  },
  "distributions": {
    "label_levels": { "-1.0": 402, "-0.5": 498, "0.0": 523, "0.5": 511, "1.0": 476 }
  }
}
```

O report só é gravado **depois** que o `.parquet` foi escrito com sucesso (§4.2) — sua simples existência com este esquema já é evidência de que a construção terminou. Não existe um `status: "rejected"` aqui: falhas de portão (F1) e de sanity check (F4) encerram por *exit code* e log, sem produzir report, porque não há "label set parcialmente construído" que valha registrar — é tudo ou nada.

## 4. Tratamento de erros e resiliência

### 4.1 Taxonomia

Mesma distinção da spec 02, adaptada:

- **Erro de portão** (F1) — o corpus não está pronto para T4. Não é uma falha do script, é uma dependência de ordem não satisfeita. *Exit* 3.
- **Erro de integridade interna** (F4) — o próprio T4 tem um defeito. É o único caso, nesta camada, em que "o script está errado" é a explicação mais provável, porque o dado de entrada já foi validado. *Exit* 4.
- **Erro de uso** — CLI mal chamada. *Exit* 2.
- **Erro de ambiente** — I/O. A única categoria onde retentar ajuda. *Exit* 1.

### 4.2 Política de retentativa

**Sobre a construção do label set: nenhuma.** É determinística (§2.3); reexecutar sobre o mesmo corpus aprovado dá o mesmo resultado. Um portão reprovado (*exit* 3) não se resolve tentando de novo — resolve-se corrigindo o ingest ou o corpus. Uma falha de sanity check (*exit* 4) não se resolve tentando de novo — resolve-se corrigindo `labels/build.py`.

**Sobre I/O: 3 tentativas, backoff 1s/2s/4s.** As mesmas duas categorias de operação da spec 02 — abrir arquivo para leitura, gravar arquivo — aplicadas aos três alvos desta camada: reler o corpus (via `load_corpus()`), ler o `validation_report.json` do portão, e escrever `.parquet` + `label_report.json`.

**Gravação atômica, em duas etapas ordenadas.** Primeiro o `.parquet` (`os.replace()` a partir de `.tmp`), depois o `label_report.json`. Se a escrita do report falhar após o parquet já ter sido substituído, uma nova execução resolve os dois de uma vez — o parquet será sobrescrito pelo mesmo conteúdo (determinismo, §2.3) e o report finalmente será gravado. Não existe janela em que o parquet fique inconsistente com o corpus que o gerou, porque o parquet é sempre a última coisa que muda antes do report confirmar.

### 4.3 *Exit codes*

| Exit | Significado | Quem age |
|---|---|---|
| 0 | Label set construído (ou reconstruído de forma idêntica) | ninguém |
| 2 | Erro de uso da CLI | operador corrige o comando |
| 3 | **Portão bloqueado** — corpus sem `validation_report.json` `ok` correspondente | operador roda (ou corrige) o ingest primeiro |
| 4 | **Sanity check falhou** — contagem ou domínio do label set não bate com o corpus lido | mantenedor do `mood-ml` corrige `labels/build.py` |
| 1 | Erro interno ou de ambiente | mantenedor do `mood-ml` |

O *exit* 4 é deliberadamente distinto do *exit* 3: um aponta para o corpus (age quem gera dado), o outro aponta para o código de T4 (age quem mantém o pipeline). Confundir os dois faria alguém "consertar" um corpus que já está correto.

### 4.4 Alertas e notificações

**Nenhum canal externo**, pela mesma razão da spec 02 §4.4: pipeline offline, sob demanda, sem plantão. O sinal de falha é o *exit code* e o log em `stderr`, avaliados por quem disparou a execução. Revisar apenas se a camada inteira migrar para CI ou execução não-supervisionada (mesma condição de revisão da spec 02).

## 5. Logs e observabilidade

### 5.1 Padrão

Idêntico ao do ingest (spec 02 §5.1): um objeto JSON por linha em `stderr`, `stdout` reservado ao caminho dos artefatos gerados e a um resumo de uma linha. Campos obrigatórios: `ts` (ISO-8601 UTC, sufixo `Z`), `level`, `event`, `run_id`, `corpus_id`.

### 5.2 Eventos e níveis

| Nível | Evento | Carga |
|---|---|---|
| INFO | `build_started` | `corpus_path` |
| INFO | `gate_checked` | `ingest_report_path`, `sha256_match: true` |
| INFO | `corpus_loaded` | `messages_read` |
| INFO | `label_rows_built` | `messages_customer`, `messages_agent_excluded` |
| INFO | `sanity_check_passed` | `rows`, `checks: ["row_count", "label_domain", "target_id_unique"]` |
| INFO | `build_finished` | `label_set_id`, `rows`, `duration_ms`, `parquet_path`, `report_path` |
| WARNING | `io_retry` | `attempt`, `operation`, `errno` |
| ERROR | `gate_blocked` | `reason` (`missing_report` \| `status_not_ok` \| `sha256_mismatch`) |
| ERROR | `sanity_check_failed` | `check`, `expected`, `observed` |
| ERROR | `io_failed` / `internal_error` | `operation`, `errno` / `exc_type` e *traceback* |

`gate_blocked` e `sanity_check_failed` são os dois eventos que justificam a existência desta camada como um componente auditável, e não uma função invocada silenciosamente dentro do split: cada um aponta para uma causa raiz diferente (corpus vs. código), com o mesmo raciocínio da tabela de *exit codes* (§4.3).

### 5.3 Observabilidade

As mesmas três perguntas da spec 02, adaptadas:

- *este label set corresponde a qual corpus, exatamente?* → `source.corpus_sha256` no `label_report.json`, e ele é o mesmo hash que aparece no `validation_report.json` do ingest — rastreável em dois arquivos, não em um só;
- *a contagem bate?* → `counts.messages_customer` do `label_report.json` DEVE ser igual a `counts.messages_customer` do `validation_report.json` do mesmo `corpus_id` (uma verificação de consistência que um teste de contrato pode automatizar, além do sanity check em tempo de execução);
- *quanto demorou, quanto processou?* → `run.duration_ms` e `label_set.rows`.

## 6. Segurança e conformidade

### 6.1 Superfície de PII: estruturalmente menor que a do ingest

`labels/` não tem campo `text` no esquema (§3.2) — nem mascarado, nem cru. Não é uma política aplicada em tempo de execução, é uma **impossibilidade do esquema**: uma coluna que não existe não pode vazar. `target_id` é `message_id`, já pseudônimo por construção do contrato `dc-1` (prefixo `syn-`). `label_report.json` carrega apenas contagens, hashes e IDs.

Isso torna a camada mais simples que o ingest no quesito P5: lá foi preciso declarar e verificar ausência de PII (spec 02 §7); aqui a ausência decorre diretamente do requisito de esquema fechado (LB-R07, §7).

### 6.2 Base LGPD/GDPR

Mesma base da spec 02 §7.1: o corpus de origem é sintético, sem titular real. `labels/` herda essa presunção sem acrescentar superfície nova — não introduz nenhum dado que não estivesse já no corpus, e remove a maior parte dele (`text`, `persona`, `customer_id`). Não há mascaramento a fazer nesta camada porque não há o que mascarar.

### 6.3 P1 e P4

Nenhuma referência a `duckdb` ou caminho `.duckdb` (P1) — mesma verificação de *lint* da spec 02, aplicada a `labels/`. P4 não se aplica diretamente (T4 não é T1/T3), mas a disciplina de fonte única se estende por analogia: `labels/build.py` reaproveita `load_corpus()` e a função de *fingerprint* do ingest, em vez de duplicá-las.

## 7. Requisitos verificáveis

Formato da spec 01/02: **QUANDO** condição, o script **DEVE** ação.

**Portão**
- **LB-R01** QUANDO não existir `data/reports/<corpus_id>/validation_report.json` com `status = "ok"`, o script DEVE encerrar com *exit* 3 sem ler o corpus.
- **LB-R02** QUANDO o `sha256` recalculado do arquivo de entrada divergir de `source.sha256` no report do ingest, o script DEVE encerrar com *exit* 3.
- **LB-R03** O recálculo do `sha256` DEVE reusar a mesma função do ingest (spec 02), não uma reimplementação.

**Construção**
- **LB-R04** O script DEVE consumir o corpus exclusivamente via `load_corpus()` do ingest, nunca abrindo o `.jsonl` por conta própria.
- **LB-R05** O script DEVE criar exatamente uma linha de label por mensagem `role = "customer"` e nenhuma linha para `role = "agent"`.
- **LB-R06** `label_set_id` DEVE ser `"ls-" + corpus_id`, sem variação entre execuções sobre o mesmo `corpus_id`.
- **LB-R07** O esquema de saída DEVE conter exatamente as colunas de `data-model.md §4.3` (§3.2 deste documento) — nenhuma coluna adicional, em especial `persona` e `customer_id`.
- **LB-R08** `labeled_at` DEVE ser copiado do `sent_at` da mensagem de origem, nunca do relógio da execução.
- **LB-R09** A ordem das linhas no `.parquet` DEVE ser a mesma devolvida por `load_corpus()`.

**Sanity check**
- **LB-R10** QUANDO o número de linhas do label set divergir do número de mensagens `role = "customer"` lidas do corpus, o script DEVE encerrar com *exit* 4 sem gravar nenhum artefato.
- **LB-R11** QUANDO algum `label_score` estiver fora de `{-1.0, -0.5, 0.0, 0.5, 1.0}` ou algum `target_id` estiver duplicado, o script DEVE encerrar com *exit* 4 sem gravar nenhum artefato.

**Idempotência**
- **LB-R12** Duas execuções sobre o mesmo corpus aprovado DEVEM produzir `.parquet` byte a byte idênticos.
- **LB-R13** O script DEVE sobrescrever `.parquet` e `label_report.json` existentes a cada execução bem-sucedida, sem exigir flag.

**Resiliência e segurança**
- **LB-R14** QUANDO a leitura ou a escrita de artefatos falhar com `OSError`, o script DEVE retentar 3 vezes com backoff 1s/2s/4s antes de encerrar com *exit* 1.
- **LB-R15** `.parquet` e `label_report.json` DEVEM ser gravados atomicamente (temporário + `os.replace()`), o `.parquet` sempre antes do report.
- **LB-R16** O `label_report.json` DEVE ser gravado somente após o `.parquet` ter sido escrito com sucesso.
- **LB-R17** Nenhum campo do label set ou do `label_report.json` DEVE conter `text` (P5) — verificável por construção do esquema (LB-R07).
- **LB-R18** O módulo `labels/` DEVE NÃO importar `duckdb` nem referenciar caminho `.duckdb` (P1).

### 7.1 Catálogo de erros

| Código | Requisito | Fase |
|---|---|---|
| `LB_R01_GATE_MISSING_REPORT` | LB-R01 | F1 |
| `LB_R01_GATE_STATUS_NOT_OK` | LB-R01 | F1 |
| `LB_R02_GATE_SHA256_MISMATCH` | LB-R02 | F1 |
| `LB_R10_ROW_COUNT_MISMATCH` | LB-R10 | F4 |
| `LB_R11_LABEL_SCORE_OUT_OF_DOMAIN` | LB-R11 | F4 |
| `LB_R11_DUPLICATE_TARGET_ID` | LB-R11 | F4 |
| `LB_R14_IO_FAILED` | LB-R14 | F1/F2/F5 |
| `INTERNAL_ERROR` | — | qualquer |

## 8. Decisões deste documento

| # | Decisão | Motivo | Consequência assumida |
|---|---|---|---|
| D1 | **Carga:** overwrite determinístico, sem verificação de hash própria | `corpus_id` já é imutável por construção do ingest; reimpor a trava seria redundante | Se o ingest um dia relaxar sua própria trava, esta decisão precisa ser revisitada |
| D2 | **Auditoria:** `label_report.json` ao lado do `.parquet`, só gravado em caso de sucesso | Espelha o `validation_report.json` e permite conferir T4 e diffar execuções sem abrir o Parquet | Falha de portão ou de sanity check não deixa rastro em arquivo, só em log — aceitável porque não há "construção parcial" a documentar |
| D3 | **Sanity check bloqueante** (contagem e domínio) | Divergência só pode significar bug em T4, já que o portão garante a qualidade do dado de entrada | Um custo fixo pequeno de CPU a cada execução, em troca de nunca propagar um label set incompleto para o split |
| D4 | **`labeled_at` = `sent_at` da mensagem**, não o instante da execução | Torna o `.parquet` determinístico byte a byte, coerente com "o rótulo nasce com o dado" (ADR-0002) | `label_report.json` é o único lugar com timestamp de execução real |

## 9. Critérios de aceite

- **CA-01** Corpus aprovado → *exit* 0, `<label_set_id>.parquet` com o esquema exato da §3.2 e `label_report.json` com todos os campos da §3.6.
- **CA-02** Corpus sem `validation_report.json` `ok`, ou com hash divergente → *exit* 3, nenhum arquivo escrito em `data/labels/` nem `label_report.json` novo.
- **CA-03** Duas execuções sucessivas sobre o mesmo corpus aprovado produzem `.parquet` idênticos byte a byte (`sha256` igual).
- **CA-04** `counts.messages_customer` do `label_report.json` é igual ao `counts.messages_customer` do `validation_report.json` do mesmo `corpus_id`.
- **CA-05** Teste de regressão do sanity check: uma versão de `labels/build.py` deliberadamente quebrada (ex.: inclui mensagens `agent`) falha em CA de teste com `LB_R10_ROW_COUNT_MISMATCH`, sem gravar `.parquet`.
- **CA-06** Nenhuma coluna fora do esquema da §3.2 aparece no Parquet gerado — testado por comparação de conjunto de colunas.
- **CA-07** Teste de P1: nenhuma ocorrência de `duckdb` ou `.duckdb` em `labels/`.
- **CA-08** Teste de P5: busca literal por qualquer `text` de mensagem do corpus de teste dentro do `.parquet` e do `label_report.json` não encontra ocorrência.

## 10. Fora de escopo

- **Validação do contrato `dc-1`.** Já feita pelo ingest (spec 02). T4 apenas verifica o portão (F1), nunca reavalia DC-R01 a DC-R16.
- **Limpeza e mascaramento de texto (T1/T2).** `labels/` não carrega texto. Specs 04.
- **Junção com `persona`/`customer_id` e split treino/validação/teste (T5).** Spec 05 — inclusive o portão equivalente que `transform/split.py` deveria ter sobre `label_report.json` (mesmo raciocínio de IG-R14/LB-R01), a ser formalizado naquela spec.
- **Outras fontes de rótulo** (`manual`, `heuristic`, `csat`). Fora do MVP por ADR-0002; quando existirem, geram novo `label_set_id` e não tocam este esquema.
- **Orquestração, CI e alvos de Makefile.** Spec 09.

## 11. Checklist de implementação

- [ ] CLI (`argparse`) e F0 — resolução de `corpus_id` a partir do nome do arquivo
- [ ] F1 — portão: leitura do `validation_report.json`, recomputação de hash reaproveitando a função do ingest (LB-R01, LB-R02, LB-R03)
- [ ] F2 — carga via `load_corpus()` (LB-R04)
- [ ] F3 — filtro `role = customer` e remodelagem para o esquema de `labels/` (LB-R05 a LB-R09)
- [ ] F4 — sanity check de contagem, domínio e unicidade (LB-R10, LB-R11)
- [ ] F5 — escrita atômica do `.parquet` e do `label_report.json`, nesta ordem (LB-R15, LB-R16)
- [ ] Retentativa de I/O com backoff 1s/2s/4s (LB-R14)
- [ ] Log estruturado JSON por evento (§5)
- [ ] Fixtures: corpus aprovado válido, corpus sem report, corpus com hash divergente, `build.py` quebrado propositalmente para CA-05
- [ ] Testes `tests/unit/` e `tests/contract/` cobrindo CA-01 a CA-08

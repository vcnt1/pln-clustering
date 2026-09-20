# 05 — Reconstrução de Janela, Montagem de Exemplos e Split (camada `transform/split.py`, T5 offline)

**Status:** Proposta · **Versão da spec:** `sp-1` · **Data:** 2026-09-20
**Implementa:** [data-model.md §4.3](../../data-structure/data-model.md) (tabela `datasets/<dataset_id>/`) — este documento diz **como** o corpus validado e o label set viram o dataset que o treino consome.
**Depende de:** [ADR-0007](../../decisions/ADR-0007-humor-por-conversa.md) (janela de 30 mensagens da conversa), [ADR-0008](../../decisions/ADR-0008-abordagens-de-modelo.md) (dois blocos, `dataset_id` compartilhado entre abordagens); [constitution.md](../../decisions/constitution.md) P2, P4, P5; [02-ingest-validate.md](02-ingest-validate.md) (portão do corpus, `load_corpus()`); [03-labels.md](03-labels.md) (portão do label set); [04-transform.md](04-transform.md) (`extract_features`, único ponto de montagem)
**Consumido por:** `train/train.py` (spec 06)
**Implementado em:** [transform/split.py](../transform/split.py)

---

## 1. Objetivo geral

Duas responsabilidades que só fazem sentido juntas, porque a segunda opera sobre a saída da primeira:

1. **Montar um exemplo de treino por mensagem rotulada.** Para cada linha do label set (spec 03), reconstruir a janela de histórico que a API montaria em produção (ADR-0007), chamar `extract_features` (spec 04) sobre ela, e juntar o resultado com `label_score`, `persona` e `customer_id` — produzindo a tabela intermediária que a `data-model.md §4.3` descreve.
2. **Particionar esses exemplos em treino/validação/teste**, agrupados por `customer_id`, para que o modelo nunca veja no teste um cliente que apareceu no treino (ADR-0002: com ground truth sintético, personas repetidas entre splits vazam o padrão do gerador e a métrica passa a medir memorização, não desempenho).

É a última camada offline antes do treino em si — e a única, em todo o pipeline, que executa T5 (a reconstrução da janela) **fora** da API. `transform/split.py` não inventa uma regra própria para isso: reconstrói exatamente a janela que a ADR-0007 define, a partir do corpus, para que o exemplo de treino e a requisição de inferência online sejam **a mesma distribuição de entrada** vista pelo modelo.

## 2. Arquitetura e fluxo de dados

### 2.1 Posição no fluxo

```
data/raw/synthetic/<corpus_id>.jsonl          (aprovado, spec 02)
data/reports/<corpus_id>/validation_report.json
data/labels/ls-<corpus_id>.parquet            (aprovado, spec 03)
data/reports/<corpus_id>/label_report.json
       │
       ▼
transform.split  ──►  data/datasets/<dataset_id>/
                          train.parquet
                          validation.parquet
                          test.parquet
                          dataset.json         (também serve de relatório de auditoria)
       │
       ▼
train/train.py (spec 06)
```

### 2.2 Passo a passo lógico

| Fase | Nome | O que faz |
|---|---|---|
| F0 | Resolução da entrada | Recebe `corpus_id` (ou caminho do corpus) e `dataset_id` (obrigatório, fornecido pelo operador) |
| F1 | Portão duplo | Confirma que **o corpus** está aprovado (spec 02) e que **o label set** está aprovado e corresponde a esse corpus (spec 03). Qualquer um reprovado, aborta antes de ler conteúdo |
| F2 | Identidade do `dataset_id` | Calcula o *fingerprint* das entradas + configuração; compara com um `dataset.json` pré-existente, se houver (§3.3) |
| F3 | Carga | `load_corpus()` (corpus, ordenado) + leitura do `.parquet` de labels |
| F4 | Reconstrução da janela (T5 offline) | Para cada linha do label set, monta o `history` da mesma forma que a API montaria (§3.4) |
| F5 | Montagem do exemplo | Chama `extract_features(history)` (spec 04); junta `label_score`, `persona`, `customer_id`, `conversation_id` |
| F6 | Sanity check | Confere contagens e ausência de vazamento entre splits (§3.3, §7) |
| F7 | Split | `GroupShuffleSplit` em duas etapas, agrupado por `customer_id` (§3.4) |
| F8 | Emissão | Grava os três `.parquet` e `dataset.json`, atômicos; loga o resumo; encerra |

### 2.3 Por que o portão é duplo

A spec 03 já garante que `labels/build.py` não roda sobre um corpus não aprovado (LB-R01). Mas `transform/split.py` **relê o corpus por conta própria** — para reconstruir a janela precisa do texto de todas as mensagens `customer`, não só das rotuladas — e também **relê o label set**. Cada leitura é uma nova oportunidade de operar sobre um arquivo que mudou de conteúdo depois de aprovado (ex.: alguém sobrescreveu o `.parquet` de labels manualmente). Por isso F1 verifica os dois portões de novo, com o mesmo raciocínio da spec 03 (§2.2): a garantia de qualidade a montante não dispensa quem lê depois de verificá-la.

## 3. Requisitos técnicos e premissas

### 3.1 Conexões e autenticação

**Não há**, mesma razão das specs anteriores. A novidade aqui é o número de dependências de arquivo local que o portão precisa checar — quatro, em vez de uma: o corpus, o report do ingest, o `.parquet` de labels e o report de labels.

```bash
python -m transform.split <caminho-do-corpus.jsonl> \
    --dataset-id ds-AAAA-MM-DD-<letra> \
    [--config configs/pipeline.yaml] \
    [--datasets-dir data/datasets] \
    [--log-format json|text]
    [--log-level INFO|WARNING|ERROR]
```

Runtime: Python 3.12. Dependências novas em relação às specs 02–04: **`scikit-learn`** (`GroupShuffleSplit`), já em `requirements.txt`.

Invocação pelo orquestrador (spec 09): `make split CORPUS=<corpus_id> DATASET=<dataset_id>` e `python -m pipeline split --config configs/pipeline.yaml`.

### 3.2 Formato dos dados

| | Entrada | Saída |
|---|---|---|
| Formato | JSONL (via `load_corpus()`) + Parquet (labels) | 3× Parquet + 1× JSON |
| Caminhos | `data/raw/synthetic/<corpus_id>.jsonl`, `data/labels/ls-<corpus_id>.parquet` | `data/datasets/<dataset_id>/{train,validation,test}.parquet`, `data/datasets/<dataset_id>/dataset.json` |
| Esquema | `dc-1` (spec 01) + `labels/*` (spec 03) | `data-model.md §4.3` (tabela `datasets/*`) |

**Esquema de cada linha**, idêntico nos três arquivos (a coluna `split` é redundante com o arquivo em que a linha está, mas mantida por fidelidade ao `data-model.md §4.3` — cada linha permanece autodescritiva se algum dia os três forem concatenados para análise):

| Coluna | Tipo | Origem |
|---|---|---|
| `example_id` | string | = `target_id` do label set (= `message_id`) |
| `customer_id` | string | do corpus, mensagem disparadora |
| `conversation_id` | string | do corpus, mensagem disparadora |
| `persona` | string | do corpus, mensagem disparadora |
| `text_clean` | string | `extract_features(history)["text_clean"]` |
| `context_clean` | lista de string | `extract_features(history)["context_clean"]` |
| `label_score` | float64 | do label set, por `target_id` |
| `split` | string | `"train"` \| `"validation"` \| `"test"` |

**`dataset.json` não tem um relatório irmão.** Ao contrário das specs 02 e 03, aqui não existe um `*_report.json` separado — o próprio `dataset.json` (exigido pelo `data-model.md §4.3`: `dataset_id`, `source_ids`, `label_set_id`, `feature_spec_version`, `split_strategy`, `row_counts`, `created_at`) já cumpre o papel de artefato de auditoria. Criar um segundo arquivo só para repetir essas contagens seria redundância sem função — diferente de `labels/`, onde o `.parquet` sozinho não conseguia carregar metadados de proveniência sem violar o esquema fechado da `data-model.md`.

### 3.3 Estratégia de carga e idempotência

`dataset_id` não vem travado por hash de um passo anterior, como `label_set_id` estava (spec 03 §3.3) — é uma identidade nova, escolhida pelo operador na convenção `ds-AAAA-MM-DD-<letra>` (mesmo padrão do `corpus_id`). Isso reabre exatamente o problema que a spec 02 resolveu para o corpus: **o mesmo `dataset_id` pode, por engano, ser reaproveitado para entradas diferentes.**

A solução é um *fingerprint* determinístico das entradas, calculado sempre, e comparado contra o `dataset.json` existente (se houver) antes de escrever qualquer coisa:

```
fingerprint = sha256(canonical_json({
    "corpus_sha256": <do validation_report.json>,
    "label_set_sha256": <do label_report.json>,
    "split_strategy": {"train_ratio", "validation_ratio", "test_ratio", "seed"},
    "feature_spec_version": "fs-1",
}))
```

| Situação | Comportamento | Exit |
|---|---|---|
| Portão aprova (F1), `dataset_id` inédito | Constrói e grava | 0 |
| Portão aprova, `dataset_id` existente, *fingerprint* **igual** | Reconstrói e **sobrescreve** de forma determinística — mesmo raciocínio de `labels/` (spec 03 §3.3): entradas iguais, `GroupShuffleSplit` com seed fixa, resultado byte a byte igual | 0 |
| Portão aprova, `dataset_id` existente, *fingerprint* **diferente** | **Aborta, sem exceção.** O `dataset_id` já significa outra coisa — resolução é escolher um `dataset_id` novo, nunca sobrescrever | 4 |
| Portão reprova (F1) | Aborta antes de qualquer leitura de conteúdo | 3 |

Isto é a mesma regra de imutabilidade de artefato já aplicada nas specs 02 e 03 (reprocessamento gera um ID novo, nunca sobrescreve um existente com conteúdo diferente), adaptada ao único ponto desta camada em que uma identidade nova é criada, em vez de herdada.

**Por que reaproveitar é seguro quando o *fingerprint* bate.** `GroupShuffleSplit` com `random_state` fixo é uma função determinística dos seus dados de entrada — mesmas linhas, mesmo agrupamento por `customer_id`, mesma seed, sempre produz a mesma partição. Combinado com `extract_features` sendo pura (spec 04), o pipeline inteiro de F3 a F7 é uma função pura de (`corpus`, `label set`, configuração). Não há razão para gerar um `dataset_id` novo por reexecutar a mesma função sobre a mesma entrada — só há razão para um `dataset_id` novo quando a entrada **muda de fato** (novo corpus, novo label set, ou configuração de split diferente), e nesse caso o operador já escolheria um `dataset_id` novo por convenção, não reaproveitaria o antigo.

### 3.4 Estratégia de transformações

#### Reconstrução da janela (T5 offline) — algoritmo

A regra é a da ADR-0007: até 30 mensagens `role = customer` **da mesma conversa**, ordem crescente, a disparadora por último, sem atravessar `conversation_id`. Como `load_corpus()` já devolve o corpus ordenado por `(conversation_id, sent_at)` (spec 02, IG-R13), a reconstrução é uma **janela deslizante de tamanho 30 por conversa**, em uma única passada:

```
para cada conversation_id, em ordem:
    buffer = []  # até 30 HistoryMessage, só role=customer
    para cada mensagem da conversa, em ordem crescente de sent_at:
        se role == "agent": continuar (não entra no buffer, ADR-0007)
        buffer.append(mensagem)
        se len(buffer) > 30: buffer.pop(0)  # mantém só as 30 mais recentes
        # neste ponto, buffer É o history da mensagem atual como disparadora
        history_desta_mensagem = list(buffer)
```

Isto evita reconstruir cada janela do zero (que seria O(*n* × 30) em leituras redundantes do mesmo texto) sem precisar de nenhuma estrutura além da ordem que `load_corpus()` já garante. `extract_features` ainda é chamada uma vez por mensagem disparadora, com sua própria janela de até 30 itens — a economia está em montar essas janelas, não em pular chamadas a `extract_features`.

**Decisão explícita: sem memoização de T1/T2 entre chamadas.** A spec 04 (§3.5) deixou em aberto se `split.py` deveria cachear o resultado de `clean_message`/`mask_pii` por `message_id`, já que a mesma mensagem de contexto é reprocessada em até 29 janelas diferentes. No pior caso do teto de volumetria (spec 02 §3.4, 40.000 mensagens `customer`), isso significa até ~1,2 milhão de aplicações de T1+T2 sobre strings de até 1000 caracteres — da ordem de poucos minutos de CPU em Python puro, aceitável para uma execução offline sob demanda. Memoização é uma otimização legítima, mas fica **fora desta versão**: adicionaria uma estrutura de cache e uma nova superfície de bug (cache por `message_id` desatualizado, por exemplo) para resolver um problema de desempenho que, nos números do MVP, não é um problema.

#### Split — `GroupShuffleSplit` em duas etapas

Chaves em `configs/pipeline.yaml`, seção `split`:

```yaml
split:
  train_ratio: 0.70
  validation_ratio: 0.15
  test_ratio: 0.15
  seed: 42
```

Duas chamadas de `GroupShuffleSplit(n_splits=1, ...)`, agrupadas por `customer_id`:

1. **Etapa 1** — `test_size = validation_ratio + test_ratio` (0.30 por padrão) sobre o conjunto inteiro de exemplos → separa `train` do restante (`temp`).
2. **Etapa 2** — `test_size = test_ratio / (validation_ratio + test_ratio)` (0.5 por padrão) sobre `temp` → separa `validation` de `test`.

A mesma `seed` alimenta as duas chamadas.

**As proporções são de cliente, não de mensagem.** `GroupShuffleSplit` mantém grupos (`customer_id`) inteiros de um lado só do corte — é o requisito não-negociável (ADR-0002). Como clientes têm números diferentes de mensagens rotuladas, a fração **de mensagens** que cai em cada split se aproxima de 70/15/15, mas não bate exatamente. Isso é esperado, não é bug, e `dataset.json.row_counts` registra os números reais para que o desvio fique visível — forçar exatidão exigiria quebrar um cliente entre dois splits, o que é exatamente o que a regra proíbe.

### 3.5 Volumetria estimada e frequência de execução

Herda a volumetria das specs 02/03 (2.000–40.000 exemplos). Custo adicional específico desta camada:

| Etapa | Custo |
|---|---|
| Reconstrução da janela (uma passada pelo corpus) | O(*n*), linear no número de mensagens |
| `extract_features` por exemplo | até 30 itens por chamada (spec 04 §3.5) — segundos a poucos minutos no teto (§3.4) |
| `GroupShuffleSplit` (duas chamadas) | O(*n* log *n*) sobre até 40.000 linhas — irrelevante |

Orçamento de desempenho: alguns segundos no corpus esperado, até poucos minutos no teto de 10×. Ainda um trabalho de lote pequeno para os padrões do MVP.

**Frequência.** Sob demanda, depois de um `labels.build` bem-sucedido — mesmo padrão em cadeia das specs 02 e 03 (§3.5 de cada uma). `make split` normalmente segue `make labels` na mesma invocação de `make all`.

## 4. Tratamento de erros e resiliência

### 4.1 Taxonomia

- **Erro de portão** (F1) — corpus ou label set não aprovados. *Exit* 3.
- **Erro de identidade** (F2) — `dataset_id` reaproveitado para entradas diferentes. *Exit* 4.
- **Erro de integridade interna** (F6) — contagem não bate, ou vazamento de `customer_id` entre splits. É bug em `split.py`, não no dado (os dois portões já garantiram isso). *Exit* 5.
- **Erro de uso** — CLI mal chamada, `--dataset-id` ausente. *Exit* 2.
- **Erro de ambiente** — I/O. *Exit* 1.

### 4.2 Política de retentativa

**Sobre a montagem e o split: nenhuma.** F3–F7 são determinísticos (§3.3); reexecutar sobre as mesmas entradas e configuração dá o mesmo resultado. Um portão reprovado, um conflito de identidade ou uma falha de sanity check não se resolvem tentando de novo.

**Sobre I/O: 3 tentativas, backoff 1s/2s/4s.** Mesmo padrão das specs 02 e 03, aplicado a quatro leituras (corpus, dois reports, `.parquet` de labels) e à escrita de quatro arquivos de saída.

**Gravação atômica, em ordem.** Os três `.parquet` primeiro (`train`, `validation`, `test`, cada um via `.tmp` + `os.replace()`), o `dataset.json` por último. `dataset.json` é o marcador de sucesso — sua presença com o esquema completo implica que os três splits foram gravados corretamente, o mesmo papel que o `label_report.json` cumpria na spec 03.

### 4.3 *Exit codes*

| Exit | Significado | Quem age |
|---|---|---|
| 0 | Dataset construído (ou reconstruído de forma idêntica) | ninguém |
| 2 | Erro de uso da CLI | operador corrige o comando |
| 3 | **Portão bloqueado** — corpus ou label set sem aprovação correspondente | operador roda (ou corrige) ingest/labels primeiro |
| 4 | **Conflito de identidade** — `dataset_id` já existe com *fingerprint* diferente | operador escolhe um `dataset_id` novo |
| 5 | **Sanity check falhou** — contagem ou vazamento de `customer_id` entre splits | mantenedor do `mood-ml` corrige `transform/split.py` |
| 1 | Erro interno ou de ambiente | mantenedor do `mood-ml` |

Três causas, três códigos, o mesmo raciocínio das specs 02/03: dado (3), identidade (4) e código (5) apontam para pessoas diferentes.

### 4.4 Alertas e notificações

**Nenhum canal externo**, pela mesma razão de sempre — pipeline offline, sob demanda, sem plantão. Revisar apenas na mesma condição já registrada nas specs 02–04 (migração para CI ou execução não supervisionada).

## 5. Logs e observabilidade

### 5.1 Padrão

Idêntico às specs anteriores: um objeto JSON por linha em `stderr`, campos `ts`, `level`, `event`, `run_id`, mais `corpus_id` e `dataset_id`.

### 5.2 Eventos e níveis

| Nível | Evento | Carga |
|---|---|---|
| INFO | `split_started` | `corpus_path`, `dataset_id` |
| INFO | `gate_checked` | `ingest_report_path`, `labels_report_path`, `both_ok: true` |
| INFO | `fingerprint_computed` | `fingerprint`, `reused_dataset_id: bool` |
| INFO | `corpus_and_labels_loaded` | `messages_read`, `label_rows` |
| INFO | `examples_built` | `examples_count`, `context_reprocessed_total` (contagem agregada de T1/T2 aplicados como contexto, útil para acompanhar o custo do §3.4) |
| INFO | `sanity_check_passed` | `checks: ["row_count", "customer_id_no_overlap", "split_sum"]` |
| INFO | `split_computed` | `train_customers`, `validation_customers`, `test_customers`, `train_rows`, `validation_rows`, `test_rows` |
| INFO | `split_finished` | `dataset_id`, `duration_ms`, `dataset_json_path` |
| WARNING | `io_retry` | `attempt`, `operation`, `errno` |
| WARNING | `ratio_deviation` | `split`, `nominal_ratio`, `observed_ratio` — quando o desvio de mensagens por split (§3.4) passa de um limiar informativo (ex.: 5 pontos percentuais) |
| ERROR | `gate_blocked` | `reason` (`corpus_not_ok` \| `labels_not_ok`) |
| ERROR | `dataset_id_conflict` | `dataset_id`, `existing_fingerprint`, `computed_fingerprint` |
| ERROR | `sanity_check_failed` | `check`, `expected`, `observed` |
| ERROR | `io_failed` / `internal_error` | `operation`, `errno` / `exc_type` e *traceback* |

### 5.3 Observabilidade

As mesmas perguntas de sempre, mais uma específica desta camada:

- *este dataset corresponde a qual corpus e a qual label set?* → `dataset.json.source_ids` + `label_set_id`;
- *o split vazou algum cliente?* → `sanity_check_passed.checks` inclui `customer_id_no_overlap`, verificado em tempo de execução, não só em teste;
- *as proporções batem com o nominal?* → `dataset.json.row_counts` traz os números reais; `ratio_deviation` avisa quando o desvio é grande o bastante para chamar atenção.

## 6. Segurança e conformidade

### 6.1 Nenhum texto bruto nesta camada

`text_clean` e `context_clean` já saíram de `extract_features` (spec 04), que aplica T1+T2 antes de devolver qualquer coisa — `split.py` nunca vê `text` cru, porque nunca lê esse campo diretamente do corpus para os fins de montagem de exemplo (só o usa como entrada para `extract_features`, nunca grava o valor de entrada). O mesmo raciocínio de P5 das specs anteriores: um campo que não é lido para gravação não pode vazar por essa via.

### 6.2 Base LGPD/GDPR e P1

Mesma base das specs 02–04: corpus sintético, sem titular real. Nenhuma referência a `duckdb`/`.duckdb` em `transform/split.py` (P1), mesma verificação de *lint*.

### 6.3 P2, por analogia

`mood_scores` é *append-only* por P2 no banco operacional; aqui, o princípio equivalente é a imutabilidade de `dataset_id` (§3.3) — um dataset publicado (usado por um treino) não pode passar a significar outro conteúdo depois. É a mesma garantia de proveniência, aplicada a um artefato de arquivo em vez de uma tabela.

## 7. Requisitos verificáveis

**Portão**
- **SP-R01** QUANDO não existir `validation_report.json` com `status = "ok"` e hash correspondente ao corpus atual, o script DEVE abortar com *exit* 3, sem ler conteúdo.
- **SP-R02** QUANDO não existir `label_report.json` cujo `label_set.sha256` corresponda ao `.parquet` de labels atual e cujo `source.corpus_sha256` corresponda ao corpus atual, o script DEVE abortar com *exit* 3.

**Identidade**
- **SP-R03** `dataset_id` DEVE ser fornecido explicitamente pelo operador; o script DEVE NÃO gerá-lo sozinho.
- **SP-R04** QUANDO existir `dataset.json` para o `dataset_id` informado com *fingerprint* diferente do calculado para a execução atual, o script DEVE abortar com *exit* 4, sem escrever nada.
- **SP-R05** QUANDO o *fingerprint* existente for igual ao calculado, o script DEVE reconstruir e sobrescrever os artefatos de forma determinística.

**Montagem de exemplos**
- **SP-R06** Para cada linha do label set, o script DEVE reconstruir `history` com as mensagens `role = customer` da mesma conversa, `sent_at` menor ou igual ao da disparadora, as até 30 mais recentes incluindo-a, em ordem crescente.
- **SP-R07** O script DEVE chamar `extract_features(history)` para montar `text_clean`/`context_clean` — nunca reimplementar T1, T2 ou a lógica de blocos.
- **SP-R08** Cada linha de exemplo DEVE conter exatamente as colunas de `data-model.md §4.3` (§3.2 deste documento).
- **SP-R09** `example_id` DEVE ser igual a `target_id` da linha de label correspondente.
- **SP-R10** `label_score` de cada exemplo DEVE vir do label set por `target_id`, nunca recalculado.

**Sanity check**
- **SP-R11** QUANDO o número de exemplos montados divergir do número de linhas do label set, o script DEVE abortar com *exit* 5, sem gravar nenhum artefato.
- **SP-R12** QUANDO algum `customer_id` aparecer em mais de um split, o script DEVE abortar com *exit* 5, sem gravar nenhum artefato.
- **SP-R13** A soma das linhas dos três splits DEVE ser igual ao total de exemplos montados.

**Split**
- **SP-R14** O split DEVE usar `GroupShuffleSplit` em duas etapas, agrupado por `customer_id`, com proporções e `seed` lidas de `configs/pipeline.yaml`.
- **SP-R15** O desvio entre a proporção nominal e a proporção observada de **mensagens** por split DEVE ser registrado em `dataset.json.row_counts`, nunca corrigido à força.

**Resiliência e segurança**
- **SP-R16** QUANDO leitura ou escrita falhar com `OSError`, o script DEVE retentar 3 vezes com backoff 1s/2s/4s antes de encerrar com *exit* 1.
- **SP-R17** Os artefatos DEVEM ser gravados atomicamente, nesta ordem: `train.parquet`, `validation.parquet`, `test.parquet`, `dataset.json` por último.
- **SP-R18** Nenhuma coluna do dataset ou do `dataset.json` DEVE conter `text` bruto (P5).
- **SP-R19** O módulo `transform/split.py` DEVE NÃO importar `duckdb` nem referenciar caminho `.duckdb` (P1).

**Teste de contrato (ADR-0008)**
- **SP-R20** DEVE existir um teste que verifica, para conversas sintéticas construídas independentemente do código de produção (mais de 30 mensagens `customer`, mensagens `agent` intercaladas, conversas de 1 mensagem), que o `history` que a janela deslizante da §3.4 produz é **exatamente** o previsto pela regra da ADR-0007 (30 mensagens `customer` mais recentes da conversa, ordem crescente, disparadora por último, `agent` nunca incluído) — calculado por um oráculo de teste separado da implementação, nunca comparando a implementação contra si mesma. A regra em si é de T5 (ADR-0007), aplicada igualmente pela API (online) e por esta camada (offline, a partir do corpus); este é o teste que prova que as duas leituras da regra coincidem.

### 7.1 Catálogo de erros

| Código | Requisito | Fase |
|---|---|---|
| `SP_R01_GATE_CORPUS_NOT_OK` | SP-R01 | F1 |
| `SP_R02_GATE_LABELS_NOT_OK` | SP-R02 | F1 |
| `SP_R04_DATASET_ID_CONFLICT` | SP-R04 | F2 |
| `SP_R11_EXAMPLE_COUNT_MISMATCH` | SP-R11 | F6 |
| `SP_R12_CUSTOMER_ID_LEAK` | SP-R12 | F6 |
| `SP_R13_SPLIT_ROW_SUM_MISMATCH` | SP-R13 | F6 |
| `SP_R16_IO_FAILED` | SP-R16 | F3/F8 |
| `INTERNAL_ERROR` | — | qualquer |

## 8. Decisões deste documento

| # | Decisão | Motivo | Consequência assumida |
|---|---|---|---|
| D1 | **Identidade do `dataset_id`:** operador escolhe, *fingerprint* decide reaproveitar ou bloquear | Reconcilia a escolha do usuário (reaproveitar quando determinístico) com a regra de imutabilidade — sem isso, reaproveitar cegamente arriscaria sobrescrever um `dataset_id` com conteúdo diferente | Introduz um cálculo de *fingerprint* que as specs 02/03 não precisaram, porque lá a identidade já vinha travada por hash de um passo anterior |
| D2 | **`dataset.json` é também o relatório de auditoria** — sem `split_report.json` separado | O esquema que o `data-model.md §4.3` já exige cobre tudo que um relatório traria | Quebra o padrão visual das specs 02/03 (um artefato + um relatório) — documentado aqui para não parecer omissão |
| D3 | **Sem memoização de T1/T2 na reconstrução da janela** | Custo aceitável nos números do MVP (§3.4); evita uma estrutura de cache e sua própria superfície de bug | Reexecuções em corpus maiores que o teto de projeto podem exigir revisitar esta decisão |
| D4 | **Proporções de split são de cliente, não de mensagem, e o desvio é registrado, não corrigido** | Forçar exatidão de mensagens quebraria a garantia de grupo intacto, que é o requisito não-negociável (ADR-0002) | `row_counts` pode se afastar visivelmente de 70/15/15 em corpora pequenos ou com poucos clientes muito ativos |

## 9. Critérios de aceite

- **CA-01** Corpus e label set aprovados, `dataset_id` inédito → *exit* 0, três `.parquet` com o esquema da §3.2 e `dataset.json` com todos os campos exigidos pela `data-model.md §4.3` mais `fingerprint`.
- **CA-02** Corpus ou label set não aprovados → *exit* 3, nenhum arquivo escrito em `data/datasets/<dataset_id>/`.
- **CA-03** `dataset_id` reexecutado com as mesmas entradas e configuração → *exit* 0, artefatos idênticos (mesmo `sha256`).
- **CA-04** `dataset_id` reexecutado com entradas ou configuração diferentes → *exit* 4, artefatos existentes preservados sem alteração.
- **CA-05** Teste obrigatório: interseção de `customer_id` entre `train`, `validation` e `test` é vazia.
- **CA-05a** Teste de contrato obrigatório (SP-R20): para uma conversa sintética de 45 mensagens `customer` intercaladas com `agent`, o `history` da 45ª mensagem tem exatamente 30 itens, todos `customer`, em ordem crescente, e bate item a item com um oráculo calculado independentemente da implementação de `split.py`.
- **CA-06** Teste de regressão do sanity check: uma versão de `split.py` deliberadamente quebrada (ex.: descarta um exemplo) falha com `SP_R11_EXAMPLE_COUNT_MISMATCH`, sem gravar nenhum `.parquet`.
- **CA-07** Caso de borda: cliente com uma única mensagem rotulada produz `history` de tamanho 1 e `context_clean == []` para esse exemplo.
- **CA-08** `dataset.json.row_counts` reflete exatamente as contagens reais de cada split (soma bate com CA-05/SP-R13).
- **CA-09** Teste de P1: nenhuma ocorrência de `duckdb` ou `.duckdb` em `transform/split.py`.
- **CA-10** Teste de P5: busca literal por qualquer `text` bruto do corpus de teste dentro dos três `.parquet` não encontra ocorrência.

## 10. Fora de escopo

- **Vetorização e treino do modelo.** Spec 06 — `train/train.py` lê `text_clean`/`context_clean` e decide como vetorizar (ADR-0008).
- **T1, T2, T3 em si.** Spec 04 — esta spec só os *consome*, via `extract_features`.
- **Validação do contrato `dc-1` e construção do label set.** Specs 02 e 03 — verificadas aqui só pelo portão (F1), nunca reavaliadas.
- **Escolha entre abordagem A e C do modelo.** ADR-0008 — irrelevante para `transform/split.py`, que produz o mesmo `dataset_id` para as duas.
- **Memoização de T1/T2 entre chamadas de `extract_features`.** Deliberadamente adiada (D3); revisitar se a volumetria crescer muito além do teto de projeto.
- **Orquestração, CI e alvos de Makefile.** Spec 09.

## 11. Checklist de implementação

- [ ] CLI (`argparse`), leitura de `configs/pipeline.yaml` (chaves `split.*`)
- [ ] F1 — portão duplo: corpus (reaproveita verificação da spec 02) e label set (reaproveita verificação da spec 03) (SP-R01, SP-R02)
- [ ] F2 — cálculo do *fingerprint* e decisão de reaproveitar/bloquear o `dataset_id` (SP-R03 a SP-R05)
- [ ] F3 — carga do corpus (`load_corpus()`) e do label set
- [ ] F4 — janela deslizante por conversa (§3.4) para reconstruir `history` por mensagem disparadora (SP-R06)
- [ ] Oráculo de teste independente para a regra da ADR-0007 e teste de contrato (SP-R20, CA-05a)
- [ ] F5 — chamada a `extract_features` e montagem da linha de exemplo (SP-R07 a SP-R10)
- [ ] F6 — sanity check: contagem, vazamento de `customer_id`, soma dos splits (SP-R11 a SP-R13)
- [ ] F7 — `GroupShuffleSplit` em duas etapas (SP-R14, SP-R15)
- [ ] F8 — escrita atômica dos três `.parquet` e do `dataset.json`, nesta ordem (SP-R17)
- [ ] Retentativa de I/O com backoff 1s/2s/4s (SP-R16)
- [ ] Log estruturado JSON por evento (§5)
- [ ] Fixtures: corpus+labels aprovados, corpus não aprovado, labels não aprovados, `dataset_id` com *fingerprint* divergente, `split.py` quebrado propositalmente para CA-06
- [ ] Testes `tests/unit/` e `tests/contract/` cobrindo CA-01 a CA-10, incluindo CA-05a

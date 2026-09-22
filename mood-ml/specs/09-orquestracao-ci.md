# 09 — Orquestração Offline e CI (`pipeline.py` / `Makefile` / CI)

**Status:** Aceita · **Versão da spec:** `pl-1` · **Data:** 2026-09-22
**Implementa:** `CLAUDE.md §6` ([9] orquestração + CI) — `python -m pipeline <etapa>`, alvos de `Makefile`, workflow de CI.
**Depende de:** [02-ingest-validate.md](02-ingest-validate.md), [03-labels.md](03-labels.md), [04-transform.md](04-transform.md), [05-split.md](05-split.md), [06-train-evaluate.md](06-train-evaluate.md), [07-registry.md](07-registry.md) — chama a CLI pública de cada um; [constitution.md](../../decisions/constitution.md) P1–P5 por composição (nenhuma regra nova, herdadas de cada camada que esta orquestra)
**Consumido por:** operador humano (linha de comando/`make`), GitHub Actions (CI)
**Implementado em:** [pipeline.py](../pipeline.py), [Makefile](../Makefile), `.github/workflows/ci.yml`

---

## 1. Objetivo geral

Três coisas, todas de colagem sobre camadas já prontas e testadas (specs 02–07) — nenhuma decisão nova de esquema, algoritmo ou contrato de dado:

1. **Um ponto de entrada único**, `python -m pipeline <subcomando>`, com um subcomando por camada (`ingest`, `labels`, `split`, `train`, `evaluate`, `register`, `promote`) mais um subcomando de conveniência, `all`, que roda a cadeia inteira `ingest → labels → split → train → evaluate → register` de uma vez, sobre um corpus bruto.
2. **Um `Makefile`** cujos alvos espelham 1:1 os subcomandos — não um caminho paralelo, só açúcar sintático sobre a mesma CLI.
3. **Um workflow de CI** (GitHub Actions) que roda `ruff` + `pytest` + `make all` sobre a fixture de teste comitada (`tests/fixtures/sample.jsonl`, com `--skip-composition`) a cada `push`/`pull_request`.

O que esta spec **não** decide: nada sobre *como* cada camada valida, treina ou registra — isso já foi decidido e implementado pelas specs 02–07. O espaço de decisão real aqui é só **como encadear** essas camadas sem reescrever a lógica que elas já têm.

## 2. Arquitetura e fluxo de dados

### 2.1 Posição no fluxo

```
data/raw/synthetic/<corpus_id>.jsonl
        │
        ▼
python -m pipeline all <path> [--config] [--skip-composition]
        │
        ├─► ingest.validate.main()   ──► validation_report.json
        ├─► labels.build.main()      ──► <label_set_id>.parquet
        ├─► transform.split.main()   ──► data/datasets/<dataset_id>/{...}.parquet + dataset.json
        │        (dataset_id gerado por pipeline.py, §2.4)
        ├─► train.train.main()       ──► models/_staging/.../{baseline,candidate}.joblib
        ├─► evaluate.metrics.main()  ──► eval.json
        │        (fingerprint recalculado por pipeline.py, §2.3 — não fica preso ao stdout do passo anterior)
        └─► registry.registry.main("register", ...)  ──► models/<model_version>/{model.joblib,manifest.json}
                 (só se eval.json.gate.passed == true; nunca chama "promote")
```

### 2.2 Dois modos de operação

**Subcomandos individuais** (`ingest`, `labels`, `split`, `train`, `evaluate`, `register`, `promote`) são wrappers finos: cada um constrói a lista de argumentos e chama o `main(argv)` **já existente** do módulo correspondente (`ingest.validate.main`, `labels.build.main`, `transform.split.main`, `train.train.main`, `evaluate.metrics.main`, `registry.registry.main`), devolvendo exatamente o mesmo *exit code* e produzindo exatamente os mesmos logs que chamar aquele módulo diretamente produziria. **Nenhuma lógica de porta, sanidade ou erro é reimplementada aqui** — `pipeline.py` não sabe nada sobre *como* cada camada decide se algo passou ou não.

**`all`** é o único subcomando que precisa de algo a mais: encadear os passos exige que o `dataset_id` gerado no passo `split` chegue até `train`/`evaluate`/`register`, e que o *fingerprint* calculado no passo `train` chegue até `register`. A saída de cada `main()` é só um *exit code* — não devolve esses valores como objeto Python. Resolvido assim, sem *parsing* de texto de stdout:
- `dataset_id`: `all` já o gerou antes de chamar `split` (§2.4) — é o mesmo valor usado nos passos seguintes, não precisa ser lido de volta.
- *fingerprint*: em vez de ler o valor que `train` calculou, `all` **recalcula-o de forma independente**, usando a mesma fórmula determinística que `train.train` usa internamente (`evaluate.metrics.dataset_parquet_hashes` + `evaluate.metrics.compute_training_fingerprint`, ambas já públicas e importadas pelo próprio `train.train`) sobre o mesmo `dataset_id`, `config` e `feature_spec_version`. Como a fórmula é pura e determinística (spec 06 §3.3), o valor recalculado é garantidamente igual ao que `train` gravou — sem *parsing* de stdout, sem duplicar a lógica de treino.

Essa é a única peça de código genuinamente nova desta camada: tudo o mais é composição do que já existe.

### 2.3 Passo a passo lógico — `all`

| Passo | Ação | Aborta com |
|---|---|---|
| 1 | Confere `path` (corpus bruto) existe | *exit* 2 (uso) |
| 2 | `ingest.validate.main(["path", ...])` | *exit* do próprio `ingest` (1–4) |
| 3 | `labels.build.main(["path", ...])` | *exit* do próprio `labels` (1–3) |
| 4 | Gera `dataset_id` (§2.4) | — (não falha) |
| 5 | `transform.split.main(["path", "--dataset-id", dataset_id, ...])` | *exit* do próprio `split` (1–3) |
| 6 | `train.train.main([dataset_id, ...])` | *exit* do próprio `train` (1–4) |
| 7 | Recalcula o *fingerprint* (§2.2) | — (não falha; é determinístico) |
| 8 | `evaluate.metrics.main([dataset_id, ...])` | *exit* do próprio `evaluate` (1–3, **5** se o *quality gate* reprovar) |
| 9 | QUANDO o passo 8 devolver 5, `all` para **sem** chamar `register` | *exit* 5 (propagado) |
| 10 | `registry.registry.main(["register", dataset_id, fingerprint, ...])` | *exit* do próprio `register` (1–3, 6) |
| 11 | Sucesso — loga o resumo (`corpus_id`, `dataset_id`, *fingerprint*, `model_version`) | *exit* 0 |

`all` **nunca** chama `promote` — continua sendo um comando manual e deliberado, decisão já registrada na spec 07.

### 2.4 Geração automática de `dataset_id`

`corpus_id` **não** é gerado por esta camada — vem do próprio arquivo `path` que o operador fornece (o nome do arquivo sob `data/raw/synthetic/<corpus_id>.jsonl`, IG-R02); `pipeline.py` só o repassa adiante.

`dataset_id`, ao contrário, não tem dono hoje — `transform.split` exige `--dataset-id` como argumento obrigatório (não o gera). `all` gera automaticamente, no mesmo formato `ds-AAAA-MM-DD-<letra>` já usado nas specs 03/05: data corrente, letra `a`, incrementando (`b`, `c`, ...) se já existir um diretório `data/datasets/<candidato>/` — mesmo padrão de numeração sequencial já usado por `registry._mint_model_version` (spec 07 §3.3), aplicado aqui a um nome em vez de um número.

## 3. Requisitos técnicos e premissas

### 3.1 Conexões e autenticação

**Nenhuma.** Processo local, mesma premissa de todas as camadas offline (specs 02–07).

```bash
python -m pipeline ingest data/raw/synthetic/<corpus_id>.jsonl [--skip-composition]
python -m pipeline labels data/raw/synthetic/<corpus_id>.jsonl
python -m pipeline split data/raw/synthetic/<corpus_id>.jsonl --dataset-id <dataset_id>
python -m pipeline train <dataset_id>
python -m pipeline evaluate <dataset_id>
python -m pipeline register <dataset_id> <fingerprint>
python -m pipeline promote <model_version> [--force]
python -m pipeline all data/raw/synthetic/<corpus_id>.jsonl [--skip-composition] [--config configs/pipeline.yaml]
```

Runtime: Python 3.12, nenhuma dependência nova — só importa os `main`/funções públicas já existentes de `ingest`, `labels`, `transform`, `train`, `evaluate`, `registry`.

### 3.2 Formato dos dados

Não introduz nenhum esquema de dado novo — `pipeline.py` não lê nem grava nenhum artefato diretamente (exceto a checagem de existência do `dataset_id` candidato em `data/datasets/`, §2.4). Todo I/O de dado real continua dentro de cada camada, como já especificado nas specs 02–07.

### 3.3 Estratégia de idempotência

`all` **não introduz nenhum mecanismo de idempotência próprio** — reaproveita o de cada camada (hash-lock em `ingest`, *overwrite* determinístico em `labels`, *fingerprint* em `split`/`train`, varredura de *fingerprint* em `register`). Reexecutar `all` sobre o mesmo corpus, com o mesmo `--dataset-id` implícito (mesma data), é seguro pelos mesmos motivos que reexecutar cada etapa isoladamente já é seguro. A única parte nova (geração de `dataset_id`, §2.4) é idempotente por construção: rodar `all` duas vezes no mesmo dia gera dois `dataset_id` distintos (`-a`, `-b`) porque cada `split` já é uma nova tentativa de particionamento — não há uma noção de "mesmo `all`" a deduplicar além do que `split`/`train`/`register` já fazem por dentro.

### 3.4 Volumetria estimada e frequência de execução

Mesma ordem de grandeza do MVP inteiro — dezenas de execuções de `all` ao longo do projeto (uma por corpus sintético gerado), não um pipeline de produção rodando em cadência fixa. `all` sobre o corpus completo (spec01/05) roda em segundos a poucos minutos — dominado pelo treino (spec06 §3.4), não por esta camada.

### 3.5 `Makefile`

Um alvo por subcomando, repassando parâmetros via variáveis de `make`:

```makefile
ingest:
	python -m pipeline ingest $(PATH) $(if $(SKIP_COMPOSITION),--skip-composition)

labels:
	python -m pipeline labels $(PATH)

split:
	python -m pipeline split $(PATH) --dataset-id $(DATASET)

train:
	python -m pipeline train $(DATASET)

evaluate:
	python -m pipeline evaluate $(DATASET)

register:
	python -m pipeline register $(DATASET) $(FINGERPRINT)

promote:
	python -m pipeline promote $(VERSION) $(if $(FORCE),--force)

all:
	python -m pipeline all $(PATH) $(if $(SKIP_COMPOSITION),--skip-composition)
```

Nenhum alvo chama um módulo (`ingest.validate`, `train.train`, ...) diretamente — todos passam por `python -m pipeline`, um único ponto de entrada.

### 3.6 CI (GitHub Actions)

`.github/workflows/ci.yml`, gatilho `push` e `pull_request` para `main`:

```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  test:
    steps:
      - checkout
      - setup-python (3.12)
      - pip install -r mood-ml/requirements.txt
      - ruff check .
      - pytest -q
      - make all PATH=tests/fixtures/sample.jsonl SKIP_COMPOSITION=1
```

Só validação — nenhum artefato do `make all` de teste é publicado ou persistido além do log do próprio job (decisão do usuário, §8 D3): é um MVP acadêmico, sem infraestrutura de deploy ainda.

## 4. Tratamento de erros e resiliência

### 4.1 Propagação de erro em `all`

**Falha rápida, sem tentar continuar.** Assim que qualquer passo de `all` devolve um *exit code* não-zero, `all` para imediatamente — nenhum passo seguinte roda, e `all` devolve **o mesmo *exit code*** que aquele passo devolveria se fosse chamado isoladamente (nenhuma tradução, nenhum código genérico "algo falhou no meio"). Isso preserva o significado que cada camada já definiu para seus próprios códigos (§4.2) sem que o operador precise adivinhar em qual etapa `all` parou — o próprio código já diz.

Caso notável: se o *quality gate* de `evaluate` reprovar (*exit* 5), `all` para ali — `register` nunca roda sobre um candidato que não passou no *gate*. Não é um erro no sentido de "algo quebrou": `eval.json` já foi gravado normalmente pela etapa `evaluate` (spec 06 §4.1, D do §4.1), só o registro é que não acontece.

### 4.2 Catálogo de *exit codes* (união, não tradução)

Cada camada já usa o mesmo vocabulário de números para os mesmos tipos de falha (2 = uso, 3 = portão, 1 = erro interno/ambiente) — a tabela abaixo é a união do que specs 02–07 já definem, não uma tradução nova:

| Exit | Significado | Onde já existia |
|---|---|---|
| 0 | Sucesso (ou no-op idempotente) | todas |
| 1 | Erro interno ou de ambiente (I/O esgotado, exceção não esperada) | todas |
| 2 | Erro de uso da CLI (argumento inválido/faltando) | todas |
| 3 | Portão bloqueado (corpus não aprovado, staging incompleto, `dataset_id`/`model_version` inexistente, etc.) | todas |
| 4 | Falha de sanidade (composição do corpus, spec 02; previsão não-finita, spec 06) | ingest, train |
| 5 | *Quality gate* de `evaluate` reprovado | evaluate |
| 6 | Candidato rejeitado em `register` (`gate.passed == false`) | register |
| 7 | Regressão de métrica bloqueada em `promote`, sem `--force` | promote |

`all` devolve sempre um destes valores — nunca um código próprio fora desta tabela.

### 4.3 Política de retentativa

**Nenhuma nova.** Cada camada já retenta sua própria I/O (3x, *backoff* 1s/2s/4s, specs 02–07); `pipeline.py` não adiciona uma segunda camada de retentativa por cima — evita duplicar espera (uma retentativa de `pipeline.py` em cima de uma retentativa de `train.train`, por exemplo, multiplicaria o tempo de espera sem motivo).

### 4.4 Alertas e notificações

**Nenhum canal externo**, mesma razão de todas as specs anteriores. O CI (§3.6) é o único "alerta" desta camada — falha vermelha no GitHub, sem notificação ativa além do que o próprio GitHub já oferece.

## 5. Logs e observabilidade

### 5.1 Padrão

Mesmo formato de todas as specs anteriores: JSON por linha em `stderr`, campos `ts`, `level`, `event`.

### 5.2 Eventos de `pipeline.py`

`pipeline.py` loga só no nível de **orquestração** — cada passo já produz seus próprios logs detalhados ao chamar o `main()` daquele módulo (§2.2), então `pipeline.py` não duplica isso.

| Nível | Evento | Carga |
|---|---|---|
| INFO | `pipeline_started` | `subcommand` (`all` ou o nome do subcomando individual) |
| INFO | `dataset_id_generated` | `dataset_id` — só em `all` |
| INFO | `step_started` | `step` (`ingest`\|`labels`\|`split`\|`train`\|`evaluate`\|`register`) — só em `all` |
| INFO | `step_finished` | `step`, `exit_code: 0` — só em `all` |
| ERROR | `step_failed` | `step`, `exit_code` — só em `all`, quando um passo devolve não-zero |
| INFO | `pipeline_finished` | `corpus_id`, `dataset_id`, `fingerprint`, `model_version`, `duration_ms` — só em `all`, só se chegou ao fim |

**Consequência assumida**, não um descuido: rodar uma etapa via `all` produz um log mais grosso (só `step_started`/`step_finished` por passo) do que rodar aquele mesmo passo isoladamente (log detalhado do módulo, spec 02–07 §5 de cada um) — porque `all` delega a esses `main()` sem nada escondido no meio, o log detalhado de cada camada **também aparece**, entrelaçado com os eventos de orquestração acima. Não há perda de detalhe, só uma camada extra de eventos por cima.

### 5.3 Observabilidade

*"Por que `all` parou?"* → o último `step_failed` no log, mais o log detalhado daquele módulo logo acima (mesma pergunta, duas granularidades). *"Qual `dataset_id`/*fingerprint*/`model_version` saiu desta corrida?"* → `pipeline_finished`, um evento só, sem precisar procurar em três lugares.

## 6. Segurança e conformidade

### 6.1 Nada de novo entra nesta camada

`pipeline.py` não lê `text` de mensagem em nenhum momento — só encaminha caminhos de arquivo e identificadores (`corpus_id`, `dataset_id`, *fingerprint*, `model_version`) entre chamadas. A mesma garantia estrutural repetida desde a spec 03: o que a camada nunca lê não pode vazar por ela.

### 6.2 Base LGPD/GDPR

Mesma base de todas as specs anteriores — corpus sintético, sem titular real; esta camada não introduz superfície nova.

### 6.3 P1

Nenhuma referência a `duckdb`/`.duckdb` em `pipeline.py` — mesma verificação de *lint* de todas as camadas anteriores.

## 7. Requisitos verificáveis

**CLI**
- **PL-R01** `python -m pipeline` DEVE oferecer os subcomandos `ingest`, `labels`, `split`, `train`, `evaluate`, `register`, `promote`, `all`.
- **PL-R02** Cada subcomando individual (todos exceto `all`) DEVE delegar para o `main(argv)` do módulo correspondente, repassando os argumentos recebidos sem alteração, e DEVE devolver exatamente o *exit code* que esse `main()` devolveu.
- **PL-R03** Nenhum subcomando individual DEVE reimplementar lógica de porta, sanidade ou erro já existente em sua camada correspondente.

**`all`**
- **PL-R04** `all` DEVE receber o caminho do corpus bruto como argumento posicional único, sem exigir `--dataset-id` do operador.
- **PL-R05** QUANDO `all` for executado, o sistema DEVE gerar `dataset_id` automaticamente no formato `ds-AAAA-MM-DD-<letra>`, incrementando a letra se já existir um diretório com esse nome em `data/datasets/`.
- **PL-R06** `all` DEVE executar, nesta ordem, os passos `ingest → labels → split → train → evaluate → register`, cada um delegando ao `main()` do módulo correspondente (mesma disciplina de PL-R02).
- **PL-R07** QUANDO qualquer passo de `all` devolver um *exit code* não-zero, `all` DEVE abortar imediatamente, sem executar os passos seguintes, devolvendo o mesmo *exit code* do passo que falhou.
- **PL-R08** `all` DEVE NUNCA chamar `registry.registry.main(["promote", ...])` — a promoção permanece exclusivamente um comando manual e separado.
- **PL-R09** O *fingerprint* usado na chamada a `register` DEVE ser recalculado por `all` através das funções públicas e determinísticas de `evaluate.metrics` (`dataset_parquet_hashes` + `compute_training_fingerprint`) — DEVE NÃO ser extraído por *parsing* de texto de stdout/log de nenhum passo anterior.

**`Makefile`**
- **PL-R10** O `Makefile` DEVE expor um alvo por subcomando, cada um invocando `python -m pipeline <subcomando> ...` — nunca um módulo individual diretamente.

**CI**
- **PL-R11** O workflow de CI DEVE rodar em todo `push` e `pull_request` para `main`.
- **PL-R12** O workflow de CI DEVE rodar, nesta ordem, `ruff check .`, `pytest -q`, e `make all` sobre `tests/fixtures/sample.jsonl` com a flag equivalente a `--skip-composition`.
- **PL-R13** QUANDO qualquer um dos três passos do CI falhar, o workflow DEVE terminar com falha (*exit* não-zero do job).
- **PL-R14** O CI DEVE NÃO publicar nem persistir nenhum artefato do `make all` de teste além do log do próprio job.

**Segurança**
- **PL-R15** `pipeline.py` DEVE NÃO importar `duckdb` nem referenciar caminho `.duckdb` (P1).

### 7.1 Catálogo de erros

Não introduz nenhum código de erro novo — reaproveita o catálogo de cada camada (spec 02 §7.1, spec 03, spec 05, spec 06 §7.1, spec 07 §7.1). A união está na tabela de §4.2.

## 8. Decisões deste documento

| # | Decisão | Motivo | Consequência assumida |
|---|---|---|---|
| D1 | **Subcomandos individuais delegam ao `main()` já existente de cada módulo**, sem reimplementar nada | Zero duplicação de lógica de porta/erro/log; a única forma de `pipeline.py` divergir do comportamento de cada camada seria um bug na própria delegação, superfície mínima | Subcomandos individuais de `pipeline.py` são estritamente equivalentes a chamar o módulo diretamente — não há ganho funcional em usá-los além de um único ponto de entrada memorizável |
| D2 | **`all` recalcula o *fingerprint* de forma independente**, em vez de propagar o valor que `train` calculou | Evita *parsing* de stdout (frágil) e evita duplicar a lógica de treino; a fórmula já é pública e determinística (spec 06 §3.3) | `all` faz uma chamada extra, barata (hashes de arquivos já gravados), só para obter um valor que já existe em disco sob outra forma (embutido no caminho de staging) |
| D3 | **CI só valida, não publica artefato** do `make all` de teste | MVP acadêmico, sem infraestrutura de deploy/armazenamento de artefato ainda | Inspecionar `validation_report.json`/`eval.json`/`manifest.json` gerados pela fixture exige rodar `make all` localmente — não há como baixá-los de uma execução de CI passada |
| D4 | **`all` nunca chama `promote`** | Mantém a decisão já registrada na spec 07 (promoção é sempre uma ação humana deliberada) — estendê-la para dentro de `all` seria contradizê-la por um caminho indireto | O pipeline automatizado (CI, ou um operador rodando `make all`) nunca troca sozinho qual modelo está ativo em produção |
| D5 | **`all` propaga o *exit code* do passo que falhou, sem traduzir** | Preserva o significado que cada camada já deu a seus próprios códigos; um operador que já conhece os códigos de `train`/`evaluate`/`register` não precisa aprender um segundo vocabulário para `all` | Nenhuma — é a leitura mais direta possível do que já existe |

## 9. Critérios de aceite

- **CA-01** `python -m pipeline ingest <path>` (corpus válido) → mesmo *exit* e mesmo `validation_report.json` que `python -m ingest.validate <path>` produziria isoladamente.
- **CA-02** `python -m pipeline all <path-do-corpus-real>` → *exit* 0, `data/datasets/<dataset_id>/` gerado com `dataset_id` no formato `ds-AAAA-MM-DD-<letra>`, `models/<model_version>/{model.joblib,manifest.json}` gravados, `active.json` **não** criado/alterado.
- **CA-03** Duas execuções de `all` no mesmo dia sobre o mesmo corpus → dois `dataset_id` distintos (`-a`, `-b`), sem colisão nem sobrescrita.
- **CA-04** Corpus que reprova a validação de `ingest` (spec 02) → `all` para no passo 2, *exit* igual ao que `ingest.validate` devolveria sozinho, nenhum diretório em `data/datasets/` criado.
- **CA-05** Dataset cujo candidato reprova o *quality gate* de `evaluate` (spec 06) → `all` para no passo 8/9, *exit* 5, `eval.json` gravado, nenhum `models/<model_version>/` criado.
- **CA-06** *Fingerprint* recalculado por `all` (§2.2) é *byte-a-byte* igual ao gravado por `train` em `train_manifest.json.fingerprint` para a mesma corrida.
- **CA-07** `make all PATH=tests/fixtures/sample.jsonl SKIP_COMPOSITION=1` → mesmo resultado de CA-02, usando a fixture de teste.
- **CA-08** Workflow de CI, rodado localmente via `act` ou inspecionado manualmente: `ruff check .`, `pytest -q`, `make all` sobre a fixture, nesta ordem; falha proposital em qualquer um dos três (ex.: quebrar um teste de propósito) faz o job falhar.
- **CA-09** Teste de P1: nenhuma ocorrência de `duckdb`/`.duckdb` em `pipeline.py`.

## 10. Fora de escopo

- **Qualquer lógica de validação, treino, avaliação ou registro** — pertence inteiramente às specs 02–07; esta camada só encadeia.
- **`promote` dentro de `all`** — decisão explícita (D4); permanece comando manual e separado.
- **Orquestração distribuída/agendada** (cron, Airflow/Prefect) — já listado como evolução de escala fora do MVP (`CLAUDE.md §8`).
- **Publicação de artefato de CI, deploy automático, ambiente de produção** — fora do escopo deste MVP (D3).
- **Serviço de inferência (`infer/predict.py`, `main.py`)** — spec 08; não faz parte da cadeia `all`, é um processo de vida longa separado.
- **Controle de concorrência entre execuções simultâneas de `pipeline.py`** — mesma premissa de operador único já assumida pela spec 07.

## 11. Checklist de implementação

- [ ] `pipeline.py` — subcomandos individuais como *wrappers* finos sobre o `main(argv)` de cada módulo (PL-R01 a PL-R03)
- [ ] `all` — geração de `dataset_id` (`ds-AAAA-MM-DD-<letra>`, incrementando) (PL-R04, PL-R05)
- [ ] `all` — encadeamento `ingest → labels → split → train → evaluate → register`, falha rápida propagando *exit code* (PL-R06, PL-R07)
- [ ] `all` — recálculo determinístico do *fingerprint* via `evaluate.metrics` (PL-R09), nunca chamar `promote` (PL-R08)
- [ ] Log de orquestração (`pipeline_started`/`step_started`/`step_finished`/`step_failed`/`pipeline_finished`) (§5.2)
- [ ] `Makefile` — um alvo por subcomando, repassando variáveis (PL-R10)
- [ ] `.github/workflows/ci.yml` — `ruff check .` + `pytest -q` + `make all` sobre a fixture, em `push`/`pull_request` para `main` (PL-R11 a PL-R14)
- [ ] Testes `tests/unit/` (geração de `dataset_id`, recálculo de *fingerprint*, propagação de *exit code*) e `tests/contract/` (subcomandos individuais equivalentes ao módulo direto, `all` ponta a ponta com a fixture) cobrindo CA-01 a CA-09

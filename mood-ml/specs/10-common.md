# 10 — Infraestrutura Transversal (`common/`)

**Status:** Aceita · **Versão da spec:** `cm-1` · **Data:** 2026-09-24
**Implementa:** correção F3 do code review de 2026-09-24. Oito módulos repetiam retry de I/O, log estruturado, classes de erro e escrita atômica.
**Depende de:** nenhuma. Não importa nenhum outro módulo do mood-ml.
**Consumido por:** [02](02-ingest-validate.md), [03](03-labels.md), [05](05-split.md), [06](06-train-evaluate.md), [07](07-registry.md), [08](08-infer.md), [09](09-orquestracao-ci.md)
**Implementado em:** `common/errors.py`, `common/io.py`, `common/log.py`

---

## 1. Objetivo

Uma única implementação do que as specs 02 a 09 já exigem de forma idêntica: retentativa de I/O (3x, *backoff* 1s/2s/4s), log com um objeto JSON por linha, escrita atômica e o formato de erro `(code, detail)`.

Esta spec **não cria comportamento novo**. Os requisitos de cada camada continuam valendo como estão: IG-R15, LB-R14, SP-R16, IF-R19, os códigos de erro e os *exit codes*. O que muda é onde o código mora.

As cópias já tinham divergido. No ingest, o evento `io_retry` saía com o JSON serializado dentro do campo `event`, e não como objeto de nível único. Esse evento volta ao formato da spec 02 §5.1.

## 2. Contrato

| Módulo | Função / classe | Contrato |
|---|---|---|
| `common/errors.py` | `PipelineError(code, detail)` | Exceção base. Grava `.code` e `.detail`; `str(exc) == detail`. Cada camada declara as suas subclasses (`LabelGateError(PipelineError)` etc.), porque o tipo decide o *exit code*. |
| `common/io.py` | `IO_RETRY_BACKOFF_SECONDS = (1, 2, 4)` | Única definição do *backoff*. |
| | `with_io_retry(operation, func, run_id=None, *, logger, error)` | Chama `func()`. Em `OSError`, loga `io_retry` (`attempt`, `operation`, `errno`) e retenta após 1s, 2s e 4s: são 4 tentativas no total. Esgotadas, levanta `error(detail)`, em que `error` é uma fábrica da exceção da camada. Outras exceções propagam sem retentativa. |
| | `atomic_write(dest, write_tmp)` | Cria o diretório pai, chama `write_tmp(<dest>.tmp)` e faz `os.replace` para `dest`. Não faz retentativa própria: quem chama embrulha em `with_io_retry`. |
| | `write_json(payload, path)` | `json.dump` com `indent=2, sort_keys=True, ensure_ascii=False`. |
| | `write_json_atomic(payload, dest)` | `atomic_write` com `write_json`. Como `atomic_write`, não faz retentativa própria. |
| `common/log.py` | `now_iso()`, `to_iso(dt)` | ISO-8601 UTC, com milissegundos e sufixo `Z`. |
| | `JsonFormatter`, `TextFormatter` | Um objeto JSON por linha (`ts`, `level`, `event` + campos), ou texto legível para `--log-format text`. |
| | `configure_logging(logger, log_format, log_level)` | Um único *handler* em `stderr`, sem propagação. Chamar de novo substitui o *handler* em vez de duplicar. |
| | `log(logger, level, event, **fields)` | Emite `event` com os campos extras. Inclui `run_id` quando não é `None` e o omite quando é `None` (o infer não tem `run_id`). |

Cada camada mantém o seu próprio `logger` (`logging.getLogger("<pacote>.<módulo>")`) e pode manter *wrappers* privados de uma linha (`_log`, `_with_io_retry`, `_write_json_atomic`), para não alterar *call sites* nem pontos de *monkeypatch* dos testes.

## 3. Requisitos verificáveis

- **CM-R01** `with_io_retry` DEVE retentar apenas `OSError`, no máximo 3 vezes, com espera de 1s, 2s e 4s, e DEVE levantar a exceção produzida por `error` quando as tentativas se esgotarem.
- **CM-R02** QUANDO o formato for `json`, toda linha de log emitida via `common.log`, inclusive `io_retry`, DEVE ser um único objeto JSON com `ts`, `level` e `event` no nível raiz.
- **CM-R03** `atomic_write` DEVE deixar em `dest` o conteúdo antigo ou o novo inteiro, nunca um arquivo parcial, e DEVE NÃO deixar `.tmp` após o sucesso.
- **CM-R04** `common/` DEVE NÃO conter regra de negócio, código de erro de camada nem importar outro pacote do mood-ml.
- **CM-R05** Nenhum módulo fora de `common/` DEVE redefinir formatador de log, *backoff* de retentativa ou `time.sleep` de retentativa. Isso é verificado por teste.

## 4. Decisões deste documento

| # | Decisão | Por quê | Custo aceito |
|---|---|---|---|
| D1 | Subclasses por camada em vez de uma exceção única | O `main()` de cada CLI decide o *exit code* pelo tipo; trocar tipo por código espalharia `if` pelos `main()` | Algumas classes de uma linha por camada |
| D2 | *Wrappers* privados finos por módulo | Mantém cerca de 150 *call sites* e os *monkeypatch* dos testes intactos; o diff fica mecânico | Um nível de indireção |
| D3 | `error` como fábrica em vez de código fixo | Cada camada tem código próprio (`LB_R14_IO_FAILED`, `SP_R16_IO_FAILED`…); o registry escolhe o código por operação | — |
| D4 | `IngestGateError`, `_LabelsGateError` e `FeatureSpecMismatchError` ficam fora de `PipelineError` | Carregam `.reason` ou `field/expected/found`, não `code` | Três exceções com forma própria |

## 5. Critérios de aceite

- **CA-01** Uma função que falha 2x com `OSError` e depois tem sucesso → `with_io_retry` devolve o valor; os *sleeps* registrados são `[1, 2]`.
- **CA-02** Uma função que sempre falha → a exceção de `error` é levantada após 4 chamadas; os *sleeps* são `[1, 2, 4]`.
- **CA-03** Um `ValueError` na primeira chamada → propaga na hora, sem *sleep*.
- **CA-04** O `io_retry` capturado em `stderr` com formato `json` → `json.loads(linha)["event"] == "io_retry"`.
- **CA-05** `atomic_write` com sucesso → `dest` com o conteúdo novo e nenhum `*.tmp` no diretório.
- **CA-06** `log(..., run_id=None)` → a linha não tem a chave `run_id`; com `run_id="x"` → `"run_id": "x"`.
- **CA-07** `configure_logging` chamado 2x no mesmo *logger* → um único *handler*.
- **CA-08** Guarda anti-duplicação: nenhum `.py` fora de `common/` e `tests/` define subclasse de `logging.Formatter` (pela árvore sintática) nem contém `IO_RETRY_BACKOFF_SECONDS =`, `time.sleep(` ou `lambda tmp: write_json(`. Uma cópia de `common/log.py` fora de `common/` é acusada.
- **CA-09** A suíte existente continua verde sem alterar asserções, só os alvos de *monkeypatch* de `time.sleep`.
- **CA-10** `write_json_atomic` grava em `dest` exatamente os bytes de `write_json` para o mesmo *payload* e não deixa `*.tmp`.

## 6. Fora de escopo

- Retentativa na camada de orquestração: a [spec 09 §4.3](09-orquestracao-ci.md) já a proíbe.
- Mudar formato de log, nomes de evento, códigos de erro ou *exit codes* de qualquer camada.
- Unificar `_load_config` (train, evaluate, pipeline) e `_load_split_config`. Eles leem seções diferentes do YAML; fica para uma revisão futura, se voltar a divergir.

## 7. Checklist de implementação

- [x] `common/errors.py`, `common/io.py`, `common/log.py` (CM-R01 a CM-R04)
- [x] `tests/unit/test_common.py` (CA-01 a CA-07)
- [x] `tests/unit/test_no_duplicated_infra.py` (CA-08, CM-R05)
- [x] Migrar `pipeline.py` (só log)
- [x] Migrar `evaluate/metrics.py`
- [x] Migrar `train/train.py`
- [x] Migrar `registry/registry.py`
- [x] Migrar `transform/split.py`
- [x] Migrar `labels/build.py` (alvo de *monkeypatch* de `time.sleep` → `common.io.time`)
- [x] Migrar `ingest/validate.py` (corrige o `io_retry` aninhado; alvo de *monkeypatch* → `common.io.time`)
- [x] Migrar `infer/predict.py` (`RegistryBrokenError(PipelineError)`)
- [x] Suíte verde após cada migração (CA-09); `make all` de ponta a ponta sobre a fixture
- [x] `write_json_atomic` no lugar do *lambda* aninhado nos 6 módulos (CA-10, CA-08)
- [x] Guarda de CM-R05 barra qualquer subclasse de `logging.Formatter` fora de `common/` (CA-08)

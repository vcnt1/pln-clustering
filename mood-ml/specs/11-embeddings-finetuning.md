# 11 — Abordagem C: embeddings com fine-tuning (revisão das specs 06, 07, 08)

**Status:** Proposta · **Versão da spec:** `ft-1` · **Data:** 2026-09-25
**Implementa:** abordagem C da [ADR-0008](../../decisions/ADR-0008-abordagens-de-modelo.md), com a mudança da [ADR-0009](../../decisions/ADR-0009-abordagem-c-com-fine-tuning.md) (**proposta, depende de aprovação**)
**Depende de:** [ADR-0001](../../decisions/ADR-0001-escala-do-humor.md), [ADR-0002](../../decisions/ADR-0002-origem-do-ground-truth.md), [ADR-0007](../../decisions/ADR-0007-humor-por-conversa.md); constitution P3, P4; specs [04](04-transform.md), [05](05-split.md), [06](06-train-evaluate.md), [07](07-registry.md), [08](08-infer.md)
**Consumido por:** `train/train.py`, `evaluate/metrics.py`, `registry/registry.py`, `infer/predict.py`
**Implementado em:** `transform/embeddings.py` (novo), `train/train.py` (ramo C), demais módulos com ajustes pontuais (§8)

---

## 1. Objetivo

Acrescentar `algorithm = "embeddings-ft"` ao pipeline: um encoder de sentenças multilíngue **ajustado** ao domínio, com cabeça linear de regressão. A abordagem A (`tfidf-ridge`) continua intacta e selecionável. Só a vetorização de T3 muda; T1, T2, a montagem dos blocos, `fs-1`, o `dataset_id` e o quality gate são os mesmos (ADR-0008).

**Pergunta que a spec responde para o TCC:** adaptar o encoder ao domínio supera TF-IDF sobre o mesmo dataset? O resultado "não supera" é válido e deve ser reportado.

## 2. Modelo

```
text_clean ────────────► encoder ─► mean pooling ─► e_t (d)          ┐
context_clean[0..n-1] ─► encoder ─► mean pooling ─► média ─► c (d)   ├─► [e_t ; w·c] ─► dropout ─► Linear(2d→1) ─► clip
                         (sem contexto: c = vetor nulo)              ┘
```

- **Encoder base (padrão):** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (d = 384), carregado com `transformers.AutoModel`. O pooling é média com máscara de atenção, implementado no código (não depende da biblioteca `sentence-transformers`).
- `w` = `train.embeddings.context.weight` (0,5, como em A).
- **O contexto é codificado sem gradiente** (`torch.no_grad`) durante o treino: o gradiente só passa pela mensagem disparadora. Evita custo de retropropagar até 29 mensagens por exemplo. A função de codificação é a mesma no treino e na inferência (P4).
- **Congelado no fine-tuning:** a matriz de palavras do encoder (~96M dos ~118M parâmetros). Treinam as camadas transformer e a cabeça.
- Saída bruta recortada por `clip_score` (spec 06 §3.5), como em A.

## 3. Treino em duas etapas

| Etapa | O que faz | Por quê |
|---|---|---|
| **1. Sondagem linear** | Encoder congelado; embeddings de `train`; `Ridge(alpha, solver="lsqr")` sobre `[e_t ; w·c]`. Registra o MAE de validação. | Inicializa a cabeça com pesos sensatos, para o gradiente inicial não destruir o encoder. O MAE registrado é a variante "C congelado" da ADR-0008, de graça, como diagnóstico. |
| **2. Fine-tuning** | Copia coef/intercepto do Ridge para a `Linear`. AdamW, perda MSE, warmup linear, early stopping no MAE de `validation` (recortado). Restaura o melhor checkpoint. | Adapta o encoder ao domínio. |

Hiperparâmetros de partida (todos em config; **valores iniciais a validar na primeira execução**, não resultado de busca — herda TN-R06):

`epochs_max: 5`, `patience: 2`, `batch_size: 16`, `lr_encoder: 2e-5`, `lr_head: 1e-3`, `weight_decay: 0.01`, `warmup_ratio: 0.1`, `max_length: 128`, `dropout: 0.1`.

## 4. Configuração (`configs/pipeline.yaml`)

`train.algorithm` escolhe A ou C. Os blocos de A continuam; C acrescenta `train.embeddings`:

```yaml
train:
  algorithm: embeddings-ft
  seed: 42
  embeddings:
    encoder:
      source: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
      revision: "<sha do commit no hub — fixar na implementação>"
      max_length: 128
      device: auto            # auto | cpu | cuda
      num_threads: null       # inferência; null = padrão do torch
    context: {weight: 0.5}
    probe: {alpha: 1.0, solver: lsqr}
    finetune:
      epochs_max: 5
      patience: 2
      batch_size: 16
      lr_encoder: 2.0e-5
      lr_head: 1.0e-3
      weight_decay: 0.01
      warmup_ratio: 0.1
      dropout: 0.1
      freeze_word_embeddings: true
```

## 5. Artefatos

| Onde | Conteúdo |
|---|---|
| `models/_staging/<dataset_id>/embeddings-ft/<fp>/` | `baseline.joblib`, `candidate.joblib` (wrapper leve), `encoder/` (pesos `safetensors` + tokenizer + config), `train_manifest.json`, `eval.json` |
| `models/<model_version>/` | `model.joblib`, `encoder/`, `manifest.json` |

- `candidate.joblib` guarda o wrapper `EmbeddingMoodModel` (`transform/embeddings.py`): hiperparâmetros, `w`, pesos da cabeça (numpy) e o caminho **relativo** `encoder/`. **Não** serializa tensores do torch no pickle. A classe vive em `transform/embeddings.py` (módulo de biblioteca, nunca ponto de entrada `-m`), pelo mesmo motivo de `join_context_list` (ver `transform/features.py`).
- O wrapper expõe `predict(df) -> np.ndarray` com as mesmas colunas de A (`text_clean`, `context_clean`), para `run_inference` funcionar sem mudança.
- `manifest.json` ganha `encoder: {source, revision, max_length}` (extensão da data-model §4.4, no padrão de `training_fingerprint`). `algorithm = "embeddings-ft"`.
- **Fingerprint** (spec 06 §3.3): inclui `algorithm`, todo o bloco `train.embeddings` (com `revision`), hashes dos `.parquet` e `feature_spec_version`.
- Pesos ficam fora do git (`models/` já está em `.gitignore`).

## 6. Métricas e comparação

Sem novo gate: vale o da spec 06 (`MAE ≤ 0,9 × baseline`). Acrescenta ao `eval.json`:

- `comparison`: quando existir `eval.json` de `tfidf-ridge` para o **mesmo `dataset_id`**, `mae_delta_vs_tfidf`, sem influência no gate.
- `latency`: p50/p95 item a item (EV-R06), agora sobre C; e p95 com **histórico de 30 mensagens** (o corpus só chega a 20, então o pior caso de produção não aparece no teste).
- `train_manifest.json`: `probe_validation_mae` (etapa 1), `finetune_validation_mae` (melhor época), `epochs_run`, `device`, `torch_version`.

Limites a declarar no TCC: corpus sintético satura as métricas (A já tem MAE 0,041); o encoder ajustado pode aprender artefatos do gerador; bit-exact só em CPU.

## 7. Requisitos verificáveis

**Configuração e dependências**
- **FT-R01** QUANDO `train.algorithm = "embeddings-ft"`, o sistema DEVE ler os hiperparâmetros de `train.embeddings`; nenhum hiperparâmetro DEVE estar fixo no código.
- **FT-R02** QUANDO `torch`/`transformers` não estiverem instalados, `train.train` DEVE abortar com *exit* 2 (`FT_R02_DEPS_MISSING`), sem ler os `.parquet`. Os módulos de A DEVEM NÃO importar `torch`.
- **FT-R03** Dependências de C DEVEM ficar em `requirements-embeddings.txt`; `requirements.txt` DEVE permanecer inalterado.
- **FT-R04** QUANDO `revision` estiver ausente ou vazia, `train.train` DEVE abortar com *exit* 2.

**Vetorização e modelo**
- **FT-R05** A entrada do modelo DEVE ser `text_clean` e `context_clean` de `fs-1`; T1/T2/T3 DEVEM NÃO ser alterados.
- **FT-R06** O bloco contexto DEVE ser a média dos embeddings das mensagens anteriores e o vetor nulo QUANDO `context_clean = []`.
- **FT-R07** A codificação (tokenização, truncamento, pooling) DEVE ser a mesma função no treino e na inferência.
- **FT-R08** O contexto DEVE ser codificado sem gradiente durante o fine-tuning.
- **FT-R09** A matriz de palavras DEVE ficar congelada QUANDO `freeze_word_embeddings = true`.

**Treino**
- **FT-R10** O treino DEVE executar a sondagem linear (etapa 1) antes do fine-tuning e inicializar a cabeça com seus pesos.
- **FT-R11** O fine-tuning DEVE usar `validation` apenas para early stopping e restaurar o melhor checkpoint; `test` DEVE NÃO ser lido em `train.train`.
- **FT-R12** Todo componente estocástico (init da cabeça, dropout, embaralhamento, `DataLoader`) DEVE receber a `seed` do config.
- **FT-R13** QUANDO a perda ou uma previsão sobre `train` não for finita, `train.train` DEVE abortar sem persistir, *exit* 4 (`FT_R13_NON_FINITE`).
- **FT-R14** O `train_manifest.json` DEVE registrar `device`, `torch_version`, `probe_validation_mae`, `finetune_validation_mae` e `epochs_run`.
- **FT-R15** O fingerprint DEVE incluir `revision` do encoder; mudar `revision` DEVE gerar staging novo.

**Artefato, registro e inferência**
- **FT-R16** O staging e a `model_version` DEVEM conter o diretório `encoder/` autossuficiente; `registry.register` DEVE copiá-lo de forma atômica junto com `model.joblib`.
- **FT-R17** A inferência DEVE carregar o encoder com `local_files_only=True` e DEVE NÃO acessar a rede.
- **FT-R18** No carregamento, `infer.load_active_model` DEVE aceitar `algorithm = "embeddings-ft"`; QUANDO `encoder/` estiver ausente ou corrompido, DEVE falhar com `RegistryBrokenError`, e QUANDO as dependências faltarem, o serviço NÃO DEVE subir. Nenhum score de fallback (P3).
- **FT-R19** O serviço DEVE fazer uma predição de aquecimento na subida; a latência do primeiro request real DEVE NÃO incluir carga do modelo.
- **FT-R20** `evaluate.metrics` DEVE medir a latência p95 com histórico de 30 mensagens além da medição item a item de EV-R06.
- **FT-R21** QUANDO existir `eval.json` de `tfidf-ridge` para o mesmo `dataset_id`, `evaluate.metrics` DEVE gravar `comparison.mae_delta_vs_tfidf`, sem afetar o gate.

**Segurança**
- **FT-R22** Nenhum módulo novo DEVE importar `duckdb` (P1). `eval.json` e `train_manifest.json` DEVEM NÃO conter `text_clean` nem `context_clean`.

### 7.1 Catálogo de erros

| Código | Requisito | Exit |
|---|---|---|
| `FT_R02_DEPS_MISSING` | FT-R02 | 2 |
| `FT_R04_REVISION_MISSING` | FT-R04 | 2 |
| `FT_R13_NON_FINITE` | FT-R13 | 4 |
| `FT_ENCODER_UNAVAILABLE` (download falhou no treino) | — | 3 |
| `TN_R17_UNSUPPORTED_ALGORITHM` | ampliado: aceita `embeddings-ft` | 2 |

## 8. Ajustes nas specs existentes (mudanças pontuais, não reescrita)

| Spec | Ajuste |
|---|---|
| 04 | Nota: a vetorização de C mora em `transform/embeddings.py`; encoder fixado aqui (a ADR-0008 delegava a "spec de T3"). `fs-1` não muda. |
| 06 | §3.5/§10: remove "reservado, não implementado"; `SUPPORTED_ALGORITHMS` inclui `embeddings-ft`; nota de que `validation` decide early stopping em C (exceção a D4). |
| 07 | `register` copia `encoder/` (FT-R16); manifesto ganha `encoder`. |
| 08 | `SUPPORTED_ALGORITHMS`, carga do encoder local, aquecimento (FT-R17 a FT-R19). |
| 09 | Job de CI separado com `requirements-embeddings.txt`; testes de C marcados `@pytest.mark.embeddings`. |

## 9. Critérios de aceite

- **CA-01** Com `algorithm: embeddings-ft` e dataset aprovado, `train.train` sai com 0 e grava `candidate.joblib`, `encoder/`, `train_manifest.json` (FT-R14).
- **CA-02** Reexecução com o mesmo config em CPU → *no-op* idempotente (TN-R03); alterar `revision` → staging novo, anterior intacto (FT-R15).
- **CA-03** Em CPU, dois treinos com a mesma seed produzem as mesmas previsões no `validation` (tolerância 1e-6).
- **CA-04** Exemplo sem contexto → bloco contexto igual ao vetor nulo; exemplo com 1 e com 29 mensagens de contexto → média correta (teste com encoder de brinquedo).
- **CA-05** Paridade P4: para o mesmo `history`, a predição do wrapper carregado do disco é igual à do modelo em memória logo após o treino.
- **CA-06** Sem `torch` instalado, `train.train` → *exit* 2; a suíte de A continua verde sem `torch`.
- **CA-07** Serviço com modelo C ativo e `encoder/` removido → não sobe (`RegistryBrokenError`); nenhum request devolve score.
- **CA-08** Com a rede bloqueada (`HF_HUB_OFFLINE=1`), o serviço sobe e prediz (FT-R17).
- **CA-09** Cabeça inicializada pela etapa 1: a época 0 do fine-tuning tem MAE de validação igual ao da sondagem (± tolerância).
- **CA-10** `eval.json` de C traz `comparison.mae_delta_vs_tfidf` quando existe o de A no mesmo `dataset_id`, e omite o campo quando não existe.
- **CA-11** `test.parquet` não é aberto durante `train.train` (teste com *mock* de leitura).
- **CA-12** Registro copia `encoder/`; `model_version` resultante carrega e prediz sem o staging.

**Estratégia de teste:** os testes usam um **BERT minúsculo de config aleatória** (`BertConfig` com 2 camadas, d = 32, tokenizer de fixture), criado em `tmp_path`. Sem download, executa em segundos na CI. O encoder real é exercitado só em um teste manual/`slow` documentado.

## 10. Riscos e perguntas em aberto

| # | Pergunta / risco | Tipo | Quem responde |
|---|---|---|---|
| Q1 | **Hardware de treino:** há GPU disponível? Sem ela, o treino em CPU precisa caber num orçamento. Hipótese a medir na 1ª tarefa: ≤ 30 min em CPU de 8 threads (estimativa: ~22 mil passagens equivalentes por época, 5 épocas). | Bloqueante | Você |
| Q2 | **Orçamento de latência p95** de inferência (ADR-0008 deixou em aberto). Histórico cheio = até 30 codificações por request. | Bloqueante para promover C, não para implementar | Você / ADR posterior |
| Q3 | **`revision`** do encoder: fixar o SHA do hub na implementação. | Não bloqueante | Implementação |
| Q4 | **Aprovação da ADR-0009.** Esta spec não deve ser implementada antes dela. | Bloqueante | Você |
| R1 | Sobreajuste ao gerador sintético; diferença A × C pouco informativa. | Risco | TCC (declarar) |
| R2 | Só CPU garante pesos idênticos entre execuções. | Risco | TCC / manifesto registra `device` |

## 11. Fora de escopo

- BERTimbau ou outro encoder (troca de config, depois da validação; ADR-0009).
- Ajuste da matriz de palavras, LoRA/adapters, busca de hiperparâmetros.
- Quantização/ONNX, cache de embeddings por conversa, *batching* entre requests (evolução de latência).
- Ponderação por recência ou atenção sobre o contexto (ADR-0007, em aberto).
- Escolha da versão ativa (ADR posterior) e rótulos reais (v2).
- Mudanças em T1, T2, `fs-1` ou no dataset.

## 12. Checklist de implementação

Fase 6, branch `feat/ml-embeddings`; a spec é commitada **antes** do código.

- [ ] Aprovar ADR-0009 e esta spec (Status → Aceita)
- [ ] `requirements-embeddings.txt`; job de CI separado (FT-R02, FT-R03)
- [ ] Fixture do BERT minúsculo; testes vermelhos a partir de CA-01 a CA-12
- [ ] `transform/embeddings.py`: codificação, pooling, `EmbeddingMoodModel`, save/load com `encoder/` (FT-R05 a FT-R09, FT-R16)
- [ ] `train/train.py`: ramo C, etapa 1 e etapa 2, early stopping, checkpoint (FT-R10 a FT-R15)
- [ ] Spike: medir tempo de treino e latência com o encoder real; responder Q1 e Q2 com números
- [ ] `evaluate/metrics.py`: latência com 30 mensagens e `comparison` (FT-R20, FT-R21)
- [ ] `registry/registry.py`: copiar `encoder/` de forma atômica (FT-R16)
- [ ] `infer/predict.py`: algoritmo aceito, carga offline, aquecimento, falha sem dependências (FT-R17 a FT-R19)
- [ ] `configs/pipeline.yaml`: bloco `train.embeddings`
- [ ] Alinhar `CLAUDE.md` (§4, §8), `00-decisoes.md`, specs 04/06/07/08/09 e `data-model.md` §4.4

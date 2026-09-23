# 04 — Limpeza, Mascaramento e Extração de Features (camada `transform/`, T1 + T2 + T3)

**Status:** Aceita · **Versão da spec:** `tf-1` · **Data:** 2026-09-20
**Implementa:** [data-model.md §6](../../data-structure/data-model.md) (transformações T1, T2, T3) — este documento fecha T1 e T2 (marcadas "em aberto" lá) e detalha a implementação de T3 (decidida pela ADR-0008).
**Depende de:** [ADR-0001](../../decisions/ADR-0001-escala-do-humor.md) (escala), [ADR-0002](../../decisions/ADR-0002-origem-do-ground-truth.md) (grão de mensagem), [ADR-0007](../../decisions/ADR-0007-humor-por-conversa.md) (janela de 30 mensagens **da conversa**), [ADR-0008](../../decisions/ADR-0008-abordagens-de-modelo.md) (dois blocos de T3); [constitution.md](../../decisions/constitution.md) P3, P4, P5; [01-dataset-contract.md](01-dataset-contract.md) (campos de origem)
**Consumido por:** `transform/split.py` (T5 + montagem de exemplos, spec 05, uma chamada por exemplo de treino); `infer/predict.py` (uma chamada por requisição online, spec 08); `ingest/validate.py` reaproveita `count_pii` para medir DC-R14 (spec 02 §3.1, já registrado como dependência)
**Implementado em:** [transform/clean.py](../transform/clean.py) (T1), [transform/mask.py](../transform/mask.py) (T2), [transform/features.py](../transform/features.py) (T3)

---

## 1. Objetivo geral

Três funções puras, sem I/O, que transformam o texto de uma mensagem em algo que um modelo de regressão consegue consumir, sem nunca vazar dado pessoal:

1. **T1 — `clean_message`**: normaliza o texto bruto (espaços, quebras de linha, caracteres invisíveis), preservando tudo que carrega humor (caixa alta, pontuação repetida, emojis).
2. **T2 — `mask_pii`**: substitui CPF, telefone e e-mail por marcadores fixos, para que nenhum artefato de ML (`datasets/*`, `model.joblib`, logs) carregue PII (P5).
3. **T3 — `extract_features`**: aplica T1 + T2 a uma janela de mensagens e monta os dois blocos que o modelo consome — `text_clean` (a mensagem disparadora) e `context_clean` (até 29 mensagens anteriores da mesma conversa) — conforme a ADR-0008.

O que torna esta camada diferente das specs 02 e 03: **não é um script com `main()`, não tem *exit code*, não escreve arquivo.** É uma biblioteca de funções importadas por quem tem I/O — `transform/split.py` no treino, `infer/predict.py` na inferência online. A garantia que este documento vende não é "roda até o fim e produz um artefato", é **P4**: o mesmo código, na mesma versão, produz a mesma transformação nas duas trilhas. Qualquer divergência entre treino e inferência aqui é *training/serving skew* silencioso — o pior tipo de bug deste projeto, porque não lança exceção, só degrada a qualidade da previsão.

## 2. Arquitetura e fluxo de dados

### 2.1 Posição no fluxo — duas chamadas, dois contextos, o mesmo código

```
TREINO (offline, spec 05)                    INFERÊNCIA (online, spec 08)
corpus validado (spec 02)                    InferRequest.history (data-model §2.2)
       │ reconstrução da janela                     │ já montado pela API (ADR-0007)
       │ (T5, mesma regra da API)                    │
       ▼                                              ▼
  history: list[HistoryMessage]  ───────┬──────  history: list[HistoryMessage]
                                          │
                                          ▼
                              extract_features(history)   [T3]
                                    │        │
                        clean_message()   mask_pii()      [T1, T2]
                                    │        │
                                    ▼        ▼
                        { "text_clean": str, "context_clean": list[str] }
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                     ▼
        grava em datasets/<dataset_id>/*.parquet   alimenta o Pipeline sklearn
        (spec 05)                                  carregado do manifesto (spec 08)
```

A reconstrução da janela em si (T5 — "quais mensagens entram em `history`") **não é desta spec**. Ela é decidida pela ADR-0007 e implementada duas vezes por necessidade estrutural — pela API online, e por `transform/split.py` no treino, a partir do corpus (spec 05) — mas **converge** no mesmo ponto: uma lista de `HistoryMessage` de 1 a 30 itens, só `role = customer`, ordem cronológica crescente, disparadora por último. `extract_features` recebe exatamente esse formato dos dois lados e não sabe, nem precisa saber, de qual trilha veio.

### 2.2 Passo a passo lógico de `extract_features`

| Passo | O que faz |
|---|---|
| 1 | Valida a forma de `history`: não vazio, no máximo 30 itens, todos `role = "customer"` (§7, TR-R13–R15) |
| 2 | Para cada item, aplica `mask_pii(clean_message(item["text"]))` — T1 sempre antes de T2, nesta ordem |
| 3 | `text_clean` recebe o resultado do **último** item (a disparadora) |
| 4 | `context_clean` recebe a lista de resultados dos itens **anteriores**, na mesma ordem cronológica; lista vazia quando `len(history) == 1` |
| 5 | Devolve `{"text_clean": str, "context_clean": list[str]}` |

Não há passo de I/O, cache ou estado entre chamadas. Cada chamada é independente.

### 2.3 Por que T2 roda sempre, mesmo se o texto já chegou mascarado

O `data-model.md §2.2` deixa em aberto se `HistoryMessage.text` chega mascarado (se o `mood-api` decidir mascarar antes de persistir) ou cru. Essa decisão é externa ao `mood-ml` e pode mudar sem aviso. `extract_features` **não faz suposição sobre a origem do texto**: aplica `mask_pii` incondicionalmente, sempre. Isso só é seguro porque T2 é **idempotente** (TR-R09, §7) — mascarar um texto já mascarado é uma operação neutra, não um novo texto malformado. É a mesma garantia que torna T2 seguro de aplicar duas vezes por engano.

## 3. Requisitos técnicos e premissas

### 3.1 Conexões e autenticação

**Não há, categoricamente.** T1, T2 e T3 não abrem arquivo, não fazem chamada de rede, não leem variável de ambiente, não têm efeito colateral. É o requisito mais forte desta spec (TR-R06, TR-R20): qualquer I/O dentro dessas três funções seria, por si só, uma violação — não porque uma regra proíbe I/O em abstrato, mas porque a mesma função roda no caminho quente da inferência online (latência de milissegundos, ADR-0008) e dentro de um laço de até 40.000 iterações no treino (§3.5). Efeito colateral em qualquer um dos dois lugares é bug, não estilo.

### 3.2 Formato dos dados

Não há arquivo próprio desta camada — "formato de dados" aqui significa o **contrato de tipos Python** na fronteira de cada função:

| Função | Entrada | Saída |
|---|---|---|
| `clean_message` | `text: str` | `str` (nunca vazio se a entrada não for vazia — TR-R05) |
| `mask_pii` | `text: str` | `str` |
| `count_pii` | `text: str` | `dict[str, int]` — chaves `"cpf"`, `"phone"`, `"email"` |
| `extract_features` | `history: list[dict]` (esquema `HistoryMessage`: `message_id`, `role`, `text`) | `dict` com `"text_clean": str` e `"context_clean": list[str]` |

Quem grava em disco é sempre o chamador: `transform/split.py` escreve `text_clean`/`context_clean` em `datasets/<dataset_id>/*.parquet` (spec 05); `infer/predict.py` passa o resultado direto para o `Pipeline` do `model.joblib` (spec 08), sem persistir nada.

### 3.3 Estratégia de carga e idempotência

Não existe "carga" no sentido das specs 02/03 — não há artefato para versionar, sobrescrever ou destravar por hash. A idempotência aqui é de outra natureza: **pureza referencial**. `clean_message(x)`, `mask_pii(x)` e `extract_features(h)` DEVEM devolver exatamente o mesmo resultado toda vez que forem chamadas com o mesmo argumento, para sempre, no processo de treino e no processo de inferência.

É a definição operacional de P4 (*"T1 e T3 aplicados no treino e na inferência são o mesmo código, na mesma versão"*): não basta serem o mesmo código-fonte, precisam se comportar como função matemática — sem relógio, sem aleatoriedade, sem estado global, sem dependência de ordem de chamadas anteriores. Um `clean_message` que normalizasse acentos de forma diferente dependendo de um *locale* do sistema operacional, por exemplo, violaria isso silenciosamente entre uma máquina de treino e um servidor de inferência.

### 3.4 Estratégia de transformações

#### T1 — `clean_message`

| Regra | Efeito |
|---|---|
| Remove espaços nas pontas | `strip()` |
| Colapsa espaços em branco internos | Qualquer sequência de espaço, tab, quebra de linha ou retorno de carro vira um único espaço ASCII |
| Normaliza Unicode | Forma NFC (composição canônica), para que "á" sempre seja o mesmo *codepoint*, independente de como o gerador escreveu |
| Remove caracteres de controle invisíveis | Categorias Unicode `Cc` e `Cf` (exceto os já convertidos em espaço acima) — ex.: espaço de largura zero, caracteres de formatação bidirecional |
| **Preserva** caixa alta/baixa | O rubrica de humor do `dc-1` usa CAIXA ALTA como sinal de raiva (spec 01 §4) |
| **Preserva** pontuação repetida | `"!!!"`, `"???"` carregam intensidade |
| **Preserva** emojis | Sinal direto de humor |

**Garantia de não-vazio (TR-R05).** O contrato `dc-1` já garante `text` não vazio após `strip()` (DC-R01). Mas a limpeza adicional de T1 (remoção de caracteres de controle) poderia, em tese, esvaziar uma mensagem composta só de caracteres invisíveis que sobrevivem a um `strip()` simples. Nesse caso — que a `dc-1` não deveria produzir, mas que T1 não pode assumir impossível —, `clean_message` devolve o texto após **apenas** `strip()`, nunca uma string vazia. Um `text_clean` vazio quebraria a vetorização TF-IDF (documento vazio) sem produzir um erro claro.

**Fora de escopo, e por quê.** O placeholder do `data-model.md §6` para T1 também cita "remoção de mensagens automáticas". Isso não se aplica ao MVP: o corpus `dc-1` só tem `role ∈ {customer, agent}`, sem conceito de mensagem de sistema, e uma função que recebe uma `str` isolada não tem como inferir se ela é "automática". Fica registrado como não implementado (§10), não como esquecido.

#### T2 — `mask_pii` e `count_pii`

Cobertura: **CPF, telefone e e-mail**, exatamente os três tipos que a `dc-1` (DC-R14) exige que o corpus sintético contenha, e nada além disso — nem nomes próprios, nem endereço, nem outros identificadores (decisão do usuário, §8). Marcadores fixos: `<CPF>`, `<TEL>`, `<EMAIL>`.

Os padrões, aplicados **nesta ordem**, cada um sobre o resultado do anterior:

| Ordem | Tipo | Padrão (informal) | Nota |
|---|---|---|---|
| 1 | E-mail | `[usuário]@[domínio].[tld]` | Menos ambíguo, roda primeiro para não deixar `@` interferir nos padrões seguintes |
| 2 | CPF formatado | `\d{3}\.\d{3}\.\d{3}-\d{2}` | Só a forma pontuada (`123.456.789-01`). Ver limitação abaixo |
| 3 | Telefone, formas com marcador | `+55 (DDD) 9XXXX-XXXX`, `(DDD) 9XXXX-XXXX`, `DDD 9XXXX-XXXX`, `9XXXX-XXXX` | Cobre celular (9 dígitos) e fixo (8 dígitos), com ou sem `+55`/parênteses/espaço |
| 4 | Telefone, sequência bruta | 10 ou 11 dígitos consecutivos, sem separador | Único padrão puramente numérico |

**Limitação declarada: CPF não formatado.** Uma sequência de 11 dígitos sem pontuação (`12345678901`) é ambígua entre CPF e telefone celular com DDD — não há como diferenciar por regex sem contexto. A ordem acima resolve o empate em favor de **telefone** (passo 4 roda por último e captura o que sobrar). Um CPF gerado sem formatação **não é mascarado como `<CPF>`**; se tiver exatamente o formato de um telefone válido, ainda assim vira `<TEL>` — mascarado, mas com o tipo errado no `count_pii`. Isso não compromete DC-R14 (que mede *que* houve mascaramento, não a classificação), mas deve ser declarado no TCC como limitação conhecida de T2.

**Idempotência obrigatória (TR-R09).** `mask_pii(mask_pii(x)) == mask_pii(x)` para qualquer `x` — nenhum marcador (`<CPF>`, `<TEL>`, `<EMAIL>`) é reconhecido pelos padrões acima como um novo alvo de mascaramento, então uma segunda passada não altera nada. Essa propriedade é o que permite T3 aplicar T2 sem checar se o texto já veio mascarado (§2.3).

**`count_pii` não é uma função separada da detecção de `mask_pii`.** Conta ocorrências usando os **mesmos** padrões, na mesma ordem, sobre o texto **original** (não mascarado) — TR-R10. Existe para servir dois consumidores sem duplicar a lógica de detecção: o próprio `mask_pii` (que precisa saber onde estão os alvos para substituí-los) e `ingest/validate.py`, que mede `pii_ratio`/`pii_by_kind` para DC-R14 (spec 02 §3.1) sobre o corpus **cru**, antes de qualquer mascaramento acontecer.

#### T3 — `extract_features`

Já especificada pela ADR-0008 no nível de contrato; esta seção fixa o comportamento de `extract_features` como função:

- Validação de forma antes de qualquer processamento (§2.2, passo 1): `history` não vazio, ≤ 30 itens, todos `role = "customer"`. Uma violação é bug de quem monta a janela (API ou `split.py`), não um caso a tratar silenciosamente — `extract_features` levanta `ValueError` (§4).
- T1 então T2, nesta ordem, em **cada** item — nunca só na disparadora.
- `text_clean` = resultado do último item. `context_clean` = resultados dos itens anteriores, ordem preservada, lista vazia se `len(history) == 1`.
- Nenhuma truncagem silenciosa: um `history` com mais de 30 itens é erro do chamador (violaria a ADR-0007), não motivo para descartar os excedentes calados.

**Fora desta spec, mencionado para contexto**: a vetorização de `text_clean`/`context_clean` (TF-IDF na abordagem A, embeddings na C) não acontece aqui — vive dentro do `Pipeline` serializado em `model.joblib` (ADR-0008, spec 06/07). `extract_features` para no texto, nunca produz vetor.

### 3.5 Volumetria estimada e frequência de execução

Não há volumetria de "arquivo processado" — há **frequência de chamada**:

| Trilha | Frequência | Ordem de grandeza (MVP) |
|---|---|---|
| Treino (`split.py`, spec 05) | Uma chamada de `extract_features` por exemplo do dataset | 2.000–40.000 chamadas por execução (herda a volumetria da spec 02 §3.4) |
| Inferência (`infer/predict.py`, spec 08) | Uma chamada por requisição `POST /internal/v1/infer` | Sem volumetria própria no MVP (sem tráfego real ainda) |

**Custo por chamada.** T1 e T2 operam sobre no máximo 1000 caracteres por mensagem (limite da `dc-1`); regex sobre uma string desse tamanho é da ordem de microssegundos. `extract_features` repete isso até 30 vezes (o tamanho máximo da janela). Mesmo no teto, o custo por chamada fica na casa de baixos milissegundos — dentro do orçamento de latência que a ADR-0008 deixa em aberto, mas de sobra para não ser o gargalo.

**Nota para a spec 05, não resolvida aqui.** No treino, a reconstrução ingênua da janela reprocessa a mesma mensagem várias vezes: uma mensagem que aparece como contexto em 29 exemplos subsequentes passa por T1+T2 29 vezes. Como T1 e T2 são puras, o resultado é sempre igual — **memoizar por `message_id`** é uma otimização segura e correta, mas é decisão de orquestração de `transform/split.py` (que decide como percorrer o corpus), não desta spec, que só garante que a função pode ser chamada quantas vezes forem necessárias sem produzir resultado diferente.

## 4. Tratamento de erros e resiliência

### 4.1 Por que não há política de retry

Retry existe para lidar com falha transitória de I/O — rede instável, arquivo temporariamente bloqueado, disco cheio. T1, T2 e T3 não têm I/O (§3.1): uma chamada que falha, falha porque a entrada viola o contrato de tipos ou de forma, e vai falhar de novo com a mesma entrada, sempre. Não há nada a retentar.

### 4.2 O que cada função levanta

| Função | Situação | Exceção |
|---|---|---|
| `clean_message` | Entrada não é `str` | `TypeError` |
| `mask_pii`, `count_pii` | Entrada não é `str` | `TypeError` |
| `extract_features` | `history` vazio | `ValueError` |
| `extract_features` | `history` com mais de 30 itens | `ValueError` |
| `extract_features` | algum item com `role != "customer"` | `ValueError` |
| `extract_features` | item sem a chave `text` | `KeyError` |

Nenhuma das três funções captura sua própria exceção para devolver um valor de reserva (`""`, `None`, texto original sem mascarar). **Isso é P3 aplicado no nível de função**: na inferência online, uma falha aqui precisa propagar até `infer/predict.py`, que a converte em `inference_failures` + HTTP `4xx`/`5xx` — nunca em um score calculado sobre um texto mal processado ou parcialmente mascarado. Engolir a exceção dentro de T1/T2/T3 "para não quebrar o pipeline" seria reintroduzir, por uma porta lateral, exatamente o fallback que P3 proíbe.

### 4.3 Alertas e notificações

**Não se aplicam a esta camada.** T1/T2/T3 não são um processo com ciclo de vida próprio — são chamadas dentro de outro processo. Quem decide se uma falha aqui vira alerta é o chamador: no treino, `transform/split.py` decide se aborta a execução (spec 05); na inferência, o comportamento já está definido pela ADR-0004 (nenhum canal externo, falha vira `inference_failures` e o próximo score bem-sucedido resolve). Esta spec não introduz um canal novo.

## 5. Logs e observabilidade

### 5.1 Por que T1/T2/T3 não logam nada

Registrar um evento de log por chamada de `clean_message` ou `extract_features` produziria uma linha de log por mensagem processada — até 40.000 por execução de treino, e uma por requisição online. É ruído, não observabilidade, e o próprio ato de logar seria um efeito colateral dentro de uma função que a §3.3 exige pura. **A responsabilidade de observabilidade em volume é do chamador**, que já teria de agregar de qualquer forma.

### 5.2 O que esta camada expõe para o chamador logar

Em vez de logar, T2 expõe `count_pii` como uma função de primeira classe (§3.4) precisamente para que `transform/split.py` (spec 05) e `ingest/validate.py` (spec 02, já em produção) possam agregar contagens de PII por execução — `messages_masked`, `pii_by_kind` — sem duplicar a lógica de detecção. É a mesma filosofia da spec 02 (`records_processed`/`records_rejected` por fase): a contagem em massa fica com quem já está iterando sobre o lote, e a função pura apenas fornece o dado por chamada.

### 5.3 Testabilidade como substituto de observabilidade em produção

Por serem funções puras, T1/T2/T3 são testáveis por casos concretos de entrada/saída, sem *mock* de I/O nem *fixture* de arquivo — só pares (entrada, saída esperada). É esse conjunto de casos (§9, CA-02 a CA-06) que garante a qualidade desta camada, no lugar de logs de execução.

## 6. Segurança e conformidade

### 6.1 T2 é o único ponto de mascaramento de todo o pipeline

Por P4, T1/T2/T3 são o mesmo código no treino e na inferência — o que implica que **T2 é o único lugar onde PII é mascarada em todo o `mood-ml`**, tanto no `text_clean`/`context_clean` gravado em `datasets/*` (que alimenta o treino) quanto no texto que entra no `Pipeline` de inferência em tempo real. Não existe uma segunda implementação de mascaramento em `split.py` nem em `infer/predict.py`: os dois chamam `extract_features`, que chama `mask_pii` internamente.

Isso fecha a lacuna que a spec 02 §7.2 deixava em aberto: `ingest/validate.py` **conta** PII (via `count_pii`) mas não mascara; **é esta spec** que mascara, e faz isso exatamente uma vez, no ponto em que o texto entra na fronteira entre corpus/API e artefato de modelo.

### 6.2 Base LGPD/GDPR

Mesma base das specs 02 e 03: o corpus de origem é sintético (`dc-1`, DC-R13), sem titular real — a PII presente é fictícia e existe para **exercitar** T2, não para proteger uma pessoa real. Ainda assim, T2 é implementada com o mesmo rigor que teria sobre dado real, porque:

- é o mesmo código que rodaria sobre um snapshot real pós-MVP (`data-model §4.2`), quando a distinção deixa de ser hipotética;
- `text_clean`/`context_clean` são o que efetivamente sai do perímetro do corpus e passa a viver em `datasets/*`, `model.joblib` e em qualquer log de treino — os artefatos mais fáceis de copiar, versionar e vazar (mesmo raciocínio de P5 já registrado na spec 02 §7.1).

### 6.3 P1

Nenhuma referência a `duckdb` ou caminho `.duckdb` nesta camada — verificação de *lint* idêntica às specs anteriores.

## 7. Requisitos verificáveis

**T1**
- **TR-R01** `clean_message` DEVE remover espaços nas pontas e colapsar qualquer sequência de espaço em branco Unicode interno (incluindo quebras de linha e tabulações) em um único espaço ASCII.
- **TR-R02** `clean_message` DEVE preservar caixa, pontuação repetida e emojis sem alteração.
- **TR-R03** `clean_message` DEVE aplicar normalização Unicode NFC ao resultado.
- **TR-R04** `clean_message` DEVE remover caracteres de controle Unicode (categorias `Cc`, `Cf`) não capturados por TR-R01.
- **TR-R05** QUANDO a aplicação de TR-R01/TR-R04 resultaria em string vazia a partir de uma entrada não vazia, `clean_message` DEVE devolver o texto após apenas `strip()`, nunca uma string vazia.
- **TR-R06** `clean_message` DEVE ser uma função pura: sem I/O, sem estado global, sem dependência de horário ou *locale* do sistema.

**T2**
- **TR-R07** `mask_pii` DEVE substituir e-mails por `<EMAIL>`, CPFs formatados por `<CPF>` e telefones (formatados ou em sequência bruta de 10–11 dígitos) por `<TEL>`.
- **TR-R08** A detecção DEVE seguir a ordem e-mail → CPF formatado → telefone (formas com marcador) → telefone (sequência bruta), cada passo sobre o resultado do anterior.
- **TR-R09** `mask_pii` DEVE ser idempotente: `mask_pii(mask_pii(x)) == mask_pii(x)` para qualquer `x`.
- **TR-R10** `count_pii` DEVE usar exatamente os mesmos padrões de `mask_pii`, sobre o texto original, sem reimplementação paralela.
- **TR-R11** CPF sem formatação (11 dígitos sem pontuação) NÃO é reconhecido como `<CPF>` — é classificado como telefone se tiver a forma de um (limitação declarada, §3.4).
- **TR-R12** Nem `mask_pii` nem `count_pii` DEVEM detectar nomes próprios ou qualquer PII fora de CPF/telefone/e-mail.

**T3**
- **TR-R13** QUANDO `history` estiver vazio, `extract_features` DEVE levantar `ValueError`.
- **TR-R14** QUANDO `history` tiver mais de 30 itens, `extract_features` DEVE levantar `ValueError`.
- **TR-R15** QUANDO algum item de `history` tiver `role != "customer"`, `extract_features` DEVE levantar `ValueError`.
- **TR-R16** `extract_features` DEVE aplicar `mask_pii(clean_message(text))` a todo item de `history`, sem exceção.
- **TR-R17** `text_clean` DEVE ser o resultado de T1+T2 sobre o último item de `history`.
- **TR-R18** `context_clean` DEVE ser a lista de T1+T2 sobre os itens anteriores ao último, em ordem crescente, e DEVE ser `[]` quando `len(history) == 1`.
- **TR-R19** `extract_features` DEVE ser a única função de montagem dos blocos texto/contexto, importada tanto pelo treino (spec 05) quanto pela inferência (spec 08).

**Erros e segurança**
- **TR-R20** Nenhuma das três funções DEVE capturar sua própria exceção para devolver um valor de reserva — toda falha propaga ao chamador (P3).
- **TR-R21** Nenhuma das três funções DEVE realizar I/O de qualquer tipo (arquivo, rede, relógio de sistema além do necessário para uso interno determinístico).
- **TR-R22** O módulo `transform/` DEVE NÃO importar `duckdb` nem referenciar caminho `.duckdb` (P1).

## 8. Decisões deste documento

| # | Decisão | Motivo | Consequência assumida |
|---|---|---|---|
| D1 | **Escopo do T2:** só CPF, telefone, e-mail via regex; sem detecção de nomes | Decisão do usuário: cobre exatamente o exigido pela DC-R14, sem depender de NER/modelo pesado | Nomes próprios em texto livre não são mascarados — limitação a declarar no TCC, já prevista como "em aberto" no `data-model.md §6` |
| D2 | **CPF não formatado não é reconhecido como CPF** | Ambiguidade irresolvível por regex com telefone (mesmos 11 dígitos) | Um CPF sintético sem pontuação pode virar `<TEL>` ou não ser mascarado, dependendo da forma — mascaramento ainda ocorre na maioria dos casos plausíveis, classificação pode errar |
| D3 | **`count_pii` e `mask_pii` compartilham os mesmos padrões** | Evita a divergência de detecção entre o que a spec 02 mede (DC-R14) e o que T2 efetivamente mascara | Uma mudança nos padrões de detecção afeta os dois consumidores ao mesmo tempo — efeito desejado, não colateral |
| D4 | **Memoização de T1/T2 por `message_id` fica fora desta spec** | É uma decisão de orquestração de `transform/split.py` (como percorrer o corpus), não da função pura em si | A spec 05 precisa decidir se implementa cache; sem ele, o custo de CPU é maior mas o resultado é idêntico (funções puras) |

## 9. Critérios de aceite

- **CA-01** `clean_message`, `mask_pii`, `count_pii` e `extract_features` não têm nenhuma chamada de I/O (arquivo, rede) — verificado por *lint*/teste estático sobre o módulo.
- **CA-02** Tabela de casos para `clean_message`: `"  Oi   tudo bem?  "` → `"Oi tudo bem?"`; `"ABSURDO!!!"` → inalterado; texto com emoji → inalterado; texto só com caracteres de controle → resultado igual ao `strip()` puro (TR-R05).
- **CA-03** Tabela de casos para `mask_pii`: e-mail simples, CPF formatado, telefone com `(DDD)`, telefone com `+55`, telefone em sequência bruta de 11 dígitos, texto sem PII (inalterado), texto com os três tipos na mesma mensagem (todos mascarados).
- **CA-04** Teste de propriedade: para uma amostra de textos gerados aleatoriamente com PII sintética, `mask_pii(mask_pii(x)) == mask_pii(x)` (TR-R09).
- **CA-05** Teste de consistência: para a mesma amostra, `count_pii(x)` reporta pelo menos uma ocorrência para cada tipo que `mask_pii(x)` efetivamente mascarou.
- **CA-06** Casos de borda de `extract_features`: `history` com 1 item (→ `context_clean == []`), `history` com 30 itens (não levanta erro), `history` vazio (`ValueError`), `history` com 31 itens (`ValueError`), item com `role = "agent"` (`ValueError`).
- **CA-07** Teste de paridade: chamar `extract_features` duas vezes com o mesmo `history` (uma simulando o caminho de treino, outra o de inferência) produz resultados idênticos.
- **CA-08** Teste de P1: nenhuma ocorrência de `duckdb` ou `.duckdb` em `transform/`.

## 10. Fora de escopo

- **Reconstrução da janela de histórico (T5)** — o que entra em `history` antes de chegar a `extract_features`. Decidido pela ADR-0007; implementado pela API (online) e por `transform/split.py` a partir do corpus (offline, spec 05). O teste de contrato que prova que as duas reconstruções seguem a mesma regra está em [05-split.md](05-split.md), requisito SP-R20 — é lá que a reconstrução acontece; esta spec só consome o resultado já pronto.
- **Vetorização** (TF-IDF ou embeddings) — vive dentro do `Pipeline` serializado, fora de `transform/` (ADR-0008, specs 06/07).
- **Detecção de nomes próprios ou outros tipos de PII** — decisão do usuário (D1); registrado como limitação, não como pendência a resolver depois.
- **Remoção de mensagens automáticas** — mencionado no placeholder do `data-model.md §6` para T1, mas inaplicável ao MVP (corpus não tem esse conceito).
- **Memoização/cache de T1+T2 por `message_id`** durante a construção do dataset — decisão de `transform/split.py` (spec 05), não desta camada.
- **Local de mascaramento no `mood-api`** (se a API também mascara antes de persistir) — fora do `mood-ml`, e irrelevante para esta spec por causa da idempotência de T2 (§2.3).

## 11. Checklist de implementação

- [ ] `clean_message` — normalização de espaços, NFC, remoção de caracteres de controle, guarda de não-vazio (TR-R01 a TR-R06)
- [ ] Catálogo de padrões de `mask.py` — e-mail, CPF formatado, telefone (todas as formas), na ordem definida (TR-R07, TR-R08)
- [ ] `count_pii` construído sobre os mesmos padrões de `mask_pii` (TR-R10)
- [ ] Teste de propriedade de idempotência de `mask_pii` (TR-R09, CA-04)
- [ ] `extract_features` — validação de forma, montagem de `text_clean`/`context_clean` (TR-R13 a TR-R19)
- [ ] Tabela de casos de `clean_message` e `mask_pii` como testes parametrizados (CA-02, CA-03)
- [ ] Casos de borda de `extract_features`: `len(history)` = 1, 30, 0, 31; `role` inválido (CA-06)
- [ ] Teste de paridade treino/inferência sobre o mesmo `history` (CA-07)
- [ ] Teste estático de ausência de I/O e de `duckdb`/`.duckdb` (CA-01, CA-08)

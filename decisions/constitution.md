# Constituição — Análise de Humor do Cliente

Princípios invioláveis do projeto. Toda spec, plano e implementação deve
respeitá-los. Uma feature que exija violar um princípio não é implementada:
primeiro se escreve um ADR que altere o princípio.

Cada princípio declara **o que proíbe** e **como verificar** que está sendo
respeitado. Um princípio sem verificação é uma intenção, não uma regra.

---

## P1 — Escritor único no DuckDB

O processo do `mood-api` é o único que abre o arquivo `.duckdb` em modo de
escrita. O `mood-ml` nunca lê nem escreve o banco: recebe dados por HTTP na
trilha online e por arquivo exportado na trilha offline.

**Por quê:** o DuckDB usa lock exclusivo de arquivo por processo. Escrita
concorrente multi-processo não é suportada, e a alternativa (abrir em
read-only no ML) criaria um acoplamento a um detalhe de armazenamento que a
arquitetura quer manter trocável.

**Verificação:** nenhum import de `duckdb` fora de `mood-api/`. Nenhuma
referência a caminho `.duckdb` em `mood-ml/`.

---

## P2 — `mood_scores` é append-only

Linhas de `mood_scores` nunca são atualizadas nem apagadas. O humor atual do
cliente é **derivado** por view (`v_customer_mood_latest`,
`v_conversation_mood_latest`), nunca armazenado como estado.

**Por quê:** humor é um evento no tempo, não um atributo do cliente.
Sobrescrever destrói a série histórica, impede comparar versões de modelo
sobre a mesma mensagem e inviabiliza a reinferência retroativa.

**Verificação:** nenhum `UPDATE` ou `DELETE` sobre `mood_scores` no código.
Reprocessamento gera linha nova com novo `request_id`.

---

## P3 — Nunca gravar score de fallback

Uma inferência que falha gera linha em `inference_failures`, nunca um score
padrão em `mood_scores`. O `mood-ml` responde erro com status HTTP `4xx`/`5xx`
e corpo `{"error_code", "detail"}`.

**Por quê:** um score neutro de fallback é indistinguível de um humor neutro
real. A falha desaparece silenciosamente, o frontend exibe um emoji errado e
a métrica de qualidade do modelo fica contaminada por dados que o modelo nunca
produziu.

**Verificação:** `mood_scores` não aceita `model_version` de placeholder
(`untrained`). Todo caminho de erro na API escreve em `inference_failures`.

---

## P4 — T1 e T3 são o mesmo código no treino e na inferência

A limpeza de texto (T1) e a extração de features (T3) executam exatamente o
mesmo código, na mesma versão, na trilha offline de treino e na trilha online
de inferência. O vínculo é declarado por `feature_spec_version` no manifesto
do modelo.

**Por quê:** é a origem clássica de training/serving skew. Se o treino
normaliza acentos e a inferência não, o modelo recebe em produção uma
distribuição que nunca viu, e a degradação é silenciosa: não há erro, só
previsão pior. Duplicar a lógica em dois lugares garante que as duas cópias
divirjam com o tempo.

**Verificação:** T1 e T3 vivem num único módulo, importado pelo pipeline de
treino e pelo endpoint de inferência. O `feature_spec_version` gravado no
manifesto é conferido no carregamento do modelo; divergência impede subir o
serviço.

---

## P5 — Nenhum dado com PII sai do DuckDB

Qualquer texto não mascarado (`messages.text`) permanece exclusivamente no
banco operacional. Snapshots, datasets, manifestos, logs e
`inference_failures.detail` não podem conter PII nem trechos de conversa.
`customer_id` é sempre pseudônimo e nunca telefone ou e-mail em claro.

**Por quê:** conversas de atendimento são dados sensíveis sob a LGPD. Artefatos
de ML são copiados, versionados e compartilhados com muito mais facilidade que
um banco, e uma vez que o dado vaza para um `.parquet` não há como recolhê-lo.
A política de retenção só é aplicável ao que está num lugar só.

**Verificação:** o esquema exportado em snapshots nunca inclui `text` sem
mascaramento (T2). `mood-api/data/`, `mood-ml/data/` e `mood-ml/models/` estão
no `.gitignore`. Logs de erro carregam `message_id`, nunca `text`.

---

## Alterando esta constituição

Um princípio só muda por ADR que o cite explicitamente e declare quais
implementações existentes passam a estar em violação. O ADR anterior é marcado
como *superseded*, nunca editado.

# 01 — Contrato do Dataset Rotulado (corpus sintético)

**Status:** Aceita · **Versão do contrato:** `dc-1` · **Data:** 2026-09-18
**Especializa:** [data-model.md §4.1](../../data-structure/data-model.md). Onde este documento for mais restrito, ele prevalece para corpora `dc-1`.
**Depende de:** ADR-001, ADR-002, ADR-003 ([00-decisoes.md](00-decisoes.md))
**Consumido por:** `ingest/` (validação), `labels/` (T4)

---

## 1. Objetivo

Definir o formato, as regras e a composição mínima do dataset rotulado que serve de ponto de partida para o treino do modelo v1. Um arquivo que cumpre este contrato passa pela etapa `ingest + validate` sem ajustes manuais.

## 2. Formato do arquivo

- JSON Lines, UTF-8, sem BOM. Uma linha = uma **mensagem**.
- Nome: `<corpus_id>.jsonl`. Destino: `mood-ml/data/raw/synthetic/`.
- Linhas ordenadas por `conversation_id`, depois por `sent_at`. É recomendado, e não obrigatório: o ingest reordena.

## 3. Esquema

| Campo | Tipo | Obrigatório | Regra |
|---|---|---|---|
| `corpus_id` | string | sim | Igual em todas as linhas e igual ao nome do arquivo. Padrão `syn-AAAA-MM-DD-<letra>` |
| `conversation_id` | string | sim | Prefixo `syn-` |
| `customer_id` | string | sim | Prefixo `syn-` |
| `message_id` | string | sim | Prefixo `syn-`. Único no corpus |
| `role` | string | sim | `customer` \| `agent` |
| `text` | string | sim | Não vazio após `strip()`. Máximo de 1000 caracteres |
| `sent_at` | string | sim | ISO-8601 UTC com sufixo `Z` (ex.: `2026-09-10T14:03:22Z`) |
| `persona` | string | sim* | `snake_case`. Constante para o mesmo `customer_id` |
| `generated_label` | number \| null | sim* | `customer`: um de `-1.0, -0.5, 0.0, 0.5, 1.0`. `agent`: `null` |

\* No data-model esses campos são anuláveis. No `dc-1` são obrigatórios: `persona` sustenta a análise de viés, e `generated_label` é o ground truth.

## 4. Semântica do rótulo (rubrica)

O rótulo mede o **humor do cliente expresso na própria mensagem**, na escala da ADR-001.

| Nível | Nome | Sinais típicos | Exemplo |
|---|---|---|---|
| -1.0 | muito_negativo | Raiva, ameaça (cancelar, reclamar publicamente), ofensa, CAIXA ALTA, vários "!!!" | "É um ABSURDO, ninguém resolve nada. Vou cancelar e expor vocês." |
| -0.5 | negativo | Frustração, cobrança, insatisfação sem hostilidade | "Já é a segunda vez que peço e o ar ainda não foi consertado." |
| 0.0 | neutro | Informativo, pergunta objetiva, confirmação | "Qual o horário do check-out?" |
| 0.5 | positivo | Cordialidade, agradecimento, satisfação | "Obrigado, deu certo!" |
| 1.0 | muito_positivo | Elogio entusiasmado, intenção de voltar ou recomendar | "Vocês são incríveis, melhor estadia que já tive!!" |

Em mensagens ambíguas, use o nível mais provável. Não crie níveis intermediários.

## 5. Requisitos

Formato: **QUANDO** condição, o validador **DEVE** ação. Cada requisito vira um teste em `tests/`.

**Esquema e integridade**
- **DC-R01** QUANDO uma linha não for um JSON válido ou violar o esquema da §3, o validador DEVE rejeitar o corpus e informar o número da linha.
- **DC-R02** QUANDO um `message_id` se repetir, o validador DEVE rejeitar o corpus.
- **DC-R03** QUANDO algum ID não tiver o prefixo `syn-`, o validador DEVE rejeitar o corpus.
- **DC-R04** QUANDO um `conversation_id` aparecer com mais de um `customer_id`, o validador DEVE rejeitar o corpus.
- **DC-R05** QUANDO um `customer_id` aparecer com mais de uma `persona`, o validador DEVE rejeitar o corpus.
- **DC-R06** QUANDO houver `sent_at` repetido ou fora de ordem dentro de uma conversa, o validador DEVE rejeitar o corpus. A ordem é estritamente crescente.

**Rótulo**
- **DC-R07** QUANDO `role = customer` e `generated_label` for nulo ou estiver fora dos 5 níveis, o validador DEVE rejeitar o corpus.
- **DC-R08** QUANDO `role = agent` e `generated_label` não for nulo, o validador DEVE rejeitar o corpus.

**Composição mínima** (garante que o split por cliente seja viável)
- **DC-R09** O corpus DEVE ter ≥ 150 clientes, ≥ 300 conversas e ≥ 2000 mensagens `customer`.
- **DC-R10** Cada nível de rótulo DEVE representar ≥ 10% das mensagens `customer`.
- **DC-R11** O corpus DEVE ter ≥ 4 personas, cada uma com ≥ 10% dos clientes.
- **DC-R12** Cada mensagem `customer` DEVE ter o rótulo inferível pelo próprio texto (ADR-003). É verificado por amostragem manual de 50 mensagens, e não automaticamente.

**Realismo e privacidade**
- **DC-R13** O corpus NÃO DEVE conter dados pessoais reais.
- **DC-R14** Entre 5% e 15% das mensagens `customer` DEVEM conter PII **fictícia** (CPF, telefone, e-mail), para exercitar o mascaramento T2.
- **DC-R15** Cada conversa DEVE ter entre 4 e 20 mensagens, com os dois papéis presentes.
- **DC-R16** Pelo menos 30% das conversas DEVEM ter variação de humor (≥ 2 níveis distintos entre as mensagens `customer`).

Idioma: pt-BR. Domínio: atendimento a hóspedes de acomodações de temporada (check-in, limpeza, manutenção, reservas, pagamentos).

## 6. Critérios de aceite

- **CA-01** Um arquivo em conformidade passa na validação e gera o relatório `validation_report.json` (contagens, distribuição de rótulos por nível e por persona, hash sha256).
- **CA-02** Para cada requisito DC-R01 a DC-R11 e DC-R14 a DC-R16, existe uma fixture inválida em `tests/fixtures/` que faz o validador falhar com um código de erro específico.
- **CA-03** A fixture válida `tests/fixtures/sample.jsonl` tem cerca de 50 linhas e passa nas regras de esquema. Ela fica isenta das regras de composição (DC-R09 a DC-R11) por uma flag `--skip-composition`.

## 7. Exemplo

```jsonl
{"corpus_id":"syn-2026-09-18-a","conversation_id":"syn-conv-0001","customer_id":"syn-cust-0001","message_id":"syn-msg-000001","role":"customer","text":"Oi, qual a senha do wi-fi?","sent_at":"2026-09-10T14:00:00Z","persona":"objetivo","generated_label":0.0}
{"corpus_id":"syn-2026-09-18-a","conversation_id":"syn-conv-0001","customer_id":"syn-cust-0001","message_id":"syn-msg-000002","role":"agent","text":"Olá! A senha está no manual da porta da geladeira.","sent_at":"2026-09-10T14:01:10Z","persona":"objetivo","generated_label":null}
{"corpus_id":"syn-2026-09-18-a","conversation_id":"syn-conv-0001","customer_id":"syn-cust-0001","message_id":"syn-msg-000003","role":"customer","text":"Não tem manual nenhum aqui, estou sem internet há 2 horas.","sent_at":"2026-09-10T14:02:30Z","persona":"objetivo","generated_label":-0.5}
```

## 8. Fora de escopo

- Snapshots de dados reais (data-model §4.2): terão um contrato próprio.
- Rótulos `manual`: o protocolo de anotação será definido na v2.
- Como o corpus é gerado (LLM, templates ou manual). O contrato cobra apenas o resultado.

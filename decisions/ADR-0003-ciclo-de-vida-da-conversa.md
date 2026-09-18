# ADR-0003: Ciclo de vida da conversa

- **Status:** superseded por [ADR-0006](ADR-0006-sem-encerramento-no-mvp.md)
- **Data:** 2026-09-17
- **Princípios tocados:** P1
- **Referência:** `data-structure/data-model.md`, seções 3.2 e 9

---

## Contexto

As colunas `conversations.status` e `conversations.closed_at` existem no
esquema, mas nenhuma regra define quando uma conversa nasce nem quando termina.
Sem isso, `status` seria uma coluna sempre nula e o frontend não teria como
separar atendimentos ativos de encerrados.

A decisão também interage com o humor: se o encerramento delimitasse o contexto
de inferência, fechar uma conversa zeraria o histórico do cliente. O ADR-0005
mostra que não é o caso, e essa independência precisa ficar explícita.

## Decisão

**Abertura.** Uma conversa é criada pelo primeiro `ingest` com
`role = 'customer'` para um `conversation_id` ainda inexistente, com
`status = 'open'` e `closed_at = null`. Uma mensagem `agent` para um
`conversation_id` desconhecido é rejeitada com `422`: o atendente não inicia
atendimento, responde a um.

**Encerramento.** Por ação explícita do atendente no frontend, através de uma
rota nova:

```
POST /v1alpha1/conversation/{conversation_id}/close
→ 200 {"conversation_id", "status": "closed", "closed_at"}
→ 404 conversa inexistente
```

A rota é **idempotente**: fechar uma conversa já fechada devolve `200` com o
`closed_at` original, sem sobrescrever.

**Reabertura.** Uma mensagem `customer` numa conversa fechada **reabre** a
conversa: `status` volta para `open` e `closed_at` volta para `null`. A
mensagem é persistida e dispara inferência normalmente.

**Domínio de `status`:** `open` | `closed`. Quarentena de inferência (ADR-0004)
**não** entra nesta coluna.

## Alternativas consideradas

### Encerramento automático por inatividade

Um job fecharia conversas sem mensagem há N horas.

**Descartada** por exigir agendador, que o MVP não tem, e por escolher um N
arbitrário sem dado que o sustente. Fica como sucessor natural quando houver
orquestração: a regra manual gera justamente o dado necessário para calibrar N.

### Encerramento terminal, sem reabertura

Mensagem em conversa fechada seria `409`, obrigando o cliente a abrir novo
`conversation_id`.

**Descartada** porque rejeitar significa **descartar uma mensagem real de
cliente**. Perder sinal é pior que perder metadado de encerramento. Em
ingestão real vinda de WhatsApp o `conversation_id` viria do sistema de origem
e não haveria como forçar um novo.

### Quarentena como valor de `status`

Reaproveitaria a coluna existente em vez de criar estado novo.

**Descartada** por misturar dimensões ortogonais: uma conversa pode estar
encerrada **e** em quarentena ao mesmo tempo, e um único `VARCHAR` não
representa as duas coisas. Ver ADR-0004.

## Consequências

**Facilita**

- O frontend ganha um critério objetivo para separar ativos de histórico.
- Fechamento manual gera o dado necessário para, no futuro, calibrar
  fechamento automático.

**Dificulta**

- Surge uma **terceira rota pública**. O README descrevia duas
  (`ingest` e `customer/{id}/mood`) e precisa ser atualizado, assim como o
  contrato consumido pelo `chat-app`.
- `conversations` confirma-se como dimensão mutável: `last_message_at`,
  `status` e `closed_at` são sobrescritos. P2 não é violado, porque vale
  apenas para `mood_scores`, mas a diferença de natureza entre as duas tabelas
  precisa ficar clara para quem lê o esquema.

**Passa a ser proibido**

- Criar conversa a partir de mensagem `agent`.
- Sobrescrever `closed_at` num fechamento repetido.

**Limitação assumida**

A reabertura **destrói o histórico de encerramentos**: uma conversa fechada e
reaberta cinco vezes guarda apenas o último `closed_at`. Se a duração de
atendimento vier a ser métrica, isso não é reconstruível. A solução futura é
uma tabela `conversation_events` append-only, no mesmo espírito de
`mood_scores`, com `status` virando estado derivado por view.

**Interação com o ADR-0005**

Encerrar uma conversa **não** limpa o contexto de inferência. A janela é por
cliente e atravessa conversas, inclusive fechadas. Fechar é um evento de
atendimento, não de dados.

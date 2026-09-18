# ADR-0006: Sem encerramento de conversa no MVP

- **Status:** aceito
- **Data:** 2026-09-18
- **Princípios tocados:** P1
- **Substitui:** [ADR-0003](ADR-0003-ciclo-de-vida-da-conversa.md)
- **Referência:** `data-structure/data-model.md`, seções 2.1, 3.2 e 7; `README.md`

---

## Contexto

O ADR-0003 criou `POST /v1alpha1/conversation/{id}/close` e, com ela, uma
terceira rota pública. O README sempre declarou duas, e o descompasso obrigava
a escolher entre atualizar o README ou reduzir o escopo.

A escolha foi reduzir o escopo. O encerramento não é requisito do problema que
o trabalho se propõe a resolver: o objetivo é medir humor durante o
atendimento, e nada no cálculo depende de saber que um atendimento terminou. O
ADR-0003 já havia registrado que a rota existia para dar ao frontend um
critério de separação entre ativos e histórico — conveniência de interface, não
necessidade de dados.

Pesou também o que o ADR-0003 chamou de terceira rota pública: é a única
escrita de estado disparada por um **operador**, e não pela chegada de uma
mensagem. Sem autenticação definida entre serviços (item que o README mantém em
aberto), ela é a superfície onde a ausência de autorização causa dano visível:
qualquer chamada encerra o atendimento de qualquer cliente.

## Decisão

**O MVP tem exatamente duas rotas públicas:** `POST /v1alpha1/ingest` e
`GET /v1alpha1/customer/{id}/mood`.

**Encerramento sai de escopo.** Não existe rota de `close`, não existe
encerramento automático e, por consequência, não existe reabertura.

**Criação da conversa** permanece como no ADR-0003: a conversa nasce no
primeiro `ingest` com `role = 'customer'` para um `conversation_id`
inexistente. Mensagem `agent` para conversa desconhecida continua rejeitada com
`422` — a validação é barata, evita conversa órfã criada por atendente e não
custa rota nenhuma.

**`conversations.status` e `conversations.closed_at` permanecem no esquema como
colunas reservadas**, escritas apenas na criação, com `status = 'open'` e
`closed_at = null` para sempre.

## Alternativas consideradas

### Remover as colunas `status` e `closed_at`

Esquema mínimo de verdade: a conversa vira só um agrupador de mensagens.

**Descartada** porque manter as colunas custa zero — nenhum código as lê ou
escreve depois da criação — e removê-las agora significaria migração de esquema
quando a feature voltar. O valor fixo documenta a intenção sem criar caminho de
escrita.

### Manter a rota de encerramento

Preservaria a separação ativos × histórico no frontend.

**Descartada** por ser a única escrita de estado por operador no sistema,
puxando a decisão de autenticação para dentro do MVP, em troca de um benefício
que é de apresentação. A lista de conversas pode ser ordenada por
`last_message_at` sem nenhum campo novo.

### Encerramento automático por inatividade

Já havia sido descartado no ADR-0003 por exigir agendador. Continua descartado
pelo mesmo motivo, agora reforçado: sem a coluna sendo escrita por ninguém, não
há nem o consumidor do dado.

## Consequências

**Facilita**

- O README volta a estar correto: duas rotas, como sempre declarou.
- A API perde um caminho de escrita inteiro e o estado mutável que vinha com
  ele. `conversations` passa a ser efetivamente imutável depois da criação,
  exceto por `last_message_at`.
- Some a interação com o ciclo de vida que o ADR-0003 precisava explicitar: não
  existe conversa fechada, então nenhuma regra precisa dizer o que acontece
  quando uma mensagem chega em uma.
- A limitação assumida no ADR-0003 — reabertura destruindo o histórico de
  encerramentos — deixa de existir junto com a feature.

**Dificulta**

- O frontend perde o critério objetivo para separar atendimentos ativos de
  encerrados. A lista de conversas passa a ser ordenada por `last_message_at`,
  e cresce indefinidamente.
- O dado que calibraria um futuro encerramento automático deixa de ser
  produzido. Quando a feature voltar, ela começa sem histórico, como o ADR-0003
  começaria.

**Passa a ser proibido**

- Escrever `status` ou `closed_at` com qualquer valor fora de `'open'` e
  `null`. Enquanto não houver ADR que reviva o encerramento, uma escrita nessas
  colunas é violação desta decisão, não uma antecipação da feature.
- Criar conversa a partir de mensagem `agent` (regra herdada do ADR-0003).

**Em aberto, adiado deliberadamente**

- Autenticação e autorização entre serviços. Sem a rota de encerramento, a
  superfície de escrita se resume ao `ingest`, mas a questão continua de pé.
- Volta do encerramento. Se vier, o caminho é uma tabela
  `conversation_events` append-only, no espírito de `mood_scores`, com `status`
  derivado por view — e não o `UPDATE` que o ADR-0003 previa.

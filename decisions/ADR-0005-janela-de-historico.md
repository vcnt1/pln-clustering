# ADR-0005: Janela de histórico com 30 mensagens do cliente, sem `agent`

- **Status:** aceito
- **Data:** 2026-09-17
- **Princípios tocados:** P4, P5
- **Depende de:** ADR-0003 (encerramento não delimita a janela)
- **Referência:** `data-structure/data-model.md`, seções 2.2, 3.3, 6 (T5) e 9

---

## Contexto

O contrato `InferRequest.history` transporta uma janela de contexto, mas o
tamanho e o critério nunca foram fixados. É a transformação T5, e ela define
literalmente o que o modelo enxerga: qualquer mudança aqui muda a distribuição
de entrada e, por P4, exige nova `model_version`.

Três eixos estavam em aberto: quantas mensagens, de qual escopo (conversa ou
cliente) e quais papéis.

## Decisão

**Tamanho e escopo.** As **30 mensagens mais recentes com `role = 'customer'`
do cliente**, em ordem cronológica crescente, com a mensagem disparadora como
último item. O escopo é o **cliente**, não a conversa: a janela atravessa
`conversation_id` e inclui conversas encerradas.

**Papéis.** Mensagens `agent` **não entram** na janela. O filtro é aplicado
**na API**, ao montar o `InferRequest`.

**Contrato.** O campo `role` permanece em `HistoryMessage`, ainda que na prática
todos os valores sejam `customer`. Manter o campo permite reativar contexto de
atendente no futuro sem alterar o esquema.

**Manifesto.** `history_window = 30` passa a ser campo obrigatório e faz parte
do contrato de feature: divergência entre o valor do manifesto e o
comportamento da API é violação de P4 e deve impedir a subida do serviço.

**Índice.** A consulta deixa de ser por conversa e passa a ser por cliente, o
que torna o índice sugerido em 3.3 insuficiente. O índice necessário é
`(customer_id, role, sent_at)`.

## Alternativas consideradas

### Janela por conversa

Escopo mais estreito e intuitivo: cada atendimento começa do zero.

**Descartada** porque o humor é atributo do **cliente**, não do atendimento.
`MoodResponse` é por cliente, e um cliente que abriu três chamados irritados na
mesma semana chega ao quarto irritado. Reiniciar o contexto a cada conversa
apagaria exatamente o sinal que o projeto quer capturar.

### Janela por tempo (ex.: últimos 7 dias)

Acompanha naturalmente a recência.

**Descartada** por produzir tamanho de entrada imprevisível: um cliente pode
mandar duas mensagens ou duzentas no período. Tamanho variável complica a
extração de features e o orçamento de latência. Uma contagem fixa dá limite
superior determinístico ao custo de inferência.

### Incluir mensagens `agent` como contexto

Daria ao modelo a informação de que o problema já foi resolvido ou que houve
pedido de desculpas.

**Descartada** para o MVP por três motivos: dobra aproximadamente o payload,
move texto de atendente sem necessidade comprovada (P5), e o efeito da resposta
do atendente já aparece indiretamente na mensagem seguinte do cliente, que é o
que está sendo medido.

### Filtrar `agent` no `mood-ml`, não na API

É o que a seção 2.2 do `data-model.md` prescrevia: o contrato transporta os dois
papéis e o modelo decide o que usar.

**Descartada**, e esta é uma **divergência consciente em relação ao
data-model**, que deve ser atualizado. A troca: filtrar na API reduz PII em
trânsito, o que serve a P5, e diminui o payload; em compensação, acopla a API a
uma decisão de modelagem, de modo que experimentar com contexto de atendente
passa a exigir deploy da API e não só do `mood-ml`.

## Consequências

**Facilita**

- Custo de inferência com limite superior fixo e previsível, o que sustenta o
  orçamento de latência que o README ainda precisa definir.
- Menos texto trafegando entre processos, coerente com P5.

**Dificulta**

- O índice `(customer_id, role, sent_at)` precisa ser criado; sem ele a query
  da janela vira scan à medida que `messages` cresce.
- Experimentar com contexto de `agent` passa a exigir alteração na API.

**Passa a ser proibido**

- Alterar o tamanho ou o critério da janela sem publicar nova `model_version`.
  A janela é parte da especificação de features (P4), não um parâmetro
  operacional ajustável em runtime.

**Limitação assumida: contexto que não expira**

Com escopo por cliente e sem corte temporal, uma reclamação de meses atrás
continua entre as 30 últimas mensagens de um cliente pouco ativo. Um problema
resolvido segue puxando o humor para baixo indefinidamente.

Mitigações naturais, fora do escopo do MVP: corte temporal combinado à contagem
(as 30 últimas **dentro de N dias**) ou ponderação por recência na extração de
features. A segunda é preferível, por manter a decisão dentro do modelo em vez
de espalhá-la pela API.

**Limitação assumida: cliente sem histórico**

Cliente novo produz janela de um único item. O modelo precisa se comportar
razoavelmente com `len(history) == 1`, e isso deve ser caso de teste explícito
na spec do modelo.

**Em aberto, adiado deliberadamente**

- Ponderação por recência dentro da janela.
- Se conversas encerradas há muito tempo devem ser excluídas.

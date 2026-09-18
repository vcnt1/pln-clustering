# ADR-0007: Humor pertence à conversa; janela de 30 mensagens da conversa

- **Status:** aceito
- **Data:** 2026-09-18
- **Princípios tocados:** P4, P5
- **Substitui:** [ADR-0005](ADR-0005-janela-de-historico.md)
- **Depende de:** [ADR-0001](ADR-0001-escala-do-humor.md) (escala), [ADR-0004](ADR-0004-retentativa-e-quarentena.md) (quarentena por conversa), [ADR-0006](ADR-0006-sem-encerramento-no-mvp.md) (duas rotas)
- **Referência:** `data-structure/data-model.md`, seções 2.2, 2.3, 3.3, 3.4, 3.7, 4.4 e 6 (T5); `README.md`

---

## Contexto

O ADR-0005 fixou escopo de **cliente** para a janela de contexto: as 30 últimas
mensagens `customer` do cliente, atravessando conversas, inclusive encerradas.
O argumento era que o humor é atributo da pessoa, e que um cliente que abriu
três chamados irritados chega ao quarto irritado.

Duas coisas puxavam na direção contrária:

1. **O README e o frontend sempre falaram em humor por conversa.** O
   [README](../README.md) descreve o frontend exibindo "o humor do cliente como
   emoji ao lado do nome, por conversa", e o `chat-app` renderiza um emoji por
   item da lista de conversas.
2. **O grão ficava inconsistente dentro do próprio sistema.** A quarentena do
   ADR-0004 conta falhas **por conversa**, enquanto a janela olhava o cliente
   inteiro. Duas consultas no mesmo caminho quente, com dois escopos
   diferentes e dois índices distintos.

Havia ainda um efeito prático incômodo do escopo por cliente: com uma única
rota de humor por cliente, um cliente com duas conversas ativas recebia o mesmo
score nas duas, e a exibição "por conversa" era, na verdade, por cliente.

## Decisão

**O humor pertence à conversa.** `mood_scores` continua com `customer_id`
desnormalizado, mas o grão semântico da linha é a conversa.

**Janela (T5).** As **30 mensagens `customer` mais recentes da conversa** da
mensagem disparadora, em ordem cronológica crescente, com a disparadora como
último item. A janela **não atravessa** `conversation_id`.

**Filtro de `agent`.** Mantido como no ADR-0005: aplicado **na API**, ao montar
o `InferRequest`. O `mood-ml` nunca recebe texto de atendente. O campo `role`
permanece em `HistoryMessage` para permitir reativar esse contexto sem alterar
o esquema.

**Rota.** `GET /v1alpha1/customer/{id}/mood` é mantida (ADR-0006 fixou duas
rotas) e passa a significar: *o humor da conversa mais recentemente pontuada
deste cliente*. `conversation_id` deixa de ser opcional e passa a ser campo
**obrigatório** do `MoodResponse` — sem ele, o consumidor não sabe a que
conversa o score se refere.

**Manifesto.** Além de `history_window: 30`, passa a ser obrigatório
`history_scope: "conversation"`. Os dois são contrato de features: divergência
em relação ao comportamento da API viola P4 e deve impedir a subida do serviço.

**Índice.** `(conversation_id, role, sent_at)` substitui
`(customer_id, role, sent_at)`.

## Alternativas consideradas

### Manter o escopo por cliente (ADR-0005)

Captura o sinal entre atendimentos, que é o mais difícil de reconstruir depois.

**Descartada** por decisão de escopo do projeto, e pelos dois motivos do
contexto: divergência com o README e com o frontend, e grão inconsistente em
relação à quarentena. O sinal perdido está registrado abaixo como consequência
assumida, não como efeito colateral esquecido.

### Trocar a rota para `GET /conversation/{id}/mood`

Seria o caminho mais honesto para um humor que pertence à conversa: cada
conversa consultaria o seu.

**Descartada** para preservar o contrato declarado no README, que o ADR-0006
acabou de reafirmar. A troca de caminho é a evolução natural desta decisão, e
deve vir por ADR próprio se o frontend passar a precisar do humor de várias
conversas ao mesmo tempo.

### Janela por conversa, sem filtrar `agent` na API

Deixaria a API mais burra: envia as 30 últimas da conversa nos dois papéis e o
modelo decide.

**Descartada** por mover texto de atendente entre processos sem necessidade
comprovada, contra P5, e por dobrar aproximadamente o payload. O argumento do
ADR-0005 nesse ponto segue válido e não foi alterado.

## Consequências

**Facilita**

- Um único grão no caminho quente: a montagem da janela e a contagem de
  quarentena (ADR-0004) passam a filtrar por `conversation_id` e a usar a mesma
  família de índice.
- A consulta da janela fica mais barata: a conversa é um recorte muito menor
  que o histórico inteiro do cliente.
- O humor exibido passa a corresponder ao atendimento em curso, que é o que o
  atendente tem diante de si.
- Some a limitação de "contexto que não expira" do ADR-0005: uma reclamação
  antiga fica presa à conversa antiga.

**Dificulta**

- **Toda conversa nova começa do zero.** O comportamento com
  `len(history) == 1` deixa de ser caso de borda de cliente novo e passa a
  ocorrer no início de todo atendimento. Continua sendo caso de teste
  obrigatório na spec do modelo, agora com mais peso.
- Com uma rota por cliente e humor por conversa, um cliente com duas conversas
  ativas tem **dois humores**, e a rota expõe só um. O frontend deve exibir o
  emoji apenas na conversa indicada por `conversation_id` e um estado vazio nas
  demais — nunca repetir o mesmo score em todas.

**Passa a ser proibido**

- Alterar o tamanho **ou o escopo** da janela sem publicar nova
  `model_version`. Ambos são especificação de features (P4), não parâmetros
  operacionais.
- Compor a janela com mensagens de outra conversa, mesmo do mesmo cliente.

**Consequência assumida: perda do sinal entre atendimentos**

Um cliente que abriu três chamados irritados na mesma semana chega ao quarto
com histórico limpo. Era exatamente o argumento do ADR-0005, e está sendo
abandonado de propósito em troca de coerência com o produto e de um grão único
no sistema.

Mitigação futura, fora do escopo: uma feature derivada do histórico do cliente
(média dos scores das conversas anteriores, por exemplo) calculada pela API e
enviada como campo do `InferRequest` — o que traria o sinal de volta sem
reabrir a janela.

**Em aberto, adiado deliberadamente**

- Rota de humor por conversa, se o frontend precisar de várias ao mesmo tempo.
- Ponderação por recência dentro da janela.

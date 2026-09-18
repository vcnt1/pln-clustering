# ADR-0004: Política de retentativa e quarentena de inferência

- **Status:** aceito
- **Data:** 2026-09-17
- **Princípios tocados:** P2, P3
- **Depende de:** ADR-0003 (quarentena não usa `conversations.status`)
- **Referência:** `data-structure/data-model.md`, seções 3.6, 3.7 e 9

---

## Contexto

`inference_failures` existe para que falhas não sumam em silêncio, mas o modelo
de dados deixou em aberto o que fazer depois de registrar uma: reprocessar em
lote, descartar, ou alguma outra coisa. A decisão define três comportamentos
distintos:

1. o que o `GET /mood` devolve enquanto a inferência está falhando;
2. quando e como uma falha é reprocessada;
3. o que acontece quando as falhas não param.

P3 amarra a primeira: é proibido gravar score de fallback. Isso descarta de
saída qualquer política que "preencha" o buraco com um valor neutro.

## Decisão

### Comportamento durante a falha: conservador

O humor exibido **permanece o último valor bem-sucedido**. Cliente sem nenhuma
linha em `mood_scores` continua devolvendo `404`, inclusive quando todas as
suas inferências falharam.

Isto **não exige código novo**: como `mood_scores` é append-only (P2) e uma
falha não grava nada, `v_customer_mood_latest` já devolve o último score válido
por construção. A decisão é registrar que esse comportamento emergente é
intencional, e não um efeito colateral.

### Retentativa: absorvida pela próxima mensagem

Não há job, fila nem agendador de retentativa. Quando o cliente envia a próxima
mensagem, a nova inferência é disparada normalmente e a **mensagem que falhou
entra na janela de histórico** (ADR-0005), por já estar persistida em
`messages`.

Cada tentativa gera um `request_id` novo, como o modelo de dados já previa.

### Quarentena: após 3 falhas consecutivas

Contagem por **conversa**, de falhas **consecutivas desde a última inferência
bem-sucedida**. Um sucesso zera a contagem.

Ao atingir 3, a conversa entra em quarentena. A partir daí:

- mensagens continuam sendo **persistidas normalmente**, sem exceção;
- **nenhuma inferência é tentada** para essa conversa;
- cada mensagem em conversa quarentenada grava
  `error_code = 'quarantined'` em `inference_failures`, preservando a trilha;
- o `GET /mood` continua devolvendo o último humor válido.

O estado de quarentena é **derivado por consulta**, não armazenado:

```sql
-- falhas consecutivas da conversa desde a última inferência bem-sucedida
SELECT count(*)
FROM inference_failures f
JOIN messages m ON m.message_id = f.trigger_message_id
WHERE m.conversation_id = ?
  AND f.error_code <> 'quarantined'
  AND f.occurred_at > coalesce(
        (SELECT max(persisted_at) FROM mood_scores WHERE conversation_id = ?),
        '-infinity'::TIMESTAMPTZ
      );
```

Saída da quarentena é **manual**, fora do escopo do MVP.

## Alternativas consideradas

### Retentativa com backoff exponencial em job de fundo

Padrão de mercado para falhas transitórias.

**Descartada** por exigir agendador ou fila que o MVP não tem, e por resolver
um problema que a próxima mensagem do cliente já resolve de graça. A retentativa
absorvida elimina uma peça inteira de infraestrutura sem perda funcional: o
humor só importa quando há atendimento em curso, e atendimento em curso produz
mensagens.

### Quarentena gravada em `conversations.status`

Reaproveitaria coluna existente.

**Descartada** por ADR-0003: encerramento e quarentena são dimensões ortogonais
e não cabem num `VARCHAR` só. Além disso, estado derivado de eventos é o mesmo
padrão que P2 já impõe a `mood_scores`; guardar contador seria criar uma segunda
fonte da verdade que pode divergir de `inference_failures`.

### Devolver `503` no `GET /mood` durante falha

Sinalizaria degradação explicitamente ao frontend.

**Descartada** por quebrar o atendimento: o atendente perderia a informação que
tinha, que continua sendo a melhor disponível. Preferimos dado antigo e datado
a nenhum dado.

## Consequências

**Facilita**

- Zero infraestrutura nova de retentativa.
- Falhas transitórias do `mood-ml` ficam invisíveis ao atendente, que é o
  comportamento desejado numa conversa em andamento.

**Dificulta**

- `inference_failures` não tem `conversation_id`, então a contagem exige JOIN
  com `messages`. No volume do MVP é irrelevante, mas **denormalizar
  `conversation_id` para `inference_failures`** torna a checagem um scan de
  tabela única. Recomendado, e compatível com o esquema atual.
- A checagem de quarentena roda a cada ingest de `customer`, somando uma query
  ao caminho quente.

**Limitação assumida: humor silenciosamente obsoleto**

A política conservadora faz o frontend exibir um humor antigo **sem saber que
é antigo**. É o preço de não violar P3, mas não pode passar em branco: o
`MoodResponse` já carrega `computed_at`, e o `chat-app` deve derivar dali um
indicador visual de obsolescência acima de um limiar. Isso não altera o
contrato, apenas usa um campo existente.

**Limitação assumida: mensagens sem score próprio**

Como a retentativa é absorvida pela mensagem seguinte, a mensagem que falhou
nunca recebe linha própria em `mood_scores`: o `trigger_message_id` do sucesso
posterior é o da mensagem **nova**. Consequência: `mood_scores` **não é um
registro completo por mensagem**, e qualquer análise de cobertura precisa
cruzar `messages` com `mood_scores` e `inference_failures` para saber o que foi
avaliado.

**Em aberto, adiado deliberadamente**

- Mecanismo de saída da quarentena (endpoint administrativo ou reprocessamento
  em lote).
- Limiar de obsolescência usado pelo frontend.
- Se o limite de 3 deve ser configurável por ambiente.

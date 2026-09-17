# ADR-0002: Ground truth do MVP vem de `generated_label` sintético

- **Status:** aceito
- **Data:** 2026-09-17
- **Princípios tocados:** P3, P4, P5
- **Depende de:** ADR-0001 (a escala do rótulo é a escala do score)
- **Referência:** `data-structure/data-model.md`, seções 4.1, 4.3 e 6 (T4)

---

## Contexto

Sem rótulo não há treino. Sem treino não existe `model_version` legítima. E
como P3 proíbe gravar score de fallback, a ausência de rótulo trava também a
integração online entre `mood-api` e `mood-ml`: não haveria valor válido para
persistir em `mood_scores`.

É a decisão mais bloqueante do projeto. O README já a listava como item
obrigatório antes de iniciar o treino, sob "Metodologia e Avaliação".

A restrição prática é o prazo: trata-se de um TCC individual, e a escolha
precisa produzir um dataset rotulado utilizável em dias, não em meses.

## Decisão

Adotar o campo **`generated_label`** do corpus sintético como ground truth do
MVP.

- `label_source = 'synthetic'`
- `target_type = 'message'`, com grão de mensagem
- `scale = '-1 to 1'`, por ADR-0001
- `annotator = null`
- O gerador define persona e trajetória emocional da conversa **antes** de
  escrever o texto; o rótulo nasce junto com o dado.
- IDs sintéticos mantêm o prefixo `syn-`, para que corpus sintético e
  snapshots reais possam coexistir sem colisão.

## Alternativas consideradas

### Anotação manual

É a de maior qualidade e a única que mede humor humano de verdade.

**Descartada para o MVP** pelo custo: exige protocolo de anotação, mais de um
anotador para medir concordância (Krippendorff ou ICC, já que ADR-0001 tornou
o alvo contínuo) e algumas centenas de exemplos no mínimo. Consumiria o prazo
inteiro do trabalho antes de existir qualquer pipeline funcionando.

### Heurística por léxico de polaridade

Barata e imediata.

**Descartada** por circularidade. Se o rótulo vem de um léxico e o modelo
aprende do rótulo, o melhor resultado possível do modelo é empatar com o
léxico. Não haveria justificativa metodológica para treinar um modelo em vez
de simplesmente aplicar a heurística em produção.

### Proxy por CSAT pós-atendimento

Conceitualmente o mais forte, porque amarra o humor a um desfecho de negócio
real em vez de a um julgamento subjetivo.

**Descartada** por inviabilidade atual: o campo de avaliação de atendimento não
existe no modelo de dados, e o próprio README já o classifica como extensão
futura. Também traria grão de conversa, não de mensagem, e uma taxa de resposta
baixa que enviesaria a amostra para os extremos.

## Consequências

**Facilita**

- Rótulo de graça, sem ambiguidade e em volume arbitrário.
- Destrava a trilha offline inteira (T3, T4, treino, manifesto) e, por
  consequência, a integração online, sem violar P3.
- Corpus sintético não contém PII, então os artefatos de ML do MVP nascem
  naturalmente em conformidade com P5.

**Dificulta**

- O split por `customer_id` exigido pelo modelo de dados fica ainda mais
  crítico. Personas sintéticas repetidas entre treino e teste vazariam o padrão
  do gerador diretamente, e a métrica mediria memorização.
- A qualidade do modelo passa a ser limitada pela qualidade do gerador. Investir
  no realismo das personas rende mais que investir no algoritmo.

**Limitação metodológica, a declarar no TCC**

O modelo aprende a função que o gerador usou para escrever o texto, não o humor
humano. As métricas serão altas e isso **não é mérito do modelo**: é
consequência de treino e avaliação compartilharem o mesmo processo gerador.
Nenhum número produzido sobre o corpus sintético é evidência de desempenho em
conversas reais.

Esta limitação deve aparecer explicitamente na seção de resultados, junto com
o valor das métricas, e não apenas em trabalhos futuros.

**Validação externa, fora do escopo do MVP**

O teste honesto seria anotar manualmente um conjunto pequeno de conversas reais
(ordem de cinquenta) e medir a correlação entre o score do modelo e o rótulo
humano. Enquanto não existir, o modelo não deve ser apresentado como validado.

**Em aberto, adiado deliberadamente**

- Estratégia de geração das personas e das trajetórias emocionais, a ser fixada
  na spec do corpus sintético (fase F3).
- Migração futura do `label_source` para `manual` ou `csat`. O esquema de
  `labels/` já suporta múltiplas fontes sobre o mesmo corpus, então a troca não
  exige alteração estrutural: gera-se um novo `label_set_id` e um novo
  `dataset_id`.

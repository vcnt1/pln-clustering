# ADR-0009: Abordagem C passa a ter fine-tuning do encoder

- **Status:** aceito
- **Data:** 2026-09-25
- **Princípios tocados:** P3, P4
- **Substitui parcialmente:** [ADR-0008](ADR-0008-abordagens-de-modelo.md), nas seções "Abordagem C — implementada depois", "Alternativas → Abordagem C com fine-tuning" e o item "versão ativa do modelo" de "Em aberto, adiado deliberadamente". O restante da ADR-0008 continua valendo.
- **Depende de:** [ADR-0001](ADR-0001-escala-do-humor.md), [ADR-0002](ADR-0002-origem-do-ground-truth.md), [ADR-0007](ADR-0007-humor-por-conversa.md)
- **Detalhamento:** [`mood-ml/specs/11-embeddings-finetuning.md`](../mood-ml/specs/11-embeddings-finetuning.md)

---

## Contexto

A ADR-0008 fixou a abordagem C como encoder **congelado** e descartou o fine-tuning para
o MVP por dois motivos: exige GPU e exige dados rotulados reais em volume.

Duas coisas mudaram o cálculo:

1. Com o encoder congelado, C compete com A usando uma representação que nunca viu o
   domínio (hotelaria, PT-BR). O resultado provável é "C não supera A", que a ADR-0008
   já previa como válido, mas que compara pouco: mede o encoder genérico, e não a família
   de representação.
2. O encoder multilíngue pequeno (~118M parâmetros, ~96M deles na matriz de palavras)
   permite fine-tuning em CPU dentro de um orçamento aceitável, com a matriz de palavras
   congelada. O treino é offline; a inferência continua em CPU.

O segundo motivo da ADR-0008 (dados reais em volume) **não muda**: o corpus segue
sintético (ADR-0002), com 1.755 exemplos de treino em 126 clientes no dataset atual.

O objetivo desta ADR deixa de ser medir qual representação de texto é melhor para
fins de comparação entre A e C. O propósito central passa a ser preparar o modelo
para operar sobre uma base real de atendimentos: a comparação entre A e C continua
acontecendo, mas como efeito colateral registrável, não como fim. O critério de
sucesso de C deixa de ser "supera A no corpus sintético" e passa a ser "serve de
base viável para produção quando dados reais estiverem disponíveis".

## Decisão

- A abordagem C passa a **ajustar o encoder** (fine-tuning) junto com a cabeça linear de
  regressão. `algorithm = "embeddings-ft"`. O valor `"embeddings-ridge"` (encoder
  congelado) deixa de ser implementado; o desempenho com o encoder congelado é medido
  como diagnóstico dentro do próprio treino (etapa de sondagem linear).
- **Mantido da ADR-0008:** dois blocos (disparadora + média do contexto, vetor nulo sem
  contexto), saída recortada em `[-1, 1]`, `feature_spec_version = "fs-1"`, mesmo
  `dataset_id` de A, split por `customer_id`, `history_window = 30`, nada de vetores em
  `datasets/*`, dependências pesadas em requirements separado.
- **Muda:** a cabeça deixa de ser o `Ridge` do scikit-learn e passa a ser uma camada linear
  treinada junto com o encoder, inicializada por um Ridge sobre embeddings congelados.
  Continua sendo um modelo linear sobre os dois blocos, com regularização L2.
- O `validation` passa a decidir o **early stopping** do fine-tuning. O `test` continua
  intocado até a avaliação.
- O encoder específico e os hiperparâmetros ficam na spec 11 e em `configs/pipeline.yaml`.
- **Hardware de treino:** GPU disponível (GTX 1660); há também orçamento em CPU — o treino cabe no tempo necessário em uma máquina com AMD Ryzen 5 5600X (6 núcleos). `device` é configurável (`auto | cpu | cuda`, spec 11), sem decisão fixa de qual usar.
- **Orçamento de latência:** para o MVP, o critério é brando. A prioridade é implementar embeddings e fine-tuning; otimizar a latência p95 fica para quando C for avaliado para promoção (ver "Versão ativa", abaixo).
- **Versão ativa:** por ora, a validação externa em dados reais (ADR-0002, "Validação
  externa, fora do escopo do MVP") não é viável. A promoção de qualquer versão (A ou C)
  a ativa continua regida só pelo quality gate já existente (candidato vs. baseline, na
  spec de treino e avaliação). Fica registrada a expectativa de que um teste em dados
  reais seja incorporado a esse critério no futuro, quando viável — não é um requisito
  em vigor hoje.

## Alternativas consideradas

### Manter C congelado (ADR-0008 como está)

**Rejeitada.** É a opção mais barata e segue válida. Perde a comparação entre "representação
genérica" e "representação adaptada ao domínio", que é o que o TCC pode discutir.

### Fine-tuning completo, inclusive a matriz de palavras

**Rejeitada.** ~96M parâmetros a mais no otimizador, mais memória e mais risco de
sobreajuste com 1,7 mil exemplos, sem ganho esperado.

### BERTimbau (BERT base PT-BR) como encoder

**Adiada.** Modelo maior, sem versão de sentenças pronta. Fica como troca de configuração
possível, depois que o pipeline com o encoder pequeno estiver validado.

### Manter a moldura de comparação entre A e C como objetivo central

**Rejeitada.** Medir qual representação de texto vence no corpus sintético não diz, por
si só, se o modelo está pronto para uma base real de atendimentos. A comparação continua
registrada como resultado, mas o critério que orienta a implementação passa a ser a
viabilidade de produção.

## Consequências

**Facilita**

- Comparação A × C-congelado (diagnóstico) × C-ajustado sobre o mesmo `dataset_id`.
- O fine-tuning é offline; a inferência não precisa de GPU.

**Dificulta**

- **Sobreajuste ao gerador sintético:** o encoder ajustado pode aprender artefatos do
  gerador, e não humor. As métricas de A já saturam (MAE 0,041 no teste). A diferença entre
  A e C pode ser pouco informativa. Deve ser declarado no TCC.
- **Reprodutibilidade:** só a CPU garante pesos idênticos entre execuções. Em GPU, o
  fingerprint identifica as entradas, não os bits do resultado.
- **Artefato maior:** o modelo passa a ser um diretório (`model.joblib` + `encoder/`, ~470 MB
  em fp32), o que exige ajuste no registro e na inferência (specs 07 e 08).
- **Latência:** uma requisição com histórico cheio codifica até 30 mensagens. O critério é
  brando no MVP (ver "Orçamento de latência" em Decisão), mas o custo por requisição é
  maior que em A — relevante quando a latência for revisitada para promoção.
- **Rede no treino:** o download do encoder base acontece uma vez, no treino, com
  `revision` fixada. A inferência nunca acessa a rede.

**Passa a ser proibido**

- Baixar pesos da rede na inferência.
- Trocar encoder, `revision` ou tokenização sem nova `model_version`.

**Em aberto, adiado deliberadamente**

- Critério de gatilho para migrar `label_source` de sintético para `manual`/`csat`
  (depende de volume de dados reais ainda não disponível).
- Protocolo de anotação manual para a validação externa prevista na ADR-0002 (depende
  de recursos/parceria ainda não definidos).
- Base legal, retenção e expurgo de dados reais (decisão fora do escopo técnico do
  mood-ml).
- Ponderação por recência no contexto (já em aberto na ADR-0007).

## Documentos alinhados

- `mood-ml/specs/00-decisoes.md`: registra esta ADR.
- `data-structure/data-model.md` §4.4: campo `encoder` opcional no manifesto.

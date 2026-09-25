# ADR-0009: Abordagem C passa a ter fine-tuning do encoder

- **Status:** proposto
- **Data:** 2026-09-25
- **Princípios tocados:** P3, P4
- **Substitui parcialmente:** [ADR-0008](ADR-0008-abordagens-de-modelo.md), nas seções "Abordagem C — implementada depois" e "Alternativas → Abordagem C com fine-tuning". O restante da ADR-0008 continua valendo.
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
- **Latência:** uma requisição com histórico cheio codifica até 30 mensagens. O orçamento
  de latência continua em aberto (ADR-0008) e passa a ser mais crítico.
- **Rede no treino:** o download do encoder base acontece uma vez, no treino, com
  `revision` fixada. A inferência nunca acessa a rede.

**Passa a ser proibido**

- Baixar pesos da rede na inferência.
- Trocar encoder, `revision` ou tokenização sem nova `model_version`.

**Em aberto, adiado deliberadamente**

- Versão ativa e orçamento de latência (ADR posterior, baseada na spec de avaliação).
- Ponderação por recência no contexto (já em aberto na ADR-0007).

## Documentos a alinhar quando aceita

- `mood-ml/specs/00-decisoes.md`: registrar esta ADR na tabela.
- `mood-ml/CLAUDE.md`: §4 (linha "C, depois") e §8 (fase 6).
- `data-structure/data-model.md` §4.4: campo `encoder` opcional no manifesto.

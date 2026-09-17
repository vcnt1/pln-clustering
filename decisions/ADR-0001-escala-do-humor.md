# ADR-0001: Escala do humor contínua `-1 to 1`

- **Status:** aceito
- **Data:** 2026-09-17
- **Princípios tocados:** P2, P3
- **Referência:** `data-structure/data-model.md`, seções 3.4, 3.5 e 6 (T6)

---

## Contexto

O modelo de dados deixou a escala do humor deliberadamente aberta: `score` é
sempre `DOUBLE` e `scale` é um `VARCHAR` que declara em qual escala o valor
está. A estrutura suporta qualquer escolha, mas quatro pontos do sistema
precisam concordar sobre um valor concreto antes que a primeira linha seja
gravada:

1. a validação de limites na API, que decide entre gravar em `mood_scores` ou
   registrar `score_out_of_range` em `inference_failures`;
2. o campo `scale` do manifesto do modelo, que é invariante por
   `model_version`;
3. a normalização em `chat-app/src/utils/mood.ts`, que mapeia score para
   emoji e para a barra de humor;
4. a métrica de avaliação do modelo, que depende de o problema ser regressão
   ou classificação.

Adiar a decisão significa retrabalho nas quatro pontas simultaneamente. É,
junto com o ADR-0002, uma das duas decisões que travam todas as demais.

## Decisão

Adotar escala **contínua `-1 to 1`**, com `mood_label` permanecendo `null` no
MVP.

- `scale = '-1 to 1'` em `mood_scores`, no `InferResponse` e no manifesto.
- Limites inclusivos: score válido em `[-1.0, 1.0]`.
- Semântica: sinal indica direção (negativo = insatisfeito, positivo =
  satisfeito), módulo indica intensidade, zero é neutro.
- A transformação T6 (score → categoria) fica fora do escopo do MVP. O
  frontend continua fazendo o mapeamento para emoji sozinho.

## Alternativas consideradas

### Categorias discretas (feliz, neutro, irritado)

O mapeamento para emoji fica trivial e a rotulagem manual fica mais barata,
porque anotar categoria exige menos calibração entre anotadores que anotar um
número.

**Descartada** por perder granularidade e por criar transições abruptas: dois
scores adjacentes na fronteira de uma categoria exibiriam emojis diferentes sem
que nada de real tenha mudado na conversa. Também impediria medir tendência
dentro de uma mesma categoria, que é o sinal mais útil para o atendente.

### Contínua `0 to 1`

Estruturalmente equivalente à escolhida, e já reconhecida pelo `mood.ts`.

**Descartada** por não ter ponto neutro natural. Zero significa "péssimo", não
"neutro", o que torna a leitura de uma série temporal contraintuitiva e obriga
o frontend a saber que o centro é `0.5`.

### Contínua sem limites (z-score, temperatura aberta)

Permitiria expressar intensidades extremas sem saturar.

**Descartada** por impossibilitar a validação de limites prevista em 3.5. Sem
limites não há `score_out_of_range`, e uma resposta corrompida do `mood-ml`
entraria no banco como um humor válido, violando P3 na prática.

## Consequências

**Facilita**

- A validação na API vira uma comparação de intervalo, sem tabela de domínio.
- `mood.ts` já reconhece `'-1 to 1'`; a divergência 8 da seção 10 se reduz a
  normalizar a barra pela mesma função que produz o emoji.
- Zero como neutro torna agregações (média por conversa, tendência por
  cliente) diretamente interpretáveis.

**Dificulta**

- O problema de ML passa a ser **regressão**, não classificação. A métrica de
  avaliação é MAE ou RMSE, e acurácia deixa de ser aplicável. Isso precisa
  estar refletido em `manifest.metrics`.
- A rotulagem manual, se vier a existir, fica mais cara: anotar um valor
  contínuo exige protocolo de calibração entre anotadores.

**Passa a ser proibido**

- Gravar `score` fora de `[-1.0, 1.0]`. Um valor fora do intervalo vira
  `inference_failures` com `error_code = 'score_out_of_range'`.
- Misturar escalas dentro de uma mesma `model_version`. Por P2, mudar a escala
  não é um `UPDATE` em `mood_scores`: exige treinar e publicar uma
  `model_version` nova, e as linhas antigas permanecem na escala antiga com o
  `scale` que declararam.

**Em aberto, adiado deliberadamente**

- Definição de `mood_labels` no manifesto e implementação de T6. Só se torna
  necessário se o frontend precisar de categorias vindas do backend.

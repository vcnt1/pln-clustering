# Análise de Humor do Cliente em Atendimento

Trabalho de Conclusão de Curso (Engenharia de Dados). Ferramenta que mede em
tempo real o estado de humor de clientes durante atendimentos de suporte de
acomodações de hotelaria, com base no histórico de conversa de cada cliente.

Status: projeto em fase de desenho; a implementação e a configuração
reprodutível ainda não estão disponíveis.

## Contexto e Motivação

**Problema**: no atendimento humano de clientes de acomodações de hotelaria, é
difícil garantir um atendimento padronizado e com qualidade consistente, pois
os atendentes não têm uma forma objetiva de saber como abordar cada perfil de
cliente.

**Por que importa**: o atendimento é parte crucial da venda e do pós-venda.
Padronizar e categorizar continuamente esses atendimentos pode gerar ganhos de
eficiência, qualidade de serviço, satisfação do cliente e marketing orgânico.

**Trabalho relacionado**: ferramentas existentes ingerem conversas via API do
WhatsApp, removem texto padrão/mensagens automáticas, mascaram dados sensíveis
e delegam a um LLM (ex.: Gemini) a categorização do problema de estadia e sua
urgência.

**Diferencial deste projeto**: em vez de delegar a categorização a um LLM
externo, o sistema treina e usa um modelo próprio de machine learning para
calcular, em tempo real, uma "temperatura de humor" a partir do histórico de
conversa do próprio cliente. Esse dado fica disponível ao atendente durante o
atendimento.

**Extensão futura (se houver tempo)**: sugerir ao atendente como responder,
com base no histórico de atendimentos bem e mal avaliados na plataforma.

## Arquitetura

```text
              ┌──────────────┐   GET mood     ┌──────────────┐
 cliente ---> │  Backend API │ -------------> │   Frontend    │
   msg        │  (DuckDB)    │                │  (chat + mood)│
              └──────┬───────┘                └──────────────┘
                     │ mensagem            ^
                     v                     │ mood
              ┌──────────────┐             │
              │ Módulo de ML │ ------------┘
              │ ingest/      │
              │ transform/   │
              │ train/infer  │
              └──────────────┘
```

O projeto terá três componentes:

1. **Backend API**: único processo com acesso de escrita ao DuckDB. Recebe
   mensagens, persiste e expõe o humor atual do cliente.
2. **Frontend**: interface de chat simples, exibe o humor do cliente como
   emoji ao lado do nome, por conversa.
3. **Módulo de Machine Learning**: processo separado, sem acesso direto de
   escrita ao DuckDB da API. Consome mensagens e devolve o humor calculado
   ao Backend API, que persiste o resultado.

**Persistência (decisão fechada)**: o DuckDB é acessado por um único
processo escritor (Backend API), pois o DuckDB usa lock exclusivo de
arquivo por processo e não é adequado para escrita concorrente
multi-processo nem para ingestão transacional linha a linha em alto volume.
O módulo de ML roda como processo separado e troca dados com a API por
chamada HTTP interna (síncrona, MVP) ou fila (assíncrona, se o volume
exigir); não acessa o arquivo DuckDB diretamente.

## Backend API

São duas rotas públicas. O encerramento de conversa está fora do escopo do MVP
(ver `decisions/ADR-0006`).

- `POST /v1alpha1/ingest`
  - Corpo: `{"conversation_id": "string", "customer_id": "string", "role": "customer|agent", "message": "string", "timestamp": "ISO 8601"}`
  - Resposta: `200 OK` (mensagem persistida) ou `202 Accepted` (se o cálculo de
    humor for assíncrono).
  - Persiste a mensagem no DuckDB; mensagens com `role=agent` são
    armazenadas mas não entram no cálculo de humor.
  - Só uma mensagem `role=customer` cria conversa e dispara inferência. O humor
    é calculado **por conversa**, sobre as últimas 30 mensagens `customer`
    daquela conversa.
- `GET /v1alpha1/customer/<id>/mood`
  - Resposta: `{"customer_id": "string", "conversation_id": "string", "score": number, "scale": "string", "model_version": "string", "computed_at": "ISO 8601"}`.
  - Devolve o humor da conversa mais recentemente pontuada do cliente;
    `conversation_id` diz qual é.
  - `404` se ainda não houver humor calculado para o cliente — o que inclui
    cliente com mensagens cujas inferências falharam.

Erros a cobrir: `400` (payload inválido), `404` (sem humor calculado),
`413` (mensagem excede tamanho máximo), `422` (mensagem `agent` para conversa
inexistente), `429` (rate limit). Autenticação/autorização entre serviços ainda
não definida.

Persistência em DuckDB, com escritas sequenciais dentro do processo da API.

## Frontend

Aplicação de chat simples. Para cada conversa exibida, mostra um emoji ao lado
do nome do cliente representando o humor daquela conversa, obtido via
`GET /v1alpha1/customer/<id>/mood`. Como o humor pertence à conversa e a rota
devolve o de uma só, o emoji aparece apenas na conversa indicada por
`conversation_id`; nas demais, um estado vazio — que nunca é um emoji neutro.

## Módulo de Machine Learning

1. **Ingest**: inicialmente, dados sintéticos de conversas de clientes para
   bootstrap do treino; em produção, o fluxo de mensagens recebido via
   `POST /v1alpha1/ingest`.
2. **Transform**: limpeza e extração de features a partir da mensagem do
   cliente e das mensagens anteriores da mesma conversa.
3. **Train**: treino do modelo sobre o corpus sintético (e, depois, dados
   reais) para prever a temperatura de humor.
4. **Infer**: a cada nova mensagem do cliente, calcula em tempo real a
   temperatura de humor atualizada, usada pelo Backend API para responder ao
   endpoint de humor.

## Decisões Fechadas

Cada uma tem um ADR em `decisions/`, e o esquema resultante está em
`data-structure/data-model.md`. Os princípios invioláveis do projeto estão em
`decisions/constitution.md`.

- **Definição de "humor"** (ADR-0001): escala contínua `-1 to 1`, sem
  categorias discretas no MVP. O mapeamento para emoji fica no frontend.
- **Origem do rótulo/ground truth** (ADR-0002): rótulo sintético gerado junto
  com o corpus, com grão de mensagem.
- **Falhas de inferência** (ADR-0004): nunca viram score padrão; a rota de
  humor devolve o último valor válido, e a conversa entra em quarentena após
  três falhas consecutivas.
- **Rotas públicas e ciclo de vida da conversa** (ADR-0006): duas rotas, sem
  encerramento de conversa.
- **Grão do humor e janela de contexto** (ADR-0007): humor por conversa,
  calculado sobre as últimas 30 mensagens `customer` daquela conversa.
- **Arquitetura do modelo** (ADR-0008): regressão com Ridge sobre dois blocos
  (mensagem disparadora e contexto da conversa). A abordagem A (TF-IDF) é
  implementada primeiro, e a C (embeddings pré-treinados congelados) depois,
  para comparação sobre o mesmo dataset.

## Decisões em Aberto

- **Geração de dados sintéticos**: estratégia para simular perfis de clientes
  e conversas plausíveis para o treino inicial.
- **Atualização do modelo**: re-treino periódico vs. atualização incremental
  por cliente conforme novas mensagens chegam.
- **Atualização do humor no frontend**: polling, SSE ou WebSocket — decide
  o quão "tempo real" a exibição é.
- **Sugestão de respostas (extensão futura)**: fora de escopo do MVP; exige
  um campo de avaliação de atendimento que ainda não existe no modelo de
  dados.

## Metodologia e Avaliação (a definir)

Item obrigatório antes de iniciar o treino do modelo, ainda em aberto:

- **Orçamento de latência**: meta de tempo de resposta por mensagem (ex.:
  p95 abaixo de um limite definido), que orienta a escolha da versão ativa
  entre as abordagens A (TF-IDF) e C (embeddings) da ADR-0008. A janela de 30
  mensagens dá um teto fixo ao custo de cada inferência.

## Ética e Privacidade

Conversas de clientes são dados sensíveis mesmo quando o bootstrap usa
dados sintéticos. Definir antes de usar dados reais: base legal (LGPD),
mascaramento/anonimização de PII, política de retenção, e avaliação de
viés do modelo entre diferentes perfis linguísticos de clientes.

## Ambiente

Python 3.12 é a versão exigida para `mood-api` e `mood-ml`, cada um com seu
próprio `.venv`. Use os scripts na raiz do repositório para reproduzir o
ambiente:

```bash
./setup-mood-api.sh   # cria mood-api/.venv (Python 3.12) e instala requirements.txt
./setup-mood-ml.sh    # cria mood-ml/.venv (Python 3.12) e instala requirements.txt
./setup-chat-app.sh   # roda npm install em chat-app
./setup.sh            # roda os três acima em sequência
```

Os scripts falham com uma mensagem clara se `python3.12` não estiver
disponível, ou se um `.venv` existente tiver sido criado com outra versão.
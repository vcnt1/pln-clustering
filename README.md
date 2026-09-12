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

- `POST /v1alpha1/ingest`
  - Corpo: `{"conversation_id": "string", "customer_id": "string", "role": "customer|agent", "message": "string", "timestamp": "ISO 8601"}`
  - Resposta: `200 OK` (mensagem persistida) ou `202 Accepted` (se o cálculo de
    humor for assíncrono).
  - Persiste a mensagem no DuckDB; mensagens com `role=agent` são
    armazenadas mas não entram no cálculo de humor do cliente.
- `GET /v1alpha1/customer/<id>/mood`
  - Resposta: `{"customer_id": "string", "score": number, "scale": "string", "model_version": "string", "computed_at": "ISO 8601"}`.
  - `404` se o cliente não tiver mensagens ainda.

Erros a cobrir: `400` (payload inválido), `404` (cliente/conversa
inexistente), `413` (mensagem excede tamanho máximo), `429` (rate limit).
Autenticação/autorização entre serviços ainda não definida.

Persistência em DuckDB, com escritas sequenciais dentro do processo da API.

## Frontend

Aplicação de chat simples. Para cada conversa exibida, mostra um emoji ao lado
do nome do cliente representando seu humor atual, obtido via
`GET /v1alpha1/customer/<id>/mood`.

## Módulo de Machine Learning

1. **Ingest**: inicialmente, dados sintéticos de conversas de clientes para
   bootstrap do treino; em produção, o fluxo de mensagens recebido via
   `POST /v1alpha1/ingest`.
2. **Transform**: limpeza e extração de features a partir do texto das
   mensagens e do histórico do cliente.
3. **Train**: treino do modelo sobre o corpus sintético (e, depois, dados
   reais) para prever a temperatura de humor.
4. **Infer**: a cada nova mensagem do cliente, calcula em tempo real a
   temperatura de humor atualizada, usada pelo Backend API para responder ao
   endpoint de humor.

## Decisões em Aberto

- **Definição de "humor"**: escala contínua (temperatura) vs. categorias
  discretas; isso define o mapeamento para emojis no frontend.
- **Geração de dados sintéticos**: estratégia para simular perfis de clientes
  e conversas plausíveis para o treino inicial.
- **Arquitetura do modelo**: features clássicas + classificador/regressor vs.
  embeddings de texto + modelo; trade-off de custo/latência para inferência
  em tempo real.
- **Atualização do modelo**: re-treino periódico vs. atualização incremental
  por cliente conforme novas mensagens chegam.
- **Esquema de dados no DuckDB**: tabelas para clientes, mensagens, histórico
  de humor por mensagem/conversa, e versão do modelo usado em cada inferência.
- **Stack do frontend**: framework a definir; consumirá as duas rotas do
  Backend API.
- **Atualização do humor no frontend**: polling, SSE ou WebSocket — decide
  o quão "tempo real" a exibição é.
- **Sugestão de respostas (extensão futura)**: fora de escopo do MVP; exige
  um campo de avaliação de atendimento que ainda não existe no modelo de
  dados.

## Metodologia e Avaliação (a definir)

Itens obrigatórios antes de iniciar o treino do modelo, ainda em aberto:

- **Origem do rótulo/ground truth**: como cada mensagem ou conversa recebe
  um valor de humor de referência (anotação manual, heurística, proxy como
  CSAT pós-atendimento).
- **Orçamento de latência**: meta de tempo de resposta por mensagem (ex.:
  p95 abaixo de um limite definido), que orienta a escolha entre features
  clássicas e embeddings.

## Ética e Privacidade

Conversas de clientes são dados sensíveis mesmo quando o bootstrap usa
dados sintéticos. Definir antes de usar dados reais: base legal (LGPD),
mascaramento/anonimização de PII, política de retenção, e avaliação de
viés do modelo entre diferentes perfis linguísticos de clientes.

## Ambiente

Todas as dependências Python ficarão em `.venv`. Este é um ambiente
provisório de desenvolvimento; pin de versões e manifesto de dependências
antes da implementação.

```bash
python -m venv .venv
source .venv/bin/activate
pip install duckdb fastapi uvicorn pydantic scikit-learn
```
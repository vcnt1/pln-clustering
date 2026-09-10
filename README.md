# Agrupamento de PLN

Pipeline em Python para agrupar decisões judiciais brasileiras por similaridade
semântica.

Status: projeto em fase de desenho; a implementação e a configuração
reprodutível ainda não estão disponíveis.

## Fluxo

```text
coleta -> transformação -> tokenização/embeddings -> agrupamento -> avaliação
```

Cada etapa será um processo Python independente. O DuckDB registrará a
identidade do documento, hashes de conteúdo, configuração dos artefatos
derivados, versões, status e erros. Um `Makefile` executará as etapas; o código
de cada etapa usará esse registro para fornecer idempotência, novas tentativas
e invalidação após alterações na entrada ou na configuração. Prefira escritas
sequenciais no arquivo DuckDB.

Artefatos de documentos inalterados podem ser reutilizados. O agrupamento e a
avaliação são executados sobre um snapshot versionado do corpus; adicionar
documentos a um grupo existente exige uma política de atribuição específica
para cada algoritmo.

## Etapas

1. **Coleta**: um web scraper descobre PDFs de decisões judiciais, faz o
	download para o armazenamento local e registra no DuckDB a URL de origem,
	hora da coleta, identificador estável da decisão (quando disponível),
	caminho do arquivo e hash do PDF. Reutilize arquivos com hash inalterado;
	preserve a proveniência da origem e as falhas de download. A extração de
	texto do PDF é o primeiro artefato de transformação.
    https://esaj.tjce.jus.br/cjsg/resultadoCompleta.do
2. **Transformação**: normalize a codificação e os espaços em branco, remova
	 trechos padronizados e duplicatas, e preserve os textos bruto e limpo.
	 - Preserve por padrão maiúsculas/minúsculas, termos jurídicos, negações,
		 datas, números, citações e URLs; eles podem ter significado jurídico.
	 - Trate stop words em português e lematização com spaCy como experimentos,
		 após validar o efeito em uma amostra representativa.
3. **Tokenização e embeddings**: tokenize as decisões limpas e gere embeddings
	 com um checkpoint local e fixado do BERTimbau. Comece com mean pooling que
	 considere a máscara de atenção e vetores normalizados; divida decisões que
	 excedam o limite de 512 tokens do BERTimbau e agregue seus embeddings.
	 Persista a revisão do modelo, versão do pré-processamento, dimensão e
	 local de armazenamento dos vetores.
4. **Agrupamento**: execute múltiplos algoritmos sobre os vetores e persista o
	 rótulo atribuído por cada execução.
5. **Avaliação**: compare as execuções e mantenha parâmetros, métricas e
	 resultados.

## Decisões em Aberto

- **Algoritmos de agrupamento**: compare inicialmente K-Means e HDBSCAN.
	Registre o snapshot do corpus, normalização, métrica de distância, semente
	aleatória, parâmetros, tamanhos dos grupos e fração de ruído do HDBSCAN.
- **Métricas**: use Silhouette com a distância pretendida e Davies-Bouldin
	para execuções baseadas em centróides; adicione estabilidade entre amostras
	e revisão de especialistas sobre decisões representativas. Se existirem
	casos rotulados, use ARI ou NMI.
- **Estratégia de embeddings**: compare a base com BERTimbau a um modelo de
	embeddings de sentenças compatível com português.
- **Esquema de dados**: decida se os vetores serão armazenados no DuckDB ou
	referenciados a partir de arquivos versionados e defina o armazenamento da
	associação entre documentos e execuções de agrupamento.

## Ambiente

Todas as dependências Python ficarão em `.venv`. Este é um ambiente
provisório de desenvolvimento; fixe as versões e adicione um manifesto de
dependências antes da implementação.

```bash
python -m venv .venv
source .venv/bin/activate
pip install duckdb transformers torch scikit-learn hdbscan requests pypdf
```
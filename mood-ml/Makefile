# Targets mirror the pipeline.py subcommands. Defined by specs/09-orquestracao-ci.md.
#
# Nenhuma lógica aqui além de montar a linha de comando — toda decisão fica
# em pipeline.py (PL-R10, PL-R11). Variáveis passadas via `make <alvo> VAR=valor`.
#
# Nota: a variável de corpus é CORPUS, não PATH — PATH colide com a variável
# de ambiente PATH (busca de executáveis do shell); o GNU Make herda
# variáveis de ambiente com o mesmo nome como valor default de make-variable,
# então usar PATH aqui seria silenciosamente errado sempre que o operador
# esquecesse de passá-la explicitamente.

PYTHON ?= python

.PHONY: ingest labels split train evaluate register promote all

ingest:
	$(PYTHON) -m pipeline ingest $(CORPUS) $(if $(REPORT_DIR),--report-dir $(REPORT_DIR)) $(if $(SKIP_COMPOSITION),--skip-composition)

labels:
	$(PYTHON) -m pipeline labels $(CORPUS) $(if $(REPORT_DIR),--report-dir $(REPORT_DIR)) $(if $(LABELS_DIR),--labels-dir $(LABELS_DIR))

split:
	$(PYTHON) -m pipeline split $(CORPUS) --dataset-id $(DATASET) $(if $(CONFIG),--config $(CONFIG)) $(if $(REPORT_DIR),--report-dir $(REPORT_DIR)) $(if $(DATASETS_DIR),--datasets-dir $(DATASETS_DIR))

train:
	$(PYTHON) -m pipeline train $(DATASET) $(if $(CONFIG),--config $(CONFIG)) $(if $(DATASETS_DIR),--datasets-dir $(DATASETS_DIR)) $(if $(STAGING_DIR),--staging-dir $(STAGING_DIR))

evaluate:
	$(PYTHON) -m pipeline evaluate $(DATASET) $(if $(CONFIG),--config $(CONFIG)) $(if $(STAGING_DIR),--staging-dir $(STAGING_DIR)) $(if $(DATASETS_DIR),--datasets-dir $(DATASETS_DIR))

register:
	$(PYTHON) -m pipeline register $(DATASET) $(FINGERPRINT) $(if $(STAGING_DIR),--staging-dir $(STAGING_DIR)) $(if $(MODELS_DIR),--models-dir $(MODELS_DIR)) $(if $(DATASETS_DIR),--datasets-dir $(DATASETS_DIR))

promote:
	$(PYTHON) -m pipeline promote $(VERSION) $(if $(FORCE),--force) $(if $(MODELS_DIR),--models-dir $(MODELS_DIR))

all:
	$(PYTHON) -m pipeline all $(CORPUS) $(if $(SKIP_COMPOSITION),--skip-composition) $(if $(CONFIG),--config $(CONFIG)) $(if $(REPORT_DIR),--report-dir $(REPORT_DIR)) $(if $(LABELS_DIR),--labels-dir $(LABELS_DIR)) $(if $(DATASETS_DIR),--datasets-dir $(DATASETS_DIR)) $(if $(STAGING_DIR),--staging-dir $(STAGING_DIR)) $(if $(MODELS_DIR),--models-dir $(MODELS_DIR))

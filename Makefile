# Makefile — Cryptonaute end-to-end recipes.
#
# Usage:
#   make install             # editable install + test deps
#   make test                # full suite
#   make test-fast           # skip chaos
#   make fetch               # download all public data into data/features/
#   make train               # fit NFP + CPI nowcast coefficients
#   make generate            # produce synthetic trended history
#   make tune                # tune against the real pipeline
#   make validate            # run the deploy gate
#   make paper               # paper-trade the tuned config
#   make drift               # measure post-release drift on the recording
#   make live                # live (requires config/live.yaml + keys)
#   make clean               # remove generated artifacts

PY ?= python
CONFIG ?= config/default.yaml
HISTORY ?= data/trend.jsonl
SYMBOLS ?= EURUSD,BTCUSDT,GOLD,SPY,QQQ,USO,UNG,GLD,SLV,
STEPS ?= 5000

.PHONY: help install test test-fast fetch train generate tune validate paper drift live clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Editable install + test deps
	$(PY) -m pip install -e ".[test]"

test:  ## Full test suite (including chaos)
	$(PY) -m pytest -q

test-fast:  ## Fast loop — skip chaos
	$(PY) -m pytest -q -m "not chaos"

fetch:  ## Download all public data into data/features/
	$(PY) -m tools.fetch_features --out data/features

train:  ## Fit NFP + CPI nowcast coefficients from data/features/
	$(PY) tools/train_nfp_nowcast.py --csv data/features/nfp_features.csv \
		--out config/nowcast/nfp_coefficients.json
	$(PY) tools/train_cpi_nowcast.py --csv data/features/cpi_features.csv \
		--out config/nowcast/cpi_coefficients.json

generate:  ## Generate synthetic trended history
	$(PY) -m main --generate $(HISTORY) --gen-steps 5000 \
		--trendy 0.0015 --reversion 0.004

tune:  ## Tune against the real pipeline (IS only)
	$(PY) -m main --tune $(HISTORY) --trials 24 --min-trades 10 --max-dd 5

validate:  ## Deploy gate — exit 1 on FAIL
	$(PY) -m main --validate $(HISTORY)

paper:  ## Paper-trade the tuned config
	$(PY) -m main --paper --steps $(STEPS)

drift:  ## Measure post-release drift and suggest rules
	$(PY) -m validation.drift_analysis --history data/events.jsonl \
		--symbols $(SYMBOLS) --out config/drift_rules.yaml \
		--report config/drift_report.json

live:  ## Live trading (requires config/live.yaml + .env)
	$(PY) -m main --live --config config/live.yaml

clean:  ## Remove generated artifacts
	rm -rf data/events.jsonl data/audit.jsonl data/features \
		config/tuned.yaml config/tuned.yaml.meta.json \
		config/drift_rules.yaml config/drift_report.json

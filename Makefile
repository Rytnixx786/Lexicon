# Thin wrappers around the `lexicon` CLI (works without make on Windows: run the CLI directly).
.PHONY: setup up up-model snapshot context-check golden-verify eval ablation bird gate publish report app test test-all lint

UV ?= uv
LX = $(UV) run lexicon
MODEL ?= qwen7b

setup:            ## deps incl. app + ARES gate runtime + the plugin (editable)
	$(UV) sync --all-extras

up:               ## Lexicon's Postgres (eval snapshot + LangGraph checkpoints)
	docker compose up -d postgres

up-model:         ## vLLM + Cinch, capped at 80% VRAM (ADR-017). LEXICON_MODEL selects the model.
	docker compose --profile model up -d

snapshot:         ## restore the frozen warehouse (marts parquet + seeded synthetic support schema)
	$(LX) snapshot

context-check:
	$(LX) context check
	$(LX) context verify

golden-verify:
	$(LX) golden-verify

eval:             ## one golden-set run at L3
	$(LX) eval --model $(MODEL) --level L3 --concurrency 4

ablation:         ## 4 context levels for MODEL (serve it first)
	$(LX) ablation --model $(MODEL) --concurrency 4

bird:             ## BIRD Mini-Dev slice, with and without evidence
	$(LX) bird --model $(MODEL) --concurrency 4

gate:             ## ARES gate on the 3-seed champion configuration
	$(LX) gate qwen7b-L3,qwen7b-L3-s1,qwen7b-L3-s2

publish:          ## send a run's calls to Tally and read cost per question back
	$(LX) publish qwen7b-L3

report:           ## ablation analysis, figures, docs/RESULTS.md, README block
	$(LX) report

app:
	$(LX) app

test:             ## everything that needs no database or GPU
	$(UV) run pytest -m "not integration" --cov --cov-fail-under=85

test-all:         ## also integration tests against the snapshot database
	$(UV) run pytest --cov --cov-fail-under=85

lint:
	$(UV) run ruff check src tests app ares_plugin
	$(UV) run ruff format --check src tests app ares_plugin
	$(UV) run mypy

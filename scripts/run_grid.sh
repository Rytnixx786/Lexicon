#!/usr/bin/env bash
# Full experiment queue (resumable: every model response is recorded, so a restart replays what is done).
# GPU limits (ADR-017): vLLM 80% VRAM + 512-token prefill chunks; concurrency 1; GPU guard from .env.
set -euo pipefail
cd "$(dirname "$0")/.."
LX=".venv/Scripts/lexicon"; [ -x "$LX" ] || LX=".venv/bin/lexicon"

serve() {  # serve one model behind Cinch and wait until healthy
  echo "== serving $1"
  LEXICON_MODEL="$1" docker compose --profile model up -d --force-recreate vllm >/dev/null
  for _ in $(seq 1 90); do sleep 5; curl -sf localhost:8010/health >/dev/null && break; done
  curl -sf localhost:8182/health >/dev/null || docker compose --profile model up -d cinch >/dev/null
  sleep 5
}

serve Qwen/Qwen2.5-7B-Instruct-AWQ
$LX ablation --model qwen7b --concurrency 4
$LX eval --model qwen7b --level L3 --seed 1 --run-id qwen7b-L3-s1 --concurrency 4
$LX eval --model qwen7b --level L3 --seed 2 --run-id qwen7b-L3-s2 --concurrency 4
$LX bird --model qwen7b --concurrency 4

serve Qwen/Qwen2.5-3B-Instruct-AWQ
$LX ablation --model qwen3b --concurrency 4

serve Qwen/Qwen2.5-Coder-7B-Instruct-AWQ
$LX ablation --model coder7b --concurrency 4
$LX bird --model coder7b --concurrency 4

serve Qwen/Qwen2.5-7B-Instruct-AWQ
echo "== grid done"

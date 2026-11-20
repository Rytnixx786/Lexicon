"""LLM calls through Cinch (OpenAI-compatible) with attribution headers and record/replay.

Every call carries X-Request-ID, X-Tenant-ID: lexicon, X-Feature: sql_agent and X-Prompt-Version, so Tally can
attribute its cost (BUILD_SPEC §6). Responses are recorded by request content; LEXICON_MODE=replay serves only
recordings and fails on a miss, so evaluation reruns are bit-for-bit reproducible without a GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx

from lexicon.settings import RECORDINGS_DIR, env

Mode = Literal["live", "replay", "auto"]
TENANT = "lexicon"
FEATURE = "sql_agent"


class ReplayMissError(RuntimeError):
    pass


class SpendNotAllowedError(RuntimeError):
    pass


HOSTED_ENDPOINTS: dict[str, tuple[str, str]] = {
    "gemini-": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "gpt-": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
}


def is_hosted_model(model: str) -> bool:
    return any(model.startswith(p) for p in HOSTED_ENDPOINTS)


def _hosted_endpoint(model: str) -> tuple[str, str]:
    for prefix, pair in HOSTED_ENDPOINTS.items():
        if model.startswith(prefix):
            return pair
    raise ValueError(f"not a hosted model: {model}")


@dataclass(frozen=True)
class LLMRequest:
    model: str
    system: str
    user: str
    max_tokens: int
    prompt_version: str
    choices: tuple[str, ...] | None = None  # guided_choice
    temperature: float = 0.0
    seed: int = 0

    def key(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k != "prompt_version"}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass
class LLMResult:
    request_id: str
    request_key: str
    model: str
    prompt_version: str
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    started_at: float
    http_status: int = 200
    error: str | None = None
    choice_prob: float | None = None
    replayed: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.http_status == 200 and self.error is None


class Recorder:
    def __init__(self, root: Path, mode: Mode) -> None:
        self.root, self.mode = root, mode
        self._idx: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def _file(self, model: str) -> Path:
        return self.root / (model.replace("/", "__") + ".jsonl")

    def _index(self, model: str) -> dict[str, dict[str, Any]]:
        if model not in self._idx:
            idx: dict[str, dict[str, Any]] = {}
            p = self._file(model)
            if p.exists():
                for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip().replace(chr(0), "")
                    if not line:
                        continue
                    try:  # a power loss can leave a torn last line; skip it rather than lose the file
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    idx.setdefault(row["request_key"], row)
            self._idx[model] = idx
        return self._idx[model]

    def get(self, req: LLMRequest) -> dict[str, Any] | None:
        with self._lock:
            return self._index(req.model).get(req.key())

    def put(self, req: LLMRequest, res: LLMResult) -> None:
        if not res.ok:
            return
        with self._lock:
            idx = self._index(req.model)
            if res.request_key in idx:
                return
            row = {k: v for k, v in asdict(res).items() if k not in ("replayed", "request_id")}
            p = self._file(req.model)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
            idx[res.request_key] = row


def _choice_prob(logprobs: dict[str, Any] | None, chosen: str, choices: tuple[str, ...]) -> float | None:
    """P(chosen label) renormalised over valid continuations at each token (same method as Tally ADR-004)."""
    if not logprobs or not logprobs.get("content"):
        return None
    prefix, conf = "", 1.0
    for tok in logprobs["content"]:
        cands = {t["token"]: float(t["logprob"]) for t in tok.get("top_logprobs") or []}
        cands[tok["token"]] = float(tok["logprob"])
        valid = {
            t: lp
            for t, lp in cands.items()
            if (prefix + t).strip() and any(c.startswith((prefix + t).lstrip()) for c in choices)
        }
        if tok["token"] not in valid:
            break
        conf *= math.exp(valid[tok["token"]]) / sum(math.exp(v) for v in valid.values())
        prefix += tok["token"]
        if prefix.strip() == chosen:
            break
    return conf


class LLMClient:
    def __init__(
        self,
        mode: Mode | None = None,
        *,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        allow_spend: bool | None = None,
    ) -> None:
        self.mode: Mode = mode or env("LEXICON_MODE", "auto")  # type: ignore[assignment]
        self.base_url = (base_url or env("CINCH_BASE_URL", "http://localhost:8181")).rstrip("/")
        self.allow_spend = allow_spend if allow_spend is not None else env("LEXICON_ALLOW_SPEND", "0") == "1"
        self.recorder = Recorder(RECORDINGS_DIR, self.mode)
        self._http = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        self._http.close()

    def call(self, req: LLMRequest, request_id: str | None = None) -> LLMResult:
        rid = request_id or str(uuid.uuid4())
        if self.mode in ("replay", "auto"):
            row = self.recorder.get(req)
            if row is not None:
                return LLMResult(request_id=rid, replayed=True, **row)
            if self.mode == "replay":
                raise ReplayMissError(f"no recording for {req.model} key={req.key()[:12]}")
        hosted = is_hosted_model(req.model)
        if hosted and not self.allow_spend:
            raise SpendNotAllowedError(
                f"live hosted call to {req.model} requires --confirm-spend or LEXICON_ALLOW_SPEND=1"
            )
        body: dict[str, Any] = {
            "model": req.model,
            "messages": [{"role": "system", "content": req.system}, {"role": "user", "content": req.user}],
            "max_tokens": max(req.max_tokens, 64) if (hosted and req.choices) else req.max_tokens,
            "temperature": req.temperature,
        }
        if not req.model.startswith("gemini-"):
            body["seed"] = req.seed
        if req.choices:
            if hosted:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "route",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"label": {"type": "string", "enum": list(req.choices)}},
                            "required": ["label"],
                            "additionalProperties": False,
                        },
                    },
                }
            else:
                body["guided_choice"] = list(req.choices)
                body["logprobs"] = True
                body["top_logprobs"] = 10
        if hosted:
            url_base, key_env = _hosted_endpoint(req.model)
            api_key = env(key_env, "")
            if not api_key:
                raise SpendNotAllowedError(f"missing {key_env} for hosted model {req.model}")
            url = f"{url_base}/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        else:
            url = f"{self.base_url}/v1/chat/completions"
            headers = {
                "X-Request-ID": rid,
                "X-Tenant-ID": TENANT,
                "X-Team-ID": "lexicon",
                "X-Feature": FEATURE,
                "X-Prompt-Version": req.prompt_version,
                "X-Prompt-Compaction": "false",
            }
        res: LLMResult | None = None
        from lexicon.gpu_guard import guard

        gpu = (
            guard()
            if (not hosted and self.base_url.startswith(("http://localhost", "http://127.0.0.1")))
            else None
        )
        for attempt in range(4):
            if gpu is not None:
                gpu.before_call()
            started, t0 = time.time(), time.perf_counter()
            try:
                r = self._http.post(url, json=body, headers=headers)
                status, data = r.status_code, (r.json() if r.status_code == 200 else None)
                err = None if status == 200 else r.text[:300]
            except httpx.HTTPError as e:
                status, data, err = 599, None, f"{type(e).__name__}: {e}"
            latency = (time.perf_counter() - t0) * 1000
            if gpu is not None:
                gpu.after_call(latency / 1000)
            if data is not None:
                ch = data["choices"][0]
                content = ch["message"].get("content") or ""
                if hosted and req.choices:
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "label" in parsed:
                            content = str(parsed["label"])
                    except json.JSONDecodeError:
                        pass
                usage = data.get("usage") or {}
                res = LLMResult(
                    request_id=rid,
                    request_key=req.key(),
                    model=req.model,
                    prompt_version=req.prompt_version,
                    content=content,
                    input_tokens=int(usage.get("prompt_tokens") or 0),
                    output_tokens=int(usage.get("completion_tokens") or 0),
                    latency_ms=latency,
                    started_at=started,
                    choice_prob=_choice_prob(ch.get("logprobs"), content.strip(), req.choices)
                    if (req.choices and not hosted)
                    else None,
                )
                break
            res = LLMResult(
                rid, req.key(), req.model, req.prompt_version, "", 0, 0, latency, started, status, err
            )
            if status not in (429, 500, 502, 503, 504, 599):
                break
            time.sleep(min(2**attempt, 15))
        assert res is not None
        self.recorder.put(req, res)
        return res

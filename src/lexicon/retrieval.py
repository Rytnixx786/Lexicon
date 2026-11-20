"""Hybrid retrieval over context cards (BUILD_SPEC §4): BM25 + static embeddings (model2vec), fused with RRF.

Cards: one per table, metric and verified example. Rules are never retrieved by similarity: a rule is attached
whenever a table it applies to is retrieved. Glossary synonyms expand the question before ranking.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from lexicon.context.schema import Context, Example, Metric, Rule, Table
from lexicon.settings import download_dir

EMBED_MODEL = "minishlab/potion-base-8M"
_TOKEN = re.compile(r"[a-z0-9]+")
RRF_K = 60


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().replace("_", " "))


@dataclass
class Retrieved:
    tables: list[Table]
    metrics: list[Metric]
    rules: list[Rule]
    examples: list[Example]
    table_scores: dict[str, float] = field(default_factory=dict)

    @property
    def table_names(self) -> list[str]:
        return [t.table for t in self.tables]


@lru_cache(maxsize=1)
def _embedder() -> object | None:
    if os.environ.get("LEXICON_NO_EMBEDDINGS") == "1":
        return None
    try:
        from model2vec import StaticModel

        os.environ.setdefault("HF_HOME", str(download_dir() / "huggingface"))
        return StaticModel.from_pretrained(EMBED_MODEL)
    except Exception:
        return None


def _embed(texts: list[str]) -> np.ndarray | None:
    model = _embedder()
    if model is None:
        return None
    v = np.asarray(model.encode(texts), dtype=np.float32)  # type: ignore[attr-defined]
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


def table_text(t: Table) -> str:
    cols = " ".join(f"{c.name} {' '.join(c.synonyms)} {c.description}" for c in t.columns if not c.pii)
    return f"{t.table} {t.name} {' '.join(t.synonyms)} {t.description} {t.ai_context} {cols}"


def metric_text(m: Metric) -> str:
    return f"{m.id} {m.label} {' '.join(m.synonyms)} {m.description}"


class _Index:
    def __init__(self, ids: list[str], texts: list[str]) -> None:
        self.ids = ids
        self.bm25 = BM25Okapi([tokens(t) or ["_"] for t in texts])
        self.emb = _embed(texts)

    def rank(self, query: str, query_vec: np.ndarray | None) -> list[tuple[str, float]]:
        bm = self.bm25.get_scores(tokens(query) or ["_"])
        orders = [np.argsort(-bm, kind="stable")]
        if self.emb is not None and query_vec is not None:
            orders.append(np.argsort(-(self.emb @ query_vec), kind="stable"))
        fused = np.zeros(len(self.ids))
        for order in orders:
            for rank, idx in enumerate(order):
                fused[idx] += 1.0 / (RRF_K + rank + 1)
        return sorted(((self.ids[i], float(fused[i])) for i in range(len(self.ids))), key=lambda x: -x[1])


class Retriever:
    def __init__(self, ctx: Context, *, k_tables: int = 4, k_metrics: int = 3, k_examples: int = 3) -> None:
        self.ctx = ctx
        self.k_tables, self.k_metrics, self.k_examples = k_tables, k_metrics, k_examples
        self.tables = _Index(list(ctx.tables), [table_text(t) for t in ctx.tables.values()])
        self.metrics = _Index(list(ctx.metrics), [metric_text(m) for m in ctx.metrics.values()])
        self.examples = (
            _Index([e.id for e in ctx.examples], [e.question for e in ctx.examples]) if ctx.examples else None
        )
        self.embeddings = self.tables.emb is not None

    def expand(self, question: str) -> str:
        q = question.lower()
        extra = [
            key
            for key, syns in self.ctx.glossary.terms.items()
            if any(re.search(rf"\b{re.escape(s)}\b", q) for s in syns)
        ]
        return question + " " + " ".join(extra)

    def retrieve(self, question: str) -> Retrieved:
        q = self.expand(question)
        qv_arr = _embed([q])
        qv = qv_arr[0] if qv_arr is not None else None
        t_rank = self.tables.rank(q, qv)
        chosen = [name for name, _ in t_rank[: self.k_tables]]
        m_rank = [m for m, _ in self.metrics.rank(q, qv)[: self.k_metrics]]
        metrics = [self.ctx.metrics[m] for m in m_rank]
        for m in metrics:  # a metric needs its base table
            if m.base_table not in chosen:
                chosen.append(m.base_table)
        tables = [self.ctx.tables[n] for n in chosen]
        rules = [r for r in self.ctx.rules.values() if any(t in chosen for t in r.applies_to)]
        rule_ids = {r.id for r in rules}
        for m in metrics:
            for rid in m.rule_refs:
                if rid not in rule_ids:
                    rules.append(self.ctx.rules[rid])
                    rule_ids.add(rid)
        examples: list[Example] = []
        if self.examples is not None:
            by_id = {e.id: e for e in self.ctx.examples}
            examples = [by_id[i] for i, _ in self.examples.rank(question, qv)[: self.k_examples]]
        return Retrieved(tables, metrics, rules, examples, dict(t_rank))

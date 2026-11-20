"""The Lexicon agent as a LangGraph StateGraph (BUILD_SPEC §1).

retrieve -> classify -> refuse | clarify | ambiguity_check -> generate -> guard -> execute -> sanity -> answer
Refusal and clarification are graph nodes, not prompt instructions (ADR-002, after Nexus's refusal gate):
refuse makes no LLM call; clarify pauses the graph with interrupt() until the user answers.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from lexicon import sql_guard
from lexicon.context.render import at_least, render
from lexicon.context.schema import Context, Level
from lexicon.executor import ExecError, Executor, ResultFrame
from lexicon.llm import LLMClient, LLMRequest, LLMResult
from lexicon.retrieval import Retriever
from lexicon.settings import PROMPTS_DIR

ROUTES = (
    "answerable",
    "ambiguous",
    "write_operation",
    "pii_or_sensitive",
    "out_of_scope",
    "unanswerable_from_schema",
)
Route = Literal[
    "answerable",
    "ambiguous",
    "write_operation",
    "pii_or_sensitive",
    "out_of_scope",
    "unanswerable_from_schema",
]
REFUSAL_MESSAGES = {
    "write_operation": "I can only read data. Changing, deleting or creating data is not something I do.",
    "pii_or_sensitive": "That asks for personal data (such as people's email addresses), which I do not reveal.",
    "out_of_scope": "That question is outside the support-operations warehouse I can query.",
    "unanswerable_from_schema": "The warehouse has no data that answers this question.",
}
VAGUE_TIME = re.compile(r"\b(recent|recently|lately|these days|nowadays|over time)\b", re.I)
_SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)
_RULES = re.compile(r"RULES_APPLIED:\s*(.*)", re.I)
_GENERIC = {"rate", "time", "to", "resolution", "escalation", "of", "the"}


class LexState(TypedDict, total=False):
    question: str
    context_level: Level
    route: str
    route_confidence: float | None
    retrieved_tables: list[str]
    clarification: dict[str, Any] | None  # {"reason", "options", "answer"}
    sql: str | None
    raw_sql: str | None
    rules_applied: list[str]
    guard: dict[str, Any] | None
    attempts: int
    feedback: str | None
    result: dict[str, Any] | None
    error: dict[str, str] | None
    sanity: list[str]
    answer: dict[str, Any] | None
    refusal: dict[str, str] | None
    calls: list[dict[str, Any]]


def _prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.v1.txt").read_text(encoding="utf-8")


@dataclass
class AgentConfig:
    model: str
    level: Level = "L3"
    max_retries: int = 2
    max_rows: int = sql_guard.MAX_ROWS
    deterministic_triggers: bool = True
    k_tables: int = 4
    seed: int = 0  # >0 only to measure run-to-run nondeterminism (changes the record key)


@dataclass
class Agent:
    ctx: Context
    llm: LLMClient
    executor: Executor
    cfg: AgentConfig
    checkpointer: BaseCheckpointSaver[Any] | None = None
    agent_names: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict
    )  # lowercase display name -> agents

    def __post_init__(self) -> None:
        self.retriever = Retriever(self.ctx, k_tables=self.cfg.k_tables)
        self.allowed = {t.lower() for t in self.ctx.tables}
        self.columns = {name: {c.name: c.type for c in t.columns} for name, t in self.ctx.tables.items()}
        self.graph = self._build().compile(checkpointer=self.checkpointer or MemorySaver())

    # ------------------------------------------------------------ helpers

    def _pv(self, node: str) -> str:
        return f"sql_agent.{node}.{self.cfg.level}.v1"

    def _call(
        self,
        state: LexState,
        node: str,
        system: str,
        user: str,
        max_tokens: int,
        choices: tuple[str, ...] | None = None,
    ) -> LLMResult:
        req = LLMRequest(
            self.cfg.model, system, user, max_tokens, self._pv(node), choices, seed=self.cfg.seed
        )
        res = self.llm.call(req)
        state.setdefault("calls", []).append(
            {
                "node": node,
                "request_id": res.request_id,
                "model": res.model,
                "prompt_version": res.prompt_version,
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens,
                "latency_ms": res.latency_ms,
                "started_at": res.started_at,
                "http_status": res.http_status,
                "replayed": res.replayed,
            }
        )
        return res

    def _context(self, question: str) -> tuple[str, list[str]]:
        r = self.retriever.retrieve(question)
        return render(self.ctx, r, self.cfg.level), r.table_names

    # ------------------------------------------------------------ nodes

    def n_retrieve(self, state: LexState) -> dict[str, Any]:
        _, tables = self._context(state["question"])
        return {"retrieved_tables": tables, "attempts": 0, "calls": state.get("calls", []), "sanity": []}

    def n_classify(self, state: LexState) -> dict[str, Any]:
        context, _ = self._context(state["question"])
        res = self._call(
            state,
            "classify",
            _prompt("classify").format(context=context),
            f"Question: {state['question']}\nLabel:",
            12,
            ROUTES,
        )
        route = res.content.strip() if res.ok and res.content.strip() in ROUTES else "answerable"
        return {"route": route, "route_confidence": res.choice_prob, "calls": state["calls"]}

    def deterministic_ambiguity(self, question: str) -> dict[str, Any] | None:
        """Rule-based triggers from the semantic layer (>= L2 only; L0/L1 rely on the classifier)."""
        if not (self.cfg.deterministic_triggers and at_least(self.cfg.level, "L2")):
            return None
        q = question.lower()
        words = set(re.findall(r"[a-z0-9]+", q))
        for m in self.ctx.metrics.values():
            if not m.requires_choice or not any(
                re.search(rf"\b{re.escape(s.lower())}\b", q) for s in m.synonyms
            ):
                continue
            keys = [set(c.id.split("_")) - _GENERIC for c in m.choices]
            if not any(k & words for k in keys):
                return {
                    "reason": f"'{m.label}' has more than one definition",
                    "options": [c.description for c in m.choices],
                }
        if VAGUE_TIME.search(q):
            return {
                "reason": "the time window is not specified",
                "options": ["last 7 days", "last 30 days", "per month over the whole period"],
            }
        for name, agents in self.agent_names.items():
            if len(agents) > 1 and name in q:
                return {
                    "reason": f"more than one agent is called {name.title()}",
                    "options": [
                        f"{a['agent_id']} ({a['team']}, {'active' if a['is_active'] else 'inactive'})"
                        for a in agents
                    ],
                }
        return None

    def n_ambiguity_check(self, state: LexState) -> dict[str, Any]:
        if state.get("clarification"):
            return {}
        trig = self.deterministic_ambiguity(state["question"])
        return {"route": "ambiguous", "clarification": trig} if trig else {}

    def n_clarify(self, state: LexState) -> dict[str, Any]:
        clar = dict(state.get("clarification") or {})
        if not clar.get("options"):
            context, _ = self._context(state["question"])
            res = self._call(
                state,
                "clarify",
                _prompt("clarify").format(context=context),
                f"Question: {state['question']}",
                160,
            )
            lines = [ln.strip(" -*0123456789.)\t") for ln in res.content.splitlines()]
            clar = {
                "reason": "the question has more than one reasonable reading",
                "options": [ln for ln in lines if ln][:4],
            }
        answer = interrupt(
            {"question": state["question"], "reason": clar.get("reason"), "options": clar.get("options", [])}
        )
        clar["answer"] = str(answer)
        return {"clarification": clar, "calls": state.get("calls", [])}

    def n_refuse(self, state: LexState) -> dict[str, Any]:
        kind = state["route"]
        return {"refusal": {"type": kind, "message": REFUSAL_MESSAGES[kind]}}

    def n_generate(self, state: LexState) -> dict[str, Any]:
        question = state["question"]
        clar = state.get("clarification") or {}
        query = f"{question} {clar['answer']}" if clar.get("answer") else question
        context, _ = self._context(query)
        rules_line = (
            "- Apply every BUSINESS RULE listed below that is relevant."
            if at_least(self.cfg.level, "L2")
            else ""
        )
        user = f"Question: {question}"
        if clar.get("answer"):
            user += f"\nClarification from the user: {clar['answer']}"
        if state.get("feedback"):
            user += f"\nYour previous query failed: {state['feedback']}\nPrevious query:\n{state.get('raw_sql')}\nFix it."
        res = self._call(
            state, "generate", _prompt("generate").format(context=context, rules_line=rules_line), user, 700
        )
        m = _SQL_BLOCK.search(res.content)
        sql = (m.group(1) if m else res.content).strip()
        rm = _RULES.search(res.content)
        rules = (
            []
            if not rm or rm.group(1).strip().lower() == "none"
            else [x.strip() for x in rm.group(1).split(",") if x.strip()]
        )
        return {
            "raw_sql": sql,
            "rules_applied": rules,
            "attempts": state.get("attempts", 0) + 1,
            "calls": state["calls"],
        }

    def n_guard(self, state: LexState) -> dict[str, Any]:
        rep = sql_guard.check(
            state.get("raw_sql") or "",
            self.allowed,
            dialect=self.executor.dialect,
            columns=self.columns,
            max_rows=self.cfg.max_rows,
        )
        g = {
            "ok": rep.ok,
            "codes": rep.codes,
            "messages": [v.message for v in rep.violations],
            "tables": rep.tables,
        }
        if rep.ok:
            return {"guard": g, "sql": rep.sql, "feedback": None}
        return {
            "guard": g,
            "sql": None,
            "feedback": "; ".join(f"{v.code}: {v.message}" for v in rep.violations),
        }

    def n_execute(self, state: LexState) -> dict[str, Any]:
        try:
            frame = self.executor.run(state["sql"] or "")
        except ExecError as e:
            return {
                "error": {"kind": e.kind, "message": e.message},
                "result": None,
                "feedback": f"{e.kind}: {e.message}",
            }
        return {"result": _frame_dict(frame), "error": None, "feedback": None}

    def n_sanity(self, state: LexState) -> dict[str, Any]:
        r = state.get("result")
        if not r:
            return {"sanity": []}
        flags = []
        rows, cols = r["rows"], r["columns"]
        if not rows:
            flags.append("empty_result")
        if r["truncated"]:
            flags.append("row_cap_hit")
        for i, c in enumerate(cols):
            vals = [row[i] for row in rows]
            if vals and sum(v is None for v in vals) / len(vals) > 0.5:
                flags.append(f"null_heavy:{c}")
            col = next(
                (
                    col
                    for t in self.ctx.tables.values()
                    for col in t.columns
                    if col.name == c and col.expected_range
                ),
                None,
            )
            if (
                col
                and col.expected_range
                and any(
                    isinstance(v, (int, float)) and not (col.expected_range[0] <= v <= col.expected_range[1])
                    for v in vals
                )
            ):
                flags.append(f"magnitude:{c}")
        return {"sanity": flags}

    def n_answer(self, state: LexState) -> dict[str, Any]:
        r = state.get("result")
        if not r:
            err = state.get("error") or {
                "kind": "guard",
                "message": "; ".join((state.get("guard") or {}).get("messages", [])),
            }
            return {
                "answer": {
                    "text": f"I could not produce a valid query ({err['kind']}: {err['message']}).",
                    "ok": False,
                }
            }
        rows, cols = r["rows"], r["columns"]
        if len(rows) == 1 and len(cols) == 1:
            text = f"{cols[0]} = {rows[0][0]}"
        else:
            text = f"{len(rows)} row(s) returned" + (" (truncated at the row cap)" if r["truncated"] else "")
        caveats = [self.ctx.rules[x].statement for x in state.get("rules_applied", []) if x in self.ctx.rules]
        caveats += [f"Check: {f}" for f in state.get("sanity", [])]
        return {
            "answer": {
                "text": text,
                "ok": True,
                "sql": state.get("sql"),
                "chart": chart_spec(cols, rows),
                "caveats": caveats,
            }
        }

    # ------------------------------------------------------------ edges

    def _after_classify(self, state: LexState) -> str:
        route = state["route"]
        if route in REFUSAL_MESSAGES:
            return "refuse"
        return "clarify" if route == "ambiguous" else "ambiguity_check"

    def _after_ambiguity(self, state: LexState) -> str:
        return (
            "clarify"
            if state.get("route") == "ambiguous" and not (state.get("clarification") or {}).get("answer")
            else "generate"
        )

    def _after_guard(self, state: LexState) -> str:
        if (state.get("guard") or {}).get("ok"):
            return "execute"
        return "generate" if state.get("attempts", 0) <= self.cfg.max_retries else "answer"

    def _after_execute(self, state: LexState) -> str:
        if state.get("result") is not None:
            return "sanity"
        return "generate" if state.get("attempts", 0) <= self.cfg.max_retries else "answer"

    def _build(self) -> StateGraph[LexState]:
        g: StateGraph[LexState] = StateGraph(LexState)
        for name in (
            "retrieve",
            "classify",
            "ambiguity_check",
            "clarify",
            "refuse",
            "generate",
            "guard",
            "execute",
            "sanity",
            "answer",
        ):
            g.add_node(name, getattr(self, f"n_{name}"))
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "classify")
        g.add_conditional_edges("classify", self._after_classify, ["refuse", "clarify", "ambiguity_check"])
        g.add_conditional_edges("ambiguity_check", self._after_ambiguity, ["clarify", "generate"])
        g.add_edge("clarify", "generate")
        g.add_edge("generate", "guard")
        g.add_conditional_edges("guard", self._after_guard, ["execute", "generate", "answer"])
        g.add_conditional_edges("execute", self._after_execute, ["sanity", "generate", "answer"])
        g.add_edge("sanity", "answer")
        g.add_edge("refuse", END)
        g.add_edge("answer", END)
        return g

    # ------------------------------------------------------------ API

    def start(self, question: str, thread_id: str | None = None) -> tuple[str, dict[str, Any]]:
        tid = thread_id or str(uuid.uuid4())
        out = self.graph.invoke(
            {"question": question, "context_level": self.cfg.level, "calls": []},
            {"configurable": {"thread_id": tid}},
        )
        return tid, out

    def resume(self, thread_id: str, answer: str) -> dict[str, Any]:
        out: dict[str, Any] = self.graph.invoke(
            Command(resume=answer), {"configurable": {"thread_id": thread_id}}
        )
        return out


def _frame_dict(f: ResultFrame) -> dict[str, Any]:
    return {
        "columns": f.columns,
        "rows": [list(r) for r in f.rows],
        "truncated": f.truncated,
        "elapsed_ms": f.elapsed_ms,
    }


def chart_spec(cols: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    """Pick a chart: time + number -> line; category + number -> bar; one number -> KPI tile; else table."""
    if len(rows) == 1 and len(cols) == 1:
        return {"type": "kpi", "label": cols[0]}
    if len(cols) == 2 and rows:
        x, y = rows[0][0], rows[0][1]
        if isinstance(y, (int, float)) and not isinstance(y, bool):
            is_time = hasattr(x, "isoformat")
            return {"type": "line" if is_time else "bar", "x": cols[0], "y": cols[1]}
    return {"type": "table"}


def is_interrupted(out: dict[str, Any]) -> bool:
    return bool(out.get("__interrupt__"))


def interrupt_payload(out: dict[str, Any]) -> dict[str, Any]:
    intr = out.get("__interrupt__") or []
    return dict(intr[0].value) if intr else {}

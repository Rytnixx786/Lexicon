"""Lexicon chat app: ask a question, answer clarifications, see the SQL, a chart and the caveats.

    lexicon app        # or: streamlit run app/app.py
The agent reads the frozen snapshot (EVAL_DSN) by default; set LEXICON_APP_DSN to point it at Tally's live
warehouse through the read-only role. Clarifications pause the LangGraph thread with interrupt() and resume it.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from lexicon.context import loader
from lexicon.eval.runner import _agent_names
from lexicon.executor import PostgresExecutor
from lexicon.graph import Agent, AgentConfig, interrupt_payload, is_interrupted
from lexicon.llm import LLMClient
from lexicon.settings import env, eval_dsn

st.set_page_config(page_title="Lexicon", layout="wide")
MODELS = {
    "Qwen2.5-7B AWQ": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "Qwen2.5-Coder-7B AWQ": "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
    "Qwen2.5-3B AWQ": "Qwen/Qwen2.5-3B-Instruct-AWQ",
    "Gemini 2.5 Flash-Lite": "gemini-2.5-flash-lite",
}


@st.cache_resource
def agent(model: str, level: str) -> Agent:
    dsn = os.environ.get("LEXICON_APP_DSN") or eval_dsn()
    ex = PostgresExecutor(dsn)
    return Agent(
        loader.load(),
        LLMClient(mode=env("LEXICON_MODE", "auto")),
        ex,  # type: ignore[arg-type]
        AgentConfig(model=model, level=level),
        agent_names=_agent_names(ex),
    )  # type: ignore[arg-type]


with st.sidebar:
    st.header("Lexicon")
    model = MODELS[st.selectbox("Model (served by Cinch)", list(MODELS))]
    level = st.select_slider(
        "Context level",
        ["L0", "L1", "L2", "L3"],
        value="L3",
        help="L0 schema only, L1 + descriptions, L2 + rules and metrics, L3 + verified examples",
    )
    st.caption(
        "Read-only by construction: SQL guard, read-only role, read-only transaction, 5 s timeout, 1,000-row cap."
    )
    if st.button("New conversation"):
        st.session_state.clear()

st.session_state.setdefault("history", [])
st.session_state.setdefault("pending", None)


def show(out: dict[str, Any], idx: int = 0) -> None:
    if out.get("refusal"):
        st.warning(f"Refused ({out['refusal']['type'].replace('_', ' ')}): {out['refusal']['message']}")
        return
    ans = out.get("answer") or {}
    st.markdown(f"**{ans.get('text', '')}**")
    res = out.get("result")
    if res and res["rows"]:
        df = pd.DataFrame(res["rows"], columns=res["columns"])
        chart = ans.get("chart", {})
        if chart.get("type") == "kpi":
            st.metric(chart.get("label", ""), f"{df.iloc[0, 0]}")
        elif chart.get("type") in ("bar", "line"):
            f = (px.bar if chart["type"] == "bar" else px.line)(df, x=chart["x"], y=chart["y"])
            f.update_traces(marker_color="#2a78d6") if chart["type"] == "bar" else f.update_traces(
                line_color="#2a78d6"
            )
            st.plotly_chart(f, width="stretch", key=f"chart-{idx}")
        st.dataframe(df, width="stretch", hide_index=True, key=f"df-{idx}")
    if out.get("sql"):
        with st.expander("SQL (as executed, after the guard)"):
            st.code(out["sql"], language="sql")
    for c in ans.get("caveats", []):
        st.caption(f"Caveat: {c}")
    calls = out.get("calls", [])
    st.caption(
        f"{len(calls)} model calls · {sum(c['input_tokens'] + c['output_tokens'] for c in calls):,} tokens · "
        f"route: {out.get('route')} · tables: {', '.join(out.get('retrieved_tables', []))}"
    )


for idx, (role, payload) in enumerate(st.session_state["history"]):
    with st.chat_message(role):
        if role == "assistant":
            show(payload, idx)
        else:
            st.write(payload)

pending = st.session_state["pending"]
if pending:
    with st.chat_message("assistant"):
        st.info(f"Before I answer: {pending['reason']}. Which do you mean?")
        cols = st.columns(len(pending["options"]) or 1)
        for col, opt in zip(cols, pending["options"], strict=False):
            if col.button(opt, key=f"opt-{opt}"):
                out = agent(model, level).resume(pending["thread"], opt)
                st.session_state["history"].append(("assistant", out))
                st.session_state["pending"] = None
                st.rerun()

if q := st.chat_input("Ask about triage accuracy, costs, tickets, agents, SLAs..."):
    st.session_state["history"].append(("user", q))
    tid, out = agent(model, level).start(q, str(uuid.uuid4()))
    if is_interrupted(out):
        p = interrupt_payload(out)
        st.session_state["pending"] = {
            "thread": tid,
            "reason": p.get("reason"),
            "options": p.get("options", []),
        }
    else:
        st.session_state["history"].append(("assistant", out))
    st.rerun()

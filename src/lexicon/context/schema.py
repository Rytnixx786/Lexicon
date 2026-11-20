"""Context-layer models (docs/CONTEXT_YAML_SPEC.md). The pydantic models are the authority."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Level = Literal["L0", "L1", "L2", "L3"]
LEVELS: tuple[Level, ...] = ("L0", "L1", "L2", "L3")


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Column(_M):
    name: str
    type: str
    description: str
    synonyms: list[str] = Field(default_factory=list)
    is_enum: bool = False
    value_meanings: dict[str, str] = Field(default_factory=dict)
    unit: str | None = None
    expected_range: tuple[float, float] | None = None
    rule_refs: list[str] = Field(default_factory=list)
    pii: bool = False
    level: Level = "L1"
    override_reason: str | None = None


class Table(_M):
    version: int = 1
    table: str
    grain: str
    primary_key: list[str]
    description: str
    ai_context: str = ""
    synonyms: list[str] = Field(default_factory=list)
    dbt_model: str | None = None
    synthetic: bool = False
    columns: list[Column]
    rule_refs: list[str] = Field(default_factory=list)
    metric_refs: list[str] = Field(default_factory=list)

    @property
    def schema_name(self) -> str:
        return self.table.split(".")[0]

    @property
    def name(self) -> str:
        return self.table.split(".")[1]

    def column(self, name: str) -> Column | None:
        return next((c for c in self.columns if c.name == name), None)


class Choice(_M):
    id: str
    description: str
    sql: str | None = None


class Metric(_M):
    id: str
    label: str
    type: Literal["simple", "ratio", "derived"]
    base_table: str
    description: str
    expr: str | None = None
    numerator: str | None = None
    denominator: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    rule_refs: list[str] = Field(default_factory=list)
    expected_range: tuple[float, float] | None = None
    requires_choice: bool = False
    choices: list[Choice] = Field(default_factory=list)
    source_of_truth: str | None = None

    @model_validator(mode="after")
    def _shape(self) -> Metric:
        if self.requires_choice and len(self.choices) < 2:
            raise ValueError(f"metric {self.id}: requires_choice needs >= 2 choices")
        if not self.requires_choice:
            if self.type == "ratio" and not (self.numerator and self.denominator):
                raise ValueError(f"ratio metric {self.id} needs numerator and denominator")
            if self.type in ("simple", "derived") and not self.expr:
                raise ValueError(f"metric {self.id} needs expr")
        return self


class Rule(_M):
    id: str
    statement: str
    applies_to: list[str]
    sql_hint: str | None = None
    severity: Literal["must", "should"] = "must"
    level: Level = "L2"


class Join(_M):
    left: str
    right: str
    condition: str  # "on" is a YAML 1.1 boolean, so the key is named condition
    relationship: Literal["one_to_one", "one_to_many", "many_to_one"]
    note: str | None = None


class ForbiddenJoin(_M):
    pair: tuple[str, str]
    reason: str


class Joins(_M):
    version: int = 1
    joins: list[Join]
    forbidden: list[ForbiddenJoin] = Field(default_factory=list)


class Example(_M):
    id: str
    question: str
    sql: str
    tables: list[str]
    metric_refs: list[str] = Field(default_factory=list)
    rule_refs: list[str] = Field(default_factory=list)
    # Verification (result hash, date) is recorded by `lexicon context verify` in context/_verified_examples.json
    # and rechecked in CI, so a snapshot change forces re-verification.
    verified_by: str = "maintainer"


class Glossary(_M):
    version: int = 1
    terms: dict[str, list[str]]


class Context(_M):
    tables: dict[str, Table]
    metrics: dict[str, Metric]
    rules: dict[str, Rule]
    joins: Joins
    examples: list[Example]
    glossary: Glossary
    samples: dict[str, dict[str, list[str]]] = Field(default_factory=dict)  # table -> column -> values

    @model_validator(mode="after")
    def _refs_resolve(self) -> Context:
        problems: list[str] = []
        for table in self.tables.values():
            refs = [*table.rule_refs, *(ref for c in table.columns for ref in c.rule_refs)]
            problems += [f"{table.table}: unknown rule {ref}" for ref in refs if ref not in self.rules]
            problems += [
                f"{table.table}: unknown metric {mid}" for mid in table.metric_refs if mid not in self.metrics
            ]
        for metric in self.metrics.values():
            if metric.base_table not in self.tables:
                problems.append(f"metric {metric.id}: unknown base table {metric.base_table}")
            problems += [
                f"metric {metric.id}: unknown rule {ref}" for ref in metric.rule_refs if ref not in self.rules
            ]
        for rule in self.rules.values():
            problems += [
                f"rule {rule.id}: unknown table {name}" for name in rule.applies_to if name not in self.tables
            ]
        for j in self.joins.joins:
            problems += [
                f"join {j.left}->{j.right}: unknown table {name}"
                for name in (j.left, j.right)
                if name not in self.tables
            ]
        ids = [e.id for e in self.examples]
        if len(ids) != len(set(ids)):
            problems.append("duplicate example ids")
        for e in self.examples:
            problems += [
                f"example {e.id}: unknown table {name}" for name in e.tables if name not in self.tables
            ]
        if problems:
            raise ValueError("; ".join(problems))
        return self

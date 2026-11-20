"""Golden set: 120 questions in six buckets (docs/GOLDEN_SET_GUIDE.md)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lexicon.settings import GOLDEN_DIR

Bucket = Literal[
    "simple", "multi_join", "metric_definition", "business_rule_trap", "should_clarify", "should_refuse"
]
BUCKETS: tuple[Bucket, ...] = (
    "simple",
    "multi_join",
    "metric_definition",
    "business_rule_trap",
    "should_clarify",
    "should_refuse",
)
ANSWERABLE: tuple[Bucket, ...] = ("simple", "multi_join", "metric_definition", "business_rule_trap")
RefusalType = Literal["write_operation", "out_of_scope", "pii_or_sensitive", "unanswerable_from_schema"]
REFUSAL_TYPES: tuple[RefusalType, ...] = (
    "write_operation",
    "out_of_scope",
    "pii_or_sensitive",
    "unanswerable_from_schema",
)
VERIFIED_FILE = GOLDEN_DIR / "verified.json"


class Clarification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ambiguity_type: Literal["metric_choice", "time_window", "entity", "scope", "attachment"]
    acceptable_options: list[str]
    simulated_answer: str
    gold_sql_after_answer: str


class Refusal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: RefusalType


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    bucket: Bucket
    question: str
    difficulty: Literal["simple", "moderate", "hard"] = "moderate"
    tests_rules: list[str] = Field(default_factory=list)
    tests_metrics: list[str] = Field(default_factory=list)
    common_wrong_answers: list[str] = Field(default_factory=list)
    gold_sql: str | None = None
    clarification: Clarification | None = None
    refusal: Refusal | None = None

    @model_validator(mode="after")
    def _shape(self) -> Item:
        if self.bucket in ANSWERABLE and not self.gold_sql:
            raise ValueError(f"{self.id}: answerable buckets need gold_sql")
        if self.bucket == "should_clarify" and self.clarification is None:
            raise ValueError(f"{self.id}: should_clarify needs a clarification block")
        if self.bucket == "should_refuse" and self.refusal is None:
            raise ValueError(f"{self.id}: should_refuse needs a refusal block")
        return self

    @property
    def scored_sql(self) -> str | None:
        return self.clarification.gold_sql_after_answer if self.clarification else self.gold_sql


def load(root: Path = GOLDEN_DIR) -> list[Item]:
    items: list[Item] = []
    for b in BUCKETS:
        data = yaml.safe_load((root / f"{b}.yaml").read_text(encoding="utf-8"))
        if data["bucket"] != b:
            raise ValueError(f"{b}.yaml declares bucket {data['bucket']}")
        items += [Item.model_validate({**x, "bucket": b}) for x in data["items"]]
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate golden ids")
    return items


def version(root: Path = GOLDEN_DIR) -> str:
    h = hashlib.sha256()
    for b in BUCKETS:
        h.update((root / f"{b}.yaml").read_bytes())
    return "golden-" + h.hexdigest()[:12]


def load_verified() -> dict[str, dict[str, object]]:
    if not VERIFIED_FILE.exists():
        return {}
    data: dict[str, dict[str, object]] = json.loads(VERIFIED_FILE.read_text(encoding="utf-8"))["items"]
    return data

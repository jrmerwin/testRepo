"""
EFF (Emergent Formalism Framework) - core primitives and infrastructure.

This module is intentionally self-contained:
- no external YAML prompt/config files
- configuration and prompts live in DEFAULT_CONFIG / PROMPTS
- agents/pipelines import from here

Design goals:
- portable (drop-in package)
- deterministic file layout for outputs
- pluggable LLM backend (OpenAI-compatible if available, otherwise user-provided)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import json
import math
import hashlib
import re
import ast


# Optional deps
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise ImportError("EFF requires pandas. Please install pandas.") from e


# =========================
# Defaults: config & prompts
# =========================

DEFAULT_CONFIG: Dict[str, Any] = {
    "project_name": "eff_run",
    "outputs_dir": "outputs",

    # Columns excluded from features, targets, and transactions everywhere.
    # Add ground-truth or metadata columns that must never leak into analysis.
    "exclude_columns": ["true_axiom_triggered"],

    # LLM
    "llm": {
        "provider": "openai",          # "openai" | "anthropic" | "custom"
        "model": "gpt-4.1-mini",
        "temperature": 0.2,
        "max_tokens": 1200,
        "request_timeout_s": 90,
        "max_retries": 2,
        "base_url": None,              # e.g. "http://localhost:11434/v1"
        "api_key_env": "OPENAI_API_KEY",
    },

    # Initializer (dynamic domain adaptation)
    "initializer": {
        "enabled": True,
        "sample_rows": 50,
        "accept_generated_python_transform": False,  # safety: default off
        "max_columns_in_prompt": 40,
    },

    # Convergence / iteration
    "convergence": {
        "max_rounds": 12,
        "plateau_threshold": 0.02,
        "plateau_rounds": 3,
    },

    # Phase 1 extraction (LLM-mediated)
    "phase1": {
        "n_agents": 3,
        "batch_size": 64,
        "max_formalisms_per_round": 12,
        "dedup": {
            "enabled": True,
            "method": "text",  # "text" | "embeddings"
            "text_similarity_threshold": 0.93,
            "embedding_similarity_threshold": 0.92,
        },
        "validation": {
            "min_confidence": 0.65,
            "min_support": 3,
            "max_contradictions": 1,
            "holdout_frac": 0.2,
            "max_examples_per_formalism": 120,
        }
    },

    # Combinatorial search (non-LLM systematic rule discovery)
    "combinatorial": {
        "enabled": True,
        "min_support": 5,
        "min_confidence": 0.80,
        # Percentiles at which to slice numeric columns for threshold rules.
        # Finer near tails where interesting thresholds typically live.
        "numeric_percentiles": [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95],
        "max_pair_columns": 6,      # max categorical cols considered for pairwise combinations
        "max_unique_values": 20,    # skip categoricals with more unique values than this
        "max_formalisms": 200,      # output cap, ranked by quality before truncation
    },

    # Phase 2 synthesis
    "phase2": {
        "max_tokens": 4000,
        "max_synthesis_rounds": 4,
        "max_axioms": 60,
        "min_axiom_confidence": 0.70,
        "min_support": 3,
        "max_contradictions": 1,
    },

    # Classical tools (reference report only, not fed into axiom output)
    "classical": {
        "enabled": True,
        "target_columns": None,  # None => auto-detect
        "decision_tree": {
            "max_depth": 5,
            "min_samples_leaf": 15,
            "min_samples_split": 30,
        },
        "association_rules": {
            "enabled": True,
            "min_support": 0.08,
            "min_confidence": 0.55,
            "max_rules": 200,
        }
    }
}


PROMPTS: Dict[str, Any] = {
    # ---------- Initializer ----------
    "initializer": {
        "analysis_system": (
            "You are an expert data scientist helping adapt an Emergent Formalism Framework (EFF) "
            "to a new dataset/domain. You will be given a dataset schema and sample rows, plus the user's "
            "goal for what kinds of causal/structural regularities to discover."
        ),
        "analysis_user": (
            "DATASET SUMMARY\n"
            "- domain_name: {domain_name}\n"
            "- description: {domain_description}\n"
            "- goals: {goals}\n\n"
            "SCHEMA\n{schema}\n\n"
            "SAMPLE ROWS (JSON)\n{sample_rows}\n\n"
            "Return JSON with keys:\n"
            "  key_variables: [..]\n"
            "  likely_relationships: [..]\n"
            "  suggested_groupings: [..]\n"
            "  suggested_derived_fields: [{{\"name\":\"...\",\"rationale\":\"...\",\"example_expression\":\"...\"}}]\n"
            "  risks_or_pitfalls: [..]\n"
        ),
        "generation_system": (
            "You generate EFF extraction prompts and a safe transformation plan. "
            "You must return strict JSON only."
        ),
        "generation_user": (
            "Using the analysis below, generate the following. Return strict JSON with the keys shown.\n\n"

            "1) observation_prompt_template (string):\n"
            "   A prompt that will be sent to an LLM with actual data rows to propose candidate formalisms.\n"
            "   REQUIRED placeholders that MUST appear literally in the string:\n"
            "     {{batch_rows}}  -- where the JSON data rows will be inserted\n"
            "     {{domain_name}}  -- the dataset domain name\n"
            "     {{domain_description}}  -- the dataset description\n"
            "   REQUIRED instructions that MUST be in the prompt text:\n"
            "     - Tell the LLM to return JSON with a key 'formalisms' containing a list of objects.\n"
            "     - Each object must have a 'statement' field (string) describing the rule.\n"
            "   Use the domain knowledge from the analysis to guide WHAT kinds of rules to look for,\n"
            "   but preserve the above structural requirements exactly.\n\n"

            "2) test_prompt_template (string):\n"
            "   A prompt that will be sent to an LLM to validate a single formalism against data rows.\n"
            "   REQUIRED placeholders that MUST appear literally in the string:\n"
            "     {{formalism_statement}}  -- the formalism text being tested\n"
            "     {{rows}}  -- the JSON data rows to test against\n"
            "   REQUIRED instructions that MUST be in the prompt text:\n"
            "     - Tell the LLM to return strict JSON with these keys:\n"
            "       support_count (int), contradiction_count (int), confidence (float 0-1),\n"
            "       support_examples (list), contradiction_examples (list), notes (string).\n\n"

            "3) transform_spec (object):\n"
            "   Safe, reviewable transformation spec. Must have a 'derived_columns' key containing a list.\n"
            "   Each item: {{\"name\": \"...\", \"expression\": \"...\", \"dtype\": \"...\", \"description\": \"...\"}}\n"
            "   Expressions must be valid Python (pandas eval or numpy). Do NOT use SQL syntax\n"
            "   (CASE WHEN, BETWEEN), pseudo-code (if/then/else), or C-style ternaries (? :).\n"
            "   Use numpy: np.where(condition, true_val, false_val) for conditionals.\n\n"

            "4) suggested_config_overrides (object):\n"
            "   Config key/value overrides for EFF based on dataset size and shape. Can be empty.\n\n"

            "5) optional_python_transform (string or null):\n"
            "   A Python function as a string 'def transform(df): ... return df'. Only include if helpful.\n\n"

            "ANALYSIS JSON\n{analysis_json}\n\n"
            "DATASET SCHEMA\n{schema}\n\n"
        ),
    },

    # ---------- Extraction (baseline; usually overridden by initializer) ----------
    "extraction": {
        "system_base": (
            "You are a formalism discovery agent. Propose concise candidate formalisms (axioms/rules) "
            "from a dataset. Prefer simple, testable, falsifiable statements linking variables."
        ),
        "observation_prompt_template": (
            "Domain: {domain_name}\n"
            "Description: {domain_description}\n"
            "You are given a batch of observations (rows). Identify 1-3 candidate formalisms.\n\n"
            "Rules:\n"
            "- Each formalism must be testable on rows.\n"
            "- Use variables/columns present in the data.\n"
            "- Prefer if/then or functional relationships.\n"
            "- Return JSON list under key 'formalisms'.\n\n"
            "BATCH ROWS:\n{batch_rows}\n"
        ),
        "test_prompt_template": (
            "You are validating a candidate formalism against data.\n"
            "Formalism:\n{formalism_statement}\n\n"
            "Given the rows below, count how many support the formalism and how many contradict it.\n"
            "Return strict JSON:\n"
            "{\n"
            "  \"support_count\": int,\n"
            "  \"contradiction_count\": int,\n"
            "  \"confidence\": float,  # 0..1\n"
            "  \"support_examples\": [row_ids],\n"
            "  \"contradiction_examples\": [row_ids],\n"
            "  \"notes\": \"short\"\n"
            "}\n\n"
            "ROWS:\n{rows}\n"
        )
    },

    # ---------- Synthesis ----------
    "synthesis": {
        "system_base": (
            "You are an axiom synthesis agent. You combine validated formalisms into a compact set of axioms. "
            "You preserve testability and avoid redundancy."
        ),
        "weave_prompt": (
            "Given the accepted formalisms below, synthesize a de-duplicated set of axioms.\n"
            "Return strict JSON: {\"axioms\":[{number:int, statement:str, confidence:float, support_count:int, contradiction_count:int, reasoning:str}]}\n\n"
            "FORMALISMS:\n{formalisms_json}\n"
        ),
        "moderation_prompt": (
            "Given candidate axioms, remove redundancy and contradictions. Prefer simpler axioms.\n"
            "Return strict JSON with the same schema under key 'axioms'.\n\n"
            "CANDIDATE AXIOMS:\n{axioms_json}\n"
        ),
    }
}


def deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Deep-merge override into base (non-destructive)."""
    if not override:
        return dict(base)
    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# =========
# Data model
# =========

class FormalismType(str, Enum):
    AXIOM = "axiom"
    RULE = "rule"
    RELATION = "relation"


@dataclass
class Formalism:
    """A candidate/tested formalism (rule) discovered from the dataset."""
    statement: str
    formalism_type: FormalismType = FormalismType.RULE
    confidence: float = 0.0
    support_count: int = 0
    contradiction_count: int = 0
    reasoning: str = ""
    source: str = ""
    id: str = field(default_factory=lambda: _short_id("F"))

    # optional: embedding vector for dedup
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["formalism_type"] = self.formalism_type.value
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Formalism":
        return Formalism(
            statement=d["statement"],
            formalism_type=FormalismType(d.get("formalism_type", "rule")),
            confidence=float(d.get("confidence", 0.0)),
            support_count=int(d.get("support_count", 0)),
            contradiction_count=int(d.get("contradiction_count", 0)),
            reasoning=str(d.get("reasoning", "")),
            id=str(d.get("id", _short_id("F"))),
            embedding=d.get("embedding"),
        )


@dataclass
class TestResult:
    support_count: int
    contradiction_count: int
    confidence: float
    support_examples: List[Any] = field(default_factory=list)
    contradiction_examples: List[Any] = field(default_factory=list)
    notes: str = ""


import itertools
_id_counter = itertools.count()

def _short_id(prefix: str) -> str:
    seq = next(_id_counter)
    ts = datetime.utcnow().strftime("%y%m%d%H%M%S%f")
    h = hashlib.sha1(f"{ts}_{seq}".encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{h}"


def _rewrite_between(expr: str) -> str:
    """
    Convert SQL-style BETWEEN into a compound comparison.

        "col BETWEEN a AND b"  ->  "((col) >= a) & ((col) <= b)"

    The AND here is part of the BETWEEN syntax, not a logical operator.
    Must run before any structural rewrite (CASE, if/then, ternary) because
    BETWEEN appears inside conditions.
    """
    pattern = r'(\w+)\s+BETWEEN\s+([^\s]+)\s+AND\s+([^\s]+)'
    def _between_replacer(m: "re.Match") -> str:
        col, lo, hi = m.group(1), m.group(2), m.group(3)
        return f"(({col}) >= {lo}) & (({col}) <= {hi})"
    return re.sub(pattern, _between_replacer, expr, flags=re.IGNORECASE)


def _rewrite_sql_case(expr: str) -> str:
    """
    Convert SQL CASE WHEN into nested np.where() calls.

        "CASE WHEN c1 THEN v1 WHEN c2 THEN v2 ELSE default END"
        -> "np.where(c1, v1, np.where(c2, v2, default))"

    Returns the expression unchanged if it doesn't start with CASE WHEN.
    """
    stripped = expr.strip()
    if not re.match(r'CASE\s+WHEN\s+', stripped, re.IGNORECASE):
        return expr

    # Remove CASE at the start and END at the end
    body = re.sub(r'^CASE\s+', '', stripped, flags=re.IGNORECASE)
    body = re.sub(r'\s+END\s*$', '', body, flags=re.IGNORECASE)

    # Pull out the ELSE clause (everything after the last ELSE)
    else_val = "None"
    else_match = re.search(r'\s+ELSE\s+(.+)$', body, re.IGNORECASE)
    if else_match:
        else_val = else_match.group(1).strip()
        body = body[:else_match.start()]

    # Extract all WHEN condition THEN value pairs
    clauses = re.findall(
        r'WHEN\s+(.+?)\s+THEN\s+(.+?)(?=\s+WHEN|\s*$)',
        body, re.IGNORECASE,
    )
    if not clauses:
        return expr  # couldn't parse, return unchanged

    # Build nested np.where from inside out
    result = else_val
    for cond, val in reversed(clauses):
        result = f"np.where({cond.strip()}, {val.strip()}, {result})"
    return result


def _rewrite_if_then_else(expr: str) -> str:
    """
    Convert pseudo-code if/then/else conditionals into nested np.where() calls.
    LLMs frequently produce this syntax instead of valid Python.

        "if A then B else if C then D else E"
        -> "np.where(A, B, np.where(C, D, E))"

    Returns the expression unchanged if no if/then/else pattern is detected.
    """
    stripped = expr.strip()
    if not re.match(r'if\s+', stripped, re.IGNORECASE):
        return expr

    branches: List[Tuple[str, str]] = []
    rest = stripped

    while True:
        # Match: if <cond> then <value> else <remainder>
        # Non-greedy on cond and value so we stop at the first then / else.
        m = re.match(
            r'if\s+(.*?)\s+then\s+(.*?)\s+else\s+(.*)',
            rest,
            re.IGNORECASE | re.DOTALL,
        )
        if not m:
            break
        branches.append((m.group(1).strip(), m.group(2).strip()))
        rest = m.group(3).strip()
        # If the remainder doesn't start with another 'if', it's the final else value
        if not re.match(r'if\s+', rest, re.IGNORECASE):
            break

    if not branches:
        return expr  # couldn't parse, return unchanged

    # Build nested np.where() from the innermost branch outward
    result = rest  # final else value
    for cond, val in reversed(branches):
        result = f"np.where({cond}, {val}, {result})"
    return result


def _rewrite_cstyle_ternary(expr: str) -> str:
    """
    Convert C/JS-style ternary operators into np.where() calls.
    LLMs sometimes produce this syntax instead of valid Python.

        "(condition ? value_if_true : value_if_false)"
        -> "np.where(condition, value_if_true, value_if_false)"

    Handles parenthesized ternaries appearing anywhere in a larger expression,
    as well as a bare ternary that spans the entire expression.  Iterates to
    replace multiple ternaries in one expression.
    """
    # Pass 1: parenthesized ternaries -- these can appear mid-expression
    # e.g. pattern + '_' + (flag ? 'high' : 'low')
    paren_pattern = r'\(\s*(.+?)\s*\?\s*(.+?)\s*:\s*(.+?)\s*\)'
    prev = None
    while prev != expr:
        prev = expr
        expr = re.sub(paren_pattern, r'np.where(\1, \2, \3)', expr)

    # Pass 2: bare ternary -- the entire expression is cond ? a : b
    # e.g. intensity > 0.8 ? 'bright' : 'dim'
    bare_pattern = r'^(.+?)\s*\?\s*(.+?)\s*:\s*(.+)$'
    m = re.match(bare_pattern, expr.strip())
    if m:
        expr = f"np.where({m.group(1).strip()}, {m.group(2).strip()}, {m.group(3).strip()})"

    return expr


# ==========
# Corpus data
# ==========

class Corpus:
    """
    Holds the dataset (pandas DataFrame) plus metadata and optional transformations.

    Supports:
    - loading from CSV/JSON/JSONL/Excel
    - safe transform spec (derived columns)
    - optional python transform (only if explicitly allowed)
    """

    def __init__(self, df: "pd.DataFrame", domain_name: str, description: str, id_col: Optional[str] = None):
        self.df = df.copy()
        self.domain_name = domain_name
        self.description = description
        self.id_col = id_col or self._infer_id_col()

        if self.id_col not in self.df.columns:
            # create stable ids
            self.df[self.id_col] = list(range(len(self.df)))

    @staticmethod
    def load(path: Union[str, Path], domain_name: str, description: str) -> "Corpus":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        elif suffix == ".json":
            # try JSONL first
            try:
                records = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        records.append(json.loads(line))
                df = pd.DataFrame(records)
            except Exception:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    # unwrap one level if needed
                    if len(data.keys()) == 1 and list(data.keys())[0] not in ["observations", "data"]:
                        data = data[list(data.keys())[0]]
                    if isinstance(data, dict) and "observations" in data and isinstance(data["observations"], list):
                        df = pd.DataFrame(data["observations"])
                    elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                        df = pd.DataFrame(data["data"])
                    else:
                        # try to find any list-valued key
                        picked = None
                        if isinstance(data, dict):
                            for k, v in data.items():
                                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                                    picked = v
                                    break
                        df = pd.DataFrame(picked if picked is not None else [data])
                else:
                    raise ValueError(f"Unsupported JSON format: {type(data)}")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        return Corpus(df=df, domain_name=domain_name, description=description)

    def schema_summary(self, max_columns: int = 40) -> str:
        cols = list(self.df.columns)
        if len(cols) > max_columns:
            cols = cols[:max_columns] + ["..."]
        dtypes = {c: str(self.df[c].dtype) for c in cols if c != "..."}
        return json.dumps({"n_rows": len(self.df), "columns": cols, "dtypes": dtypes}, indent=2)

    def sample_rows_json(self, n: int = 50) -> str:
        n = max(1, min(n, len(self.df)))
        sample = self.df.sample(n=n, random_state=42) if len(self.df) > 1 else self.df
        return sample.to_json(orient="records")

    def get_batches(self, batch_size: int) -> Iterable["pd.DataFrame"]:
        if batch_size <= 0:
            batch_size = len(self.df)
        for i in range(0, len(self.df), batch_size):
            yield self.df.iloc[i:i + batch_size]

    def _infer_id_col(self) -> str:
        for c in ["id", "sample_id", "row_id", "uid"]:
            if c in self.df.columns:
                return c
        return "row_id"

    # -------- transformations --------

    def apply_transform_spec(self, transform_spec: Dict[str, Any]) -> None:
        """
        Apply safe derived-field spec:
        transform_spec = {
          "derived_columns": [
             {"name": "...", "expression": "...", "dtype": "float|int|str|bool|category", "description": "..."}
          ]
        }
        Expressions are evaluated using pandas.eval where possible; fallback to python eval on a safe namespace.
        """
        if not transform_spec:
            return
        derived = transform_spec.get("derived_columns") or []
        if not derived:
            return

        safe_ns = {
            "np": np, "pd": pd, "math": math,
            # Common builtins that LLMs use in expressions -- e.g. .astype(str), int(...), round(...)
            "str": str, "int": int, "float": float, "bool": bool,
            "len": len, "abs": abs, "round": round, "min": min, "max": max,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
        }

        for coldef in derived:
            name = coldef.get("name")
            expr = coldef.get("expression")
            if not name or not expr:
                continue

            # Strip assignment target if expression includes "name = expr" format.
            # LLMs commonly produce this even though `name` is already the target key.
            # We check for = but not == so we don't mangle equality tests.
            stripped = expr.strip()
            if stripped.startswith(name):
                remainder = stripped[len(name):].lstrip()
                if remainder.startswith("=") and not remainder.startswith("=="):
                    expr = remainder[1:].strip()

            # Rewrite non-Python syntax into valid expressions.
            # Order matters: BETWEEN first (appears inside conditions),
            # then structural rewrites (CASE, if/then, ternary).
            expr = _rewrite_between(expr)
            expr = _rewrite_sql_case(expr)
            expr = _rewrite_if_then_else(expr)
            expr = _rewrite_cstyle_ternary(expr)

            # Try pandas.eval first (vectorized), fall back to python eval.
            # If both fail, skip this column with a warning rather than
            # crashing the run -- derived columns are best-effort.
            try:
                self.df[name] = self.df.eval(expr)
            except Exception:
                try:
                    local_ns = {c: self.df[c] for c in self.df.columns}
                    local_ns["df"] = self.df
                    local_ns.update(safe_ns)
                    self.df[name] = eval(expr, {"__builtins__": {}}, local_ns)  # noqa: S307
                except Exception as e:
                    print(f"  Warning: skipping derived column '{name}' -- expression could not be evaluated")
                    print(f"    Expression: {expr}")
                    print(f"    Error: {e}")
                    continue

            dtype = (coldef.get("dtype") or "").lower().strip()
            if dtype:
                try:
                    if dtype in ("int", "int64"):
                        self.df[name] = self.df[name].astype("int64")
                    elif dtype in ("float", "float64"):
                        self.df[name] = self.df[name].astype("float64")
                    elif dtype in ("str", "string"):
                        self.df[name] = self.df[name].astype("string")
                    elif dtype in ("bool", "boolean"):
                        self.df[name] = self.df[name].astype("boolean")
                    elif dtype in ("category",):
                        self.df[name] = self.df[name].astype("category")
                except Exception:
                    pass

    def apply_python_transform(self, python_code: str) -> None:
        """
        Apply a user-reviewed python transform function that must define:
            def transform(df): return df
        This is powerful but unsafe if you execute untrusted code.
        """
        g: Dict[str, Any] = {"pd": pd, "np": np, "math": math}
        l: Dict[str, Any] = {}
        exec(python_code, g, l)  # noqa: S102
        if "transform" not in l or not callable(l["transform"]):
            raise ValueError("python transform must define a callable 'transform(df)'")
        new_df = l["transform"](self.df.copy())
        if not isinstance(new_df, pd.DataFrame):
            raise ValueError("transform(df) must return a pandas DataFrame")
        self.df = new_df


# ==========
# LLM backend
# ==========

JsonDict = Dict[str, Any]
Messages = List[Dict[str, str]]


def _repair_json_string_newlines(text: str) -> str:
    """
    Replace literal newlines/tabs inside JSON string values with their escape
    sequences.  LLMs frequently output multiline strings inside JSON without
    escaping them, which makes json.loads fail.

    Walks the text character by character, tracking whether we're inside a
    quoted string.  Escape sequences (backslash + next char) are passed through
    as-is so we don't double-escape or misread \\\" as a string boundary.
    """
    result: list = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == '\\':
                # Pass through the escape sequence intact -- but if the
                # next char is a literal control character (LLM doing a
                # line continuation like \<newline>), we need to escape
                # it properly rather than passing it through.
                result.append(ch)
                if i + 1 < len(text):
                    i += 1
                    nxt = text[i]
                    if nxt == '\n':
                        result.append('n')       # \<newline> -> \n
                    elif nxt == '\r':
                        result.append('r')       # \<CR>      -> \r
                    elif nxt == '\t':
                        result.append('t')       # \<tab>     -> \t
                    else:
                        result.append(nxt)       # \", \\, \/ etc -- pass through
            elif ch == '"':
                in_string = False
                result.append(ch)
            elif ch == '\n':
                result.append('\\n')
            elif ch == '\r':
                result.append('\\r')
            elif ch == '\t':
                result.append('\\t')
            else:
                result.append(ch)
        else:
            if ch == '"':
                in_string = True
            result.append(ch)
        i += 1
    return ''.join(result)


class LLMClient:
    """
    Thin wrapper around an LLM backend.

    Options:
    - OpenAI-compatible: uses `openai` python package if installed.
    - Custom: user passes `complete_fn(messages, **kwargs) -> str`.
    """

    def __init__(self, config: Dict[str, Any], complete_fn: Optional[Callable[..., str]] = None):
        self.config = config
        self.llm_cfg = config.get("llm", {})
        self.complete_fn = complete_fn

        self.provider = self.llm_cfg.get("provider", "custom")
        self.model = self.llm_cfg.get("model")
        self.temperature = float(self.llm_cfg.get("temperature", 0.2))
        self.max_tokens = int(self.llm_cfg.get("max_tokens", 1200))
        self.base_url = self.llm_cfg.get("base_url")
        self.api_key_env = self.llm_cfg.get("api_key_env", "OPENAI_API_KEY")

        if self.provider == "custom" and self.complete_fn is None:
            raise ValueError("For provider='custom', you must pass complete_fn")

        self._openai_client = None
        if self.provider == "openai":
            self._init_openai()

    def _init_openai(self) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise ImportError(
                "OpenAI provider selected but openai package not installed. "
                "Install with: pip install openai"
            ) from e

        import os
        api_key = os.environ.get(self.api_key_env)
        if not api_key and not self.base_url:
            raise EnvironmentError(
                f"Missing API key env var {self.api_key_env}. "
                "Set it or configure llm.base_url for local OpenAI-compatible servers."
            )

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url

        self._openai_client = OpenAI(**kwargs)

    def complete(self, messages: Messages, **kwargs: Any) -> str:
        if self.provider == "custom":
            return str(self.complete_fn(messages=messages, **kwargs))

        if self.provider == "openai":
            assert self._openai_client is not None
            model = kwargs.get("model", self.model)
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)

            resp = self._openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""

        raise ValueError(f"Unsupported provider: {self.provider}")

    def complete_json(self, messages: Messages, **kwargs: Any) -> JsonDict:
        """
        Robust JSON parse:
        1) Try strict json
        2) Extract first JSON object/array from text
        3) Fall back to ast.literal_eval for python-ish dicts (single quotes)
        """
        text = self.complete(messages, **kwargs)
        if text is None:
            raise ValueError("LLM returned None")
        text = text.strip()
        if not text:
            raise ValueError("LLM returned empty response")

        text = _strip_code_fences(text)

        # 1) strict JSON
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {"_": obj}
        except Exception:
            pass

        # 2) extract first {...} or [...]
        extracted = None
        m_obj = re.search(r"\{.*\}", text, flags=re.DOTALL)
        m_arr = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if m_obj:
            extracted = m_obj.group(0)
        elif m_arr:
            extracted = m_arr.group(0)

        if extracted:
            extracted = extracted.strip()
            try:
                obj = json.loads(extracted)
                return obj if isinstance(obj, dict) else {"_": obj}
            except Exception:
                pass

            # LLMs often put literal newlines inside JSON string values.
            # Repair those and retry json.loads before falling back to literal_eval.
            try:
                repaired = _repair_json_string_newlines(extracted)
                obj = json.loads(repaired)
                return obj if isinstance(obj, dict) else {"_": obj}
            except Exception:
                pass

            # python-ish fallback (single quotes, True/False, None)
            try:
                pyobj = ast.literal_eval(extracted)
                return pyobj if isinstance(pyobj, dict) else {"_": pyobj}
            except Exception:
                pass

        # Truncated JSON recovery -- LLMs often hit max_tokens mid-response,
        # leaving unclosed arrays/objects.  Find the last complete object (last
        # '}') and try closing the structure with progressively more brackets.
        if text.lstrip().startswith("{") or text.lstrip().startswith("["):
            last_brace = text.rfind("}")
            if last_brace > 0:
                truncated = text[:last_brace + 1]
                # Try appending closing sequences up to depth 5
                closers = ["", "]", "}",  "]}", "]}",  "]]}",  "]]}}", "]]}}"]
                for suffix in closers:
                    try:
                        obj = json.loads(truncated + suffix)
                        if obj:  # don't accept empty results
                            print(f"  [complete_json] Recovered truncated JSON (appended '{suffix}')")
                            return obj if isinstance(obj, dict) else {"_": obj}
                    except Exception:
                        continue

        # last resort: try literal_eval on whole text
        try:
            pyobj = ast.literal_eval(text)
            return pyobj if isinstance(pyobj, dict) else {"_": pyobj}
        except Exception as e:
            raise ValueError(f"Could not parse JSON from model output. First 400 chars:\n{text[:400]}") from e



def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


# ===================
# Embeddings & dedup
# ===================

class EmbeddingManager:
    """
    Optional embedding manager. If sentence-transformers is available, it can be used;
    otherwise falls back to a TF-IDF baseline for similarity checks.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._st_model = None
        self._tfidf = None
        self._tfidf_fitted = False

        # prefer sentence-transformers if installed
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self._st_model = None

        if self._st_model is None:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
                self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            except Exception:
                self._tfidf = None

    def embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        if self._st_model is not None:
            vecs = self._st_model.encode(texts, normalize_embeddings=True).tolist()
            return vecs

        if self._tfidf is not None:
            # Fit once on the first batch to establish vocabulary.  All subsequent
            # calls use transform only, so every vector lives in the same feature
            # space and cosine comparisons between batches are valid.
            if not self._tfidf_fitted:
                self._tfidf.fit(texts)
                self._tfidf_fitted = True
            matrix = self._tfidf.transform(texts)
            dense = matrix.toarray()
            norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12
            dense = dense / norms
            return dense.tolist()

        return None

    @staticmethod
    def cosine(a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b:
            return 0.0
        return float(sum(x*y for x, y in zip(a, b)))

    @staticmethod
    def text_similarity(a: str, b: str) -> float:
        # cheap similarity: normalized token overlap (Jaccard)
        ta = set(_normalize_text(a).split())
        tb = set(_normalize_text(b).split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ==================
# Validation & ledger
# ==================

class FormalismValidator:
    """
    Validates formalisms by asking the LLM to test support/contradiction counts on rows.
    """

    def __init__(self, corpus: Corpus, llm: LLMClient, config: Dict[str, Any], prompts: Optional[Dict[str, Any]] = None):
        self.corpus = corpus
        self.llm = llm
        self.config = config
        self.prompts = prompts or PROMPTS["extraction"]

        vcfg = (config.get("phase1", {}) or {}).get("validation", {}) or {}
        self.min_confidence = float(vcfg.get("min_confidence", 0.65))
        self.min_support = int(vcfg.get("min_support", 3))
        self.max_contradictions = int(vcfg.get("max_contradictions", 1))
        self.max_examples = int(vcfg.get("max_examples_per_formalism", 120))
        self.holdout_frac = float(vcfg.get("holdout_frac", 0.2))

    def batch_validate(self, formalisms: List[Formalism]) -> List[Formalism]:
        # simple: validate on a subsample for speed
        df = self.corpus.df
        n = len(df)
        k = min(n, self.max_examples)
        sample = df.sample(n=k, random_state=123) if n > k else df

        rows_json = sample.to_json(orient="records")

        out: List[Formalism] = []
        for f in formalisms:
            messages = [
                {"role": "system", "content": "You are a careful validator. Return strict JSON only."},
                {"role": "user", "content": self.prompts["test_prompt_template"].format(
                    formalism_statement=f.statement,
                    rows=rows_json,
                )}
            ]
            try:
                j = self.llm.complete_json(messages)
                tr = TestResult(
                    support_count=int(j.get("support_count", 0)),
                    contradiction_count=int(j.get("contradiction_count", 0)),
                    confidence=float(j.get("confidence", 0.0)),
                    support_examples=list(j.get("support_examples", []))[:20],
                    contradiction_examples=list(j.get("contradiction_examples", []))[:20],
                    notes=str(j.get("notes", ""))[:500],
                )
                f.support_count = tr.support_count
                f.contradiction_count = tr.contradiction_count
                f.confidence = tr.confidence
                f.reasoning = tr.notes
            except Exception as e:
                f.confidence = 0.0
                f.reasoning = f"validation_error: {e}"
            out.append(f)
        return out

    def meets_acceptance_criteria(self, f: Formalism) -> bool:
        return (
            f.confidence >= self.min_confidence
            and f.support_count >= self.min_support
            and f.contradiction_count <= self.max_contradictions
        )


class FormalismLedger:
    """Tracks all proposed formalisms across rounds and writes JSON artifacts to disk."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rounds: List[Dict[str, Any]] = []
        self.current_round: int = 0
        self.all_formalisms: Dict[str, Formalism] = {}
        self.accepted_ids: set = set()
        self.rejected_ids: set = set()

    def start_round(self, round_number: int) -> None:
        self.current_round = round_number
        self.rounds.append({
            "round": round_number,
            "timestamp": datetime.now().isoformat(),
            "proposed": [],
            "accepted": [],
            "rejected": [],
        })
        print(f"\n  --- Round {round_number} started ---")

    def log_formalism(self, formalism: Formalism, accepted: bool) -> None:
        self.all_formalisms[formalism.id] = formalism
        cur = self.rounds[-1]
        cur["proposed"].append(formalism.id)
        if accepted:
            cur["accepted"].append(formalism.id)
            self.accepted_ids.add(formalism.id)
            print(f"    [+] {formalism.id}  conf={formalism.confidence:.2f}")
        else:
            cur["rejected"].append(formalism.id)
            self.rejected_ids.add(formalism.id)
            print(f"    [-] {formalism.id}  conf={formalism.confidence:.2f}")

    def check_convergence(self, config: Dict[str, Any]) -> bool:
        conv = config.get("convergence", {}) or {}
        max_rounds = int(conv.get("max_rounds", 12))
        plateau_threshold = float(conv.get("plateau_threshold", 0.02))
        plateau_rounds = int(conv.get("plateau_rounds", 3))

        if self.current_round >= max_rounds:
            print(f"  Max rounds ({max_rounds}) reached")
            return True

        if len(self.rounds) < plateau_rounds:
            return False

        recent = self.rounds[-plateau_rounds:]
        rates = [(len(r["accepted"]) / len(r["proposed"])) if r["proposed"] else 0.0 for r in recent]
        if all(rate < plateau_threshold for rate in rates):
            print(f"  Plateau detected: avg acceptance {sum(rates)/len(rates):.1%} over last {plateau_rounds} rounds")
            return True
        return False

    def get_accepted_formalisms(self) -> List[Formalism]:
        return [self.all_formalisms[i] for i in sorted(self.accepted_ids)]

    def save_round(self, round_number: Optional[int] = None) -> None:
        if round_number is None:
            round_number = self.current_round
        rd = self.rounds[round_number - 1]
        (self.output_dir / f"round_{round_number:02d}_summary.json").write_text(json.dumps(rd, indent=2), encoding="utf-8")

        details = {
            "accepted": [self.all_formalisms[i].to_dict() for i in rd["accepted"]],
            "rejected": [self.all_formalisms[i].to_dict() for i in rd["rejected"]],
        }
        (self.output_dir / f"round_{round_number:02d}_formalisms.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
        print(f"  Round {round_number} saved to {self.output_dir}")

    def save_final(self) -> None:
        stats = {
            "total_rounds": len(self.rounds),
            "total_proposed": len(self.all_formalisms),
            "total_accepted": len(self.accepted_ids),
            "total_rejected": len(self.rejected_ids),
            "overall_acceptance_rate": (len(self.accepted_ids)/len(self.all_formalisms)) if self.all_formalisms else 0.0,
        }
        final = {
            "metadata": {"timestamp": datetime.now().isoformat(), **stats},
            "rounds": self.rounds,
            "accepted_formalisms": [self.all_formalisms[i].to_dict() for i in self.accepted_ids],
            "rejected_formalisms": [self.all_formalisms[i].to_dict() for i in self.rejected_ids],
        }
        (self.output_dir / "final_ledger.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
        print(f"\n  Final ledger saved to {self.output_dir / 'final_ledger.json'}")
        print(f"  Accepted: {stats['total_accepted']} / Proposed: {stats['total_proposed']}")
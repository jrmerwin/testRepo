"""
EFF pipelines and orchestrator entry point.

- Phase1Pipeline: LLM extraction + validation + ledger + dedup
- Phase2Pipeline: synthesis -> axioms
- CombinatorialSearcher: non-LLM systematic rule discovery
- ClassicalToolsRunner: decision trees + association rules
- EFFOrchestrator: main entry point

Full flow:
  1. Initializer        -- domain adaptation (prompts, transforms, config overrides)
  2. Phase 1            -- LLM formalism extraction + validation, iterative
  3. Combinatorial      -- systematic non-LLM rule discovery
  4. Classical tools    -- decision trees (reference) + association rules (converted to formalisms)
  5. Merge + dedup      -- combine Phase 1, combinatorial, and classical formalisms
  6. Phase 2            -- LLM axiom synthesis on the merged set
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import json
import re

import pandas as pd

from .eff_core import (
    DEFAULT_CONFIG,
    PROMPTS,
    Corpus,
    EmbeddingManager,
    Formalism,
    FormalismLedger,
    FormalismType,
    FormalismValidator,
    LLMClient,
    deep_merge,
)
from .eff_agents import AxiomWeaver, GodelAgentPool, InitializerAgent, InitializerOutput


# ---------------------------------------------------------------------------
# Shared constants & helpers
# ---------------------------------------------------------------------------

# Columns that are always skipped when selecting features, targets, or
# transactions.  Extended at runtime by the global exclude_columns config key.
_METADATA_COLUMNS = frozenset({
    "id", "sample_id", "row_id", "description", "timestamp", "batch_id", "observer"
})


def _make_run_dir(outputs_dir: Union[str, Path], project_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{project_name}_{ts}"
    out = Path(outputs_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    return out


# Keywords that suggest a column is an outcome/target to predict.
_OUTCOME_KEYWORDS = {
    "result", "outcome", "target", "label", "class", "category",
    "type", "interaction", "stability", "status", "prediction",
    "response", "effect", "decision", "output", "dependent",
}

# Keywords that suggest a column is an input feature.
_FEATURE_KEYWORDS = {
    "color", "size", "pattern", "count", "intensity", "position",
    "rotation", "opacity", "layer", "shape", "primary", "secondary",
    "edge", "input", "feature",
}


def _target_score(col_name: str) -> int:
    """
    Score a column name for how likely it is to be a target/outcome.
    Positive = more likely target.  Negative = more likely feature.
    Splits on underscores/separators and checks each token against the
    keyword sets.
    """
    tokens = set(re.split(r'[_\s\-\.]+', col_name.lower()))
    score = 0
    for t in tokens:
        if t in _OUTCOME_KEYWORDS:
            score += 2
        if t in _FEATURE_KEYWORDS:
            score -= 1
    return score


def _detect_target_columns(df: pd.DataFrame, config: Dict[str, Any], max_targets: int = 4) -> List[str]:
    """
    Heuristic target detection.  First filters to columns with 2-8 unique
    values (categorical-ish).  Then scores each candidate by name: columns
    with outcome-suggestive tokens (result, category, interaction, type,
    stability, ...) rank above columns with feature-suggestive tokens
    (color, size, pattern, opacity, ...).  Among equal scores, later column
    positions are preferred -- outcome columns conventionally appear after
    features in tabular datasets.
    """
    skip = set(_METADATA_COLUMNS)
    skip.update(config.get("exclude_columns", []))

    candidates: List[tuple] = []  # (col_name, position_index)
    for i, c in enumerate(df.columns):
        if c in skip:
            continue
        nunique = df[c].nunique(dropna=True)
        if 2 <= nunique <= 8:
            candidates.append((c, i))

    # Sort by target score desc, then by column position desc as tiebreaker
    candidates.sort(key=lambda x: (_target_score(x[0]), x[1]), reverse=True)
    return [c for c, _ in candidates[:max_targets]]


def _merge_and_dedup(
    phase1: List[Formalism],
    combinatorial: List[Formalism],
    classical: List[Formalism],
    config: Dict[str, Any],
) -> List[Formalism]:
    """
    Merge Phase 1 (LLM-validated), combinatorial, and classical formalisms,
    deduplicating on text similarity.  Priority order: Phase 1 first, then
    combinatorial, then classical.  A lower-priority formalism is only added
    when it has no close match already in the merged set.
    """
    dcfg = (config.get("phase1", {}) or {}).get("dedup", {}) or {}
    text_thr = float(dcfg.get("text_similarity_threshold", 0.93))

    merged: List[Formalism] = list(phase1)
    seen_texts: List[str] = [f.statement for f in merged]

    for f in list(combinatorial) + list(classical):
        is_dup = any(
            EmbeddingManager.text_similarity(f.statement, s) >= text_thr
            for s in seen_texts
        )
        if not is_dup:
            merged.append(f)
            seen_texts.append(f.statement)

    return merged


def _filter_subsumed(formalisms: List[Formalism]) -> List[Formalism]:
    """
    Drop rules that are subsumed by simpler rules with the same consequent.

    "A AND B -> C" is subsumed by "A -> C" when A alone already predicts C
    at equal or better confidence.  These arise from the combinatorial search:
    it generates "pentagon AND size is tiny -> oscillating" alongside "pentagon
    -> oscillating", both validating at conf=1.0 because pentagon alone is
    sufficient.  They're textually different enough to pass dedup but
    informationally redundant.

    Detection: split each statement on " then " to get (antecedent, consequent).
    Group by consequent.  Within each group, sort by antecedent length ascending
    (simpler rules first).  A rule is subsumed if any already-kept simpler rule
    in the same group has its antecedent fully contained in this rule's
    antecedent as a substring, and has >= confidence.

    Keeps all formalisms that don't parse cleanly (no " then " delimiter).
    """
    # Separate into parseable and unparseable
    parsed: List[tuple] = []      # (formalism, antecedent_lower, consequent_lower)
    unparseable: List[Formalism] = []

    for f in formalisms:
        parts = f.statement.split(" then ", 1)
        if len(parts) != 2:
            unparseable.append(f)
            continue
        ant = parts[0].strip().lower()
        # Strip leading "if " for clean substring matching
        if ant.startswith("if "):
            ant = ant[3:]
        cons = parts[1].strip().lower()
        parsed.append((f, ant, cons))

    # Group by consequent
    from collections import defaultdict
    groups: Dict[str, List[tuple]] = defaultdict(list)
    for item in parsed:
        groups[item[2]].append(item)

    kept: List[Formalism] = list(unparseable)
    dropped = 0

    for cons, group in groups.items():
        # Sort by antecedent length ascending -- simpler rules first
        group.sort(key=lambda x: len(x[1]))

        # Walk through; each rule checks against all previously kept rules
        # in this consequent group
        kept_in_group: List[tuple] = []  # (formalism, antecedent)
        for f, ant, _ in group:
            subsumed = False
            for kept_f, kept_ant in kept_in_group:
                # Strict subset: kept antecedent is shorter AND is contained
                # in this rule's antecedent, AND kept rule has >= confidence
                if kept_ant != ant and kept_ant in ant and kept_f.confidence >= f.confidence:
                    subsumed = True
                    break
            if subsumed:
                dropped += 1
            else:
                kept_in_group.append((f, ant))
                kept.append(f)

    if dropped:
        print(f"  Filtered {dropped} subsumed rules (antecedent is superset of a simpler rule with same consequent)")

    return kept


def _compute_search_space_size(df: pd.DataFrame, config: Dict[str, Any]) -> int:
    """
    Analytically compute the total number of (antecedent, consequent) pairs
    the combinatorial searcher evaluates.  Used as the Bonferroni correction
    denominator -- must mirror CombinatorialSearcher's column selection and
    loop structure exactly.

    Structure:
      For each target column t (V_t unique consequent values):
        single_cat:  sum over cat cols c of  nunique(c) * V_t
        single_num:  sum over num cols n of  n_thresholds(n) * 2 * V_t
        cat_pairs:   sum over pairs (c1,c2) of  nunique(c1)*nunique(c2) * V_t
        cat_num:     sum over (c, n) of  nunique(c)*n_thresholds(n)*2 * V_t

    cat_pairs and cat_num are limited to the first max_pair_columns categorical
    columns, matching the searcher.
    """
    ccfg = config.get("combinatorial", {}) or {}
    max_pair_cols = int(ccfg.get("max_pair_columns", 6))
    max_unique = int(ccfg.get("max_unique_values", 20))
    percentiles = list(ccfg.get("numeric_percentiles",
                                [5,10,15,20,25,30,40,50,60,70,75,80,85,90,95]))

    exclude = set(_METADATA_COLUMNS)
    exclude.update(config.get("exclude_columns", []))

    targets = _detect_target_columns(df, config)
    target_set = set(targets)

    # Column selectors -- mirror _categorical_cols / _numeric_cols
    def cat_cols(extra_exclude: set) -> List[str]:
        skip = exclude | extra_exclude
        return [c for c in df.select_dtypes(include=["object","string","category"]).columns
                if c not in skip and df[c].nunique(dropna=True) <= max_unique]

    def num_cols(extra_exclude: set) -> List[str]:
        skip = exclude | extra_exclude
        return [c for c in df.select_dtypes(include=["number"]).columns if c not in skip]

    def n_thresholds(col: str) -> int:
        """Distinct threshold count at configured percentiles, matching _thresholds_for."""
        vals = df[col].dropna()
        if len(vals) == 0:
            return 0
        seen: set = set()
        for p in percentiles:
            seen.add(round(float(vals.quantile(p / 100.0)), 4))
        return len(seen)

    total = 0
    for t in targets:
        v_t = df[t].dropna().nunique()
        cats = cat_cols({t})
        nums = num_cols({t})
        pair_cats = cats[:max_pair_cols]

        # 1) single categorical
        total += sum(df[c].dropna().nunique() for c in cats) * v_t

        # 2) single numeric
        total += sum(n_thresholds(n) * 2 for n in nums) * v_t

        # 3) categorical pairs
        for i in range(len(pair_cats)):
            for j in range(i + 1, len(pair_cats)):
                total += df[pair_cats[i]].dropna().nunique() * df[pair_cats[j]].dropna().nunique() * v_t

        # 4) categorical + numeric
        for c in pair_cats:
            c_unique = df[c].dropna().nunique()
            for n in nums:
                total += c_unique * n_thresholds(n) * 2 * v_t

    return total


def _filter_by_significance(
    axioms: Dict[str, Any],
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Final quality gate: drop axioms whose support is explainable by chance
    given the size of the search space.

    For each axiom:
      1) Parse the consequent.  If it doesn't reference a detected target
         column, drop outright (catches tautologies like shape -> edge_count).
      2) Compute base_rate = P(consequent value) in the full dataset.
         If the specific value can't be extracted, fall back to 1/nunique
         for that target column (conservative -- makes the test harder to pass).
      3) Binomial one-sided test: P(X >= support | n=antecedent_count, p=base_rate)
         where antecedent_count = support + contradictions.
      4) Bonferroni-correct p by the total combinatorial search space size.
      5) Drop if corrected p >= alpha (default 0.05).

    Also prints a summary table so the user can see exactly what was kept
    and why.
    """
    from scipy.stats import binom

    axiom_list = axioms.get("axioms", [])
    if not axiom_list:
        return axioms

    targets = _detect_target_columns(df, config)
    target_set = set(targets)

    # Compute search space for Bonferroni denominator
    search_space = _compute_search_space_size(df, config)

    alpha = float((config.get("significance_filter", {}) or {}).get("alpha", 0.05))

    # Build a lookup: target_col -> {value: count}
    target_value_counts: Dict[str, Dict[str, int]] = {}
    for t in targets:
        vc = df[t].dropna().value_counts()
        target_value_counts[t] = {str(v): int(c) for v, c in vc.items()}

    n_rows = len(df)

    # --- Parse consequent from each axiom statement ---
    # Try to match "target_col is <value>" or "target_col == <value>"
    # in the consequent (everything after " then ").
    def _parse_consequent(statement: str):
        """
        Returns (target_col, target_val_or_None) or (None, None) if
        no target column is mentioned in the consequent.
        """
        parts = statement.split(" then ", 1)
        if len(parts) != 2:
            # Try splitting on " -> " for combinatorial-style statements
            parts = statement.split(" -> ", 1)
        if len(parts) != 2:
            return None, None

        cons = parts[1].strip().rstrip(".")

        # Check each target column -- use word-boundary match so that
        # "shape" does not match inside "shape_complexity".
        import re as _re
        for t in targets:
            if not _re.search(r'(?<![A-Za-z0-9_])' + _re.escape(t.lower()) + r'(?![A-Za-z0-9_])', cons.lower()):
                continue
            # Try to extract the value after "is" or "=="
            # Patterns: "target_col is value", "target_col == value"
            for sep in [" is ", " == ", "=="]:
                idx = cons.lower().find(t.lower() + sep)
                if idx >= 0:
                    val_str = cons[idx + len(t) + len(sep):].strip()
                    # Strip trailing punctuation and quotes
                    val_str = val_str.strip(".'\"")
                    # Take first word/token if multiple words follow
                    # (handles "is repulsive or oscillating" -> just "repulsive")
                    val_str = val_str.split()[0] if val_str else ""
                    return t, val_str if val_str else None
            # Target column mentioned but value not extractable
            return t, None

        return None, None

    kept: List[Dict] = []
    dropped_tautology: List[Dict] = []
    dropped_insignificant: List[Dict] = []

    print(f"\n  --- Significance filter (Bonferroni, search space = {search_space:,}) ---")
    print(f"  {'#':>3}  {'support':>7}  {'base_rate':>9}  {'p_raw':>10}  {'p_corr':>10}  {'verdict':<12}  statement")
    print(f"  {'---':>3}  {'-------':>7}  {'---------':>9}  {'----------':>10}  {'----------':>10}  {'------------':<12}  ---------")

    for ax in axiom_list:
        statement = ax.get("statement", "")
        support = int(ax.get("support_count", 0))
        contradictions = int(ax.get("contradiction_count", 0))
        antecedent_count = support + contradictions

        target_col, target_val = _parse_consequent(statement)

        # --- Drop tautologies: consequent doesn't reference a target column ---
        if target_col is None:
            dropped_tautology.append(ax)
            print(f"  {ax.get('number','?'):>3}  {support:>7}  {'--':>9}  {'--':>10}  {'--':>10}  {'TAUTOLOGY':<12}  {statement}")
            continue

        # --- Compute base rate ---
        if target_val and target_val in target_value_counts.get(target_col, {}):
            base_rate = target_value_counts[target_col][target_val] / n_rows
        else:
            # Fallback: 1/nunique (conservative)
            nunique = len(target_value_counts.get(target_col, {}))
            base_rate = 1.0 / nunique if nunique > 0 else 0.5

        # --- Binomial test ---
        # P(X >= support | n=antecedent_count, p=base_rate)
        # binom.sf(k, n, p) = P(X > k), so sf(support-1) = P(X >= support)
        if antecedent_count == 0 or support == 0:
            p_raw = 1.0
        else:
            p_raw = float(binom.sf(support - 1, antecedent_count, base_rate))

        # --- Bonferroni correction ---
        p_corrected = min(p_raw * search_space, 1.0)

        if p_corrected < alpha:
            kept.append(ax)
            verdict = "KEPT"
        else:
            dropped_insignificant.append(ax)
            verdict = "DROPPED"

        print(f"  {ax.get('number','?'):>3}  {support:>7}  {base_rate:>9.4f}  {p_raw:>10.2e}  {p_corrected:>10.2e}  {verdict:<12}  {statement}")

    print(f"\n  Significance filter result: {len(kept)} kept, "
          f"{len(dropped_tautology)} tautologies dropped, "
          f"{len(dropped_insignificant)} insignificant dropped")

    # Renumber kept axioms
    for i, ax in enumerate(kept, 1):
        ax["number"] = i

    return {"axioms": kept}



def _association_rules_to_formalisms(
    classical_results: Dict[str, Any],
    target_cols: List[str],
    n_rows: int,
) -> List[Formalism]:
    """
    Convert association rules output into Formalism objects.

    Only converts rules whose consequent is a single target column -- this
    filters out reversed rules (e.g. result_category=delta -> pattern=checkered)
    which are just the forward rules restated backwards.

    Statement format matches the rest of the pipeline:
        "If shape is pentagon then interaction_type is oscillating"

    Support and contradiction counts are derived from the association rule
    stats without re-scanning the data:
        support_count  = round(support * n_rows)
        antecedent_count = round(support_count / confidence)
        contradiction_count = antecedent_count - support_count
    """
    rules = classical_results.get("association_rules", {}).get("rules", [])
    if not rules:
        return []

    target_set = set(target_cols)
    formalisms: List[Formalism] = []

    for rule in rules:
        consequents = rule.get("consequents", [])

        # Only take rules with exactly one consequent, and only if that
        # consequent's column is one of the detected targets.
        if len(consequents) != 1:
            continue
        cons_col = consequents[0].split("=")[0]
        if cons_col not in target_set:
            continue

        antecedents = rule.get("antecedents", [])
        confidence = rule["confidence"]
        support = rule["support"]

        # Build statement in the same style as the rest of the pipeline
        ant_parts = [item.replace("=", " is ") for item in antecedents]
        cons_part = consequents[0].replace("=", " is ")
        if len(ant_parts) == 1:
            statement = f"If {ant_parts[0]} then {cons_part}"
        else:
            statement = f"If {' and '.join(ant_parts)} then {cons_part}"

        # Derive counts
        support_count = round(support * n_rows)
        antecedent_count = round(support_count / confidence) if confidence > 0 else support_count
        contradiction_count = max(0, antecedent_count - support_count)

        formalisms.append(Formalism(
            statement=statement,
            confidence=confidence,
            support_count=support_count,
            contradiction_count=contradiction_count,
            source="classical_association_rules",
        ))

    return formalisms


# ---------------------------------------------------------------------------
# Phase 1 -- LLM extraction
# ---------------------------------------------------------------------------

class Phase1Pipeline:
    def __init__(
        self,
        corpus: Corpus,
        config: Dict[str, Any],
        llm: LLMClient,
        output_dir: Path,
        extraction_prompts: Dict[str, str],
    ):
        self.corpus = corpus
        self.config = config
        self.llm = llm
        self.output_dir = Path(output_dir)

        self.agent_pool = GodelAgentPool(config, llm, corpus, extraction_prompts)
        self.validator = FormalismValidator(corpus, llm, config, prompts=extraction_prompts)
        self.embed_mgr = EmbeddingManager(config)
        self.ledger = FormalismLedger(self.output_dir / "formalisms")

        self._seen_texts: List[str] = []
        self._seen_embeddings: List[List[float]] = []

    def execute(self) -> List[Formalism]:
        print("\n" + "=" * 70)
        print("  PHASE 1: FORMALISM EXTRACTION (LLM)")
        print("=" * 70)

        iteration = 0
        while True:
            iteration += 1
            self.ledger.start_round(iteration)

            proposed = self.agent_pool.execute_round(iteration)
            if not proposed:
                print("  No formalisms proposed this round")
                self.ledger.save_round()
                # Do NOT hard-exit here.  Let the convergence detector decide
                # whether to keep trying or stop.  Previously this was a bare
                # break which killed the loop after the first empty round.
                if self.ledger.check_convergence(self.config):
                    break
                continue

            proposed = self._deduplicate(proposed)

            if not proposed:
                print("  All proposed formalisms were duplicates")
                self.ledger.save_round()
                if self.ledger.check_convergence(self.config):
                    break
                continue

            print(f"  Validating {len(proposed)} unique formalisms...")
            validated = self.validator.batch_validate(proposed)

            for f in validated:
                accepted = self.validator.meets_acceptance_criteria(f)
                self.ledger.log_formalism(f, accepted)

            self.ledger.save_round()

            if self.ledger.check_convergence(self.config):
                break

        self.ledger.save_final()
        accepted = self.ledger.get_accepted_formalisms()
        print(f"\n  Phase 1 complete: {len(accepted)} accepted formalisms")
        return accepted

    def _deduplicate(self, new_formalisms: List[Formalism]) -> List[Formalism]:
        dcfg = ((self.config.get("phase1", {}) or {}).get("dedup", {}) or {})
        if not dcfg.get("enabled", True):
            return new_formalisms

        method = dcfg.get("method", "text")
        text_thr = float(dcfg.get("text_similarity_threshold", 0.93))
        emb_thr = float(dcfg.get("embedding_similarity_threshold", 0.92))

        unique: List[Formalism] = []

        if method == "embeddings":
            vecs = self.embed_mgr.embed_texts([f.statement for f in new_formalisms])
            if vecs is None:
                method = "text"  # fallback
            else:
                for f, v in zip(new_formalisms, vecs):
                    f.embedding = v
                    is_dup = False
                    for sv in self._seen_embeddings:
                        if EmbeddingManager.cosine(v, sv) >= emb_thr:
                            is_dup = True
                            break
                    if not is_dup:
                        unique.append(f)
                        self._seen_embeddings.append(v)
                return unique

        # text fallback
        for f in new_formalisms:
            is_dup = False
            for s in self._seen_texts:
                if EmbeddingManager.text_similarity(f.statement, s) >= text_thr:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(f)
                self._seen_texts.append(f.statement)
        return unique


# ---------------------------------------------------------------------------
# Phase 2 -- synthesis
# ---------------------------------------------------------------------------

class Phase2Pipeline:
    def __init__(self, corpus: Corpus, config: Dict[str, Any], llm: LLMClient, output_dir: Path):
        self.corpus = corpus
        self.config = config
        self.llm = llm
        self.output_dir = Path(output_dir)
        self.weaver = AxiomWeaver(config, llm, corpus)

    def execute(self, accepted_formalisms: List[Formalism]) -> Dict[str, Any]:

        if not accepted_formalisms:
            empty: Dict[str, Any] = {"axioms": []}
            out_path = self.output_dir / "axioms.json"
            out_path.write_text(json.dumps(empty, indent=2), encoding="utf-8")
            print("  No accepted formalisms; wrote empty axioms.json")
            return empty

        print("\n" + "=" * 70)
        print("  PHASE 2: AXIOM SYNTHESIS")
        print("=" * 70)

        axioms = self.weaver.weave(accepted_formalisms)

        out_path = self.output_dir / "axioms.json"
        out_path.write_text(json.dumps(axioms, indent=2), encoding="utf-8")
        print(f"  Saved axioms to {out_path}")
        return axioms


# ---------------------------------------------------------------------------
# Combinatorial search -- non-LLM systematic rule discovery
# ---------------------------------------------------------------------------

class CombinatorialSearcher:
    """
    Exhaustively tests single and paired predicates against detected target
    columns, computing support / confidence / contradiction directly from the
    data.  No LLM calls.  Produces Formalism objects ranked by quality.

    Predicate types tested (in order):
      1) Single categorical:        col == value
      2) Single numeric threshold:  col < threshold  /  col >= threshold
      3) Categorical pair:          col1 == v1  AND  col2 == v2
      4) Categorical + numeric:     col_cat == v  AND  col_num < threshold

    For each antecedent, every unique value in every target column is tested
    as a consequent.  Rules that meet min_support and min_confidence are kept.
    Final output is sorted by (confidence desc, support desc) and capped at
    max_formalisms.
    """

    def __init__(self, config: Dict[str, Any], corpus: Corpus):
        self.config = config
        self.corpus = corpus
        self.df = corpus.df

        ccfg = config.get("combinatorial", {}) or {}
        self.min_support = int(ccfg.get("min_support", 5))
        self.min_confidence = float(ccfg.get("min_confidence", 0.80))
        self.numeric_percentiles: List[int] = list(
            ccfg.get("numeric_percentiles", [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95])
        )
        self.max_pair_columns = int(ccfg.get("max_pair_columns", 6))
        self.max_unique_values = int(ccfg.get("max_unique_values", 20))
        self.max_formalisms = int(ccfg.get("max_formalisms", 200))

        # Combined skip set: hard-coded metadata + user-configured exclusions
        self._exclude = set(_METADATA_COLUMNS)
        self._exclude.update(config.get("exclude_columns", []))

    # ---- column selectors --------------------------------------------------

    def _categorical_cols(self, extra_exclude: Optional[set] = None) -> List[str]:
        skip = set(self._exclude)
        if extra_exclude:
            skip.update(extra_exclude)
        return [
            c for c in self.df.select_dtypes(include=["object", "string", "category"]).columns
            if c not in skip and self.df[c].nunique(dropna=True) <= self.max_unique_values
        ]

    def _numeric_cols(self, extra_exclude: Optional[set] = None) -> List[str]:
        skip = set(self._exclude)
        if extra_exclude:
            skip.update(extra_exclude)
        return [c for c in self.df.select_dtypes(include=["number"]).columns if c not in skip]

    # ---- threshold computation ---------------------------------------------

    def _thresholds_for(self, col: str) -> List[float]:
        """
        Distinct threshold values at the configured percentiles for one column,
        rounded to 4 decimal places.  The rounded value is used for both the
        boolean mask and the statement string so they always match.
        """
        values = self.df[col].dropna()
        if len(values) == 0:
            return []
        seen: set = set()
        out: List[float] = []
        for p in self.numeric_percentiles:
            t = round(float(values.quantile(p / 100.0)), 4)
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    # ---- core rule test ----------------------------------------------------

    def _test_rule(
        self,
        antecedent: "pd.Series",
        target_col: str,
        target_val: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate antecedent -> target_col == target_val on self.df.
        Returns a stats dict if both min_support and min_confidence are met,
        otherwise None.
        """
        ant_count = int(antecedent.sum())
        if ant_count == 0:
            return None
        consequent = self.df[target_col] == target_val
        support = int((antecedent & consequent).sum())
        if support < self.min_support:
            return None
        confidence = support / ant_count
        if confidence < self.min_confidence:
            return None
        contradiction = int((antecedent & ~consequent).sum())
        return {
            "support_count": support,
            "contradiction_count": contradiction,
            "confidence": confidence,
        }

    # ---- main search -------------------------------------------------------

    def search(self) -> List[Formalism]:
        """
        Run the full combinatorial search.  Returns formalisms sorted by
        confidence desc then support desc, capped at max_formalisms.
        """
        targets = _detect_target_columns(self.df, self.config)
        if not targets:
            print("  No target columns detected for combinatorial search")
            return []

        print(f"  Targets: {targets}")
        results: List[Formalism] = []

        for target_col in targets:
            target_vals = self.df[target_col].dropna().unique().tolist()
            cat_cols = self._categorical_cols(extra_exclude={target_col})
            num_cols = self._numeric_cols(extra_exclude={target_col})

            # ----------------------------------------------------------
            # 1) Single categorical predicates
            # ----------------------------------------------------------
            for col in cat_cols:
                for val in self.df[col].dropna().unique():
                    mask = self.df[col] == val
                    if mask.sum() < self.min_support:
                        continue
                    for tval in target_vals:
                        r = self._test_rule(mask, target_col, tval)
                        if r:
                            results.append(Formalism(
                                statement=f"{col} == {val} -> {target_col} == {tval}",
                                formalism_type=FormalismType.RULE,
                                confidence=r["confidence"],
                                support_count=r["support_count"],
                                contradiction_count=r["contradiction_count"],
                                reasoning="combinatorial:single_categorical",
                            ))

            # ----------------------------------------------------------
            # 2) Single numeric threshold predicates
            # ----------------------------------------------------------
            for col in num_cols:
                for thresh in self._thresholds_for(col):
                    for op_str, mask in [("<", self.df[col] < thresh),
                                         (">=", self.df[col] >= thresh)]:
                        if mask.sum() < self.min_support:
                            continue
                        for tval in target_vals:
                            r = self._test_rule(mask, target_col, tval)
                            if r:
                                results.append(Formalism(
                                    statement=f"{col} {op_str} {thresh} -> {target_col} == {tval}",
                                    formalism_type=FormalismType.RULE,
                                    confidence=r["confidence"],
                                    support_count=r["support_count"],
                                    contradiction_count=r["contradiction_count"],
                                    reasoning="combinatorial:single_numeric",
                                ))

            # ----------------------------------------------------------
            # 3) Categorical pairs
            # ----------------------------------------------------------
            pair_cats = cat_cols[:self.max_pair_columns]
            for i in range(len(pair_cats)):
                for j in range(i + 1, len(pair_cats)):
                    col1, col2 = pair_cats[i], pair_cats[j]
                    for v1 in self.df[col1].dropna().unique():
                        mask1 = self.df[col1] == v1
                        if mask1.sum() == 0:
                            continue
                        for v2 in self.df[col2].dropna().unique():
                            mask = mask1 & (self.df[col2] == v2)
                            if mask.sum() < self.min_support:
                                continue
                            for tval in target_vals:
                                r = self._test_rule(mask, target_col, tval)
                                if r:
                                    results.append(Formalism(
                                        statement=f"{col1} == {v1} AND {col2} == {v2} -> {target_col} == {tval}",
                                        formalism_type=FormalismType.RULE,
                                        confidence=r["confidence"],
                                        support_count=r["support_count"],
                                        contradiction_count=r["contradiction_count"],
                                        reasoning="combinatorial:categorical_pair",
                                    ))

            # ----------------------------------------------------------
            # 4) Categorical + numeric threshold pairs
            # ----------------------------------------------------------
            for col_cat in pair_cats:
                for v_cat in self.df[col_cat].dropna().unique():
                    mask_cat = self.df[col_cat] == v_cat
                    if mask_cat.sum() == 0:
                        continue
                    for col_num in num_cols:
                        for thresh in self._thresholds_for(col_num):
                            for op_str, mask_num in [("<", self.df[col_num] < thresh),
                                                     (">=", self.df[col_num] >= thresh)]:
                                mask = mask_cat & mask_num
                                if mask.sum() < self.min_support:
                                    continue
                                for tval in target_vals:
                                    r = self._test_rule(mask, target_col, tval)
                                    if r:
                                        results.append(Formalism(
                                            statement=(
                                                f"{col_cat} == {v_cat} AND "
                                                f"{col_num} {op_str} {thresh} -> "
                                                f"{target_col} == {tval}"
                                            ),
                                            formalism_type=FormalismType.RULE,
                                            confidence=r["confidence"],
                                            support_count=r["support_count"],
                                            contradiction_count=r["contradiction_count"],
                                            reasoning="combinatorial:cat_numeric_pair",
                                        ))

        # ----------------------------------------------------------
        # Redundancy filter: drop pair rules that add no predictive
        # value over an existing single-predicate rule.
        #
        # If "A -> target == val" already passes thresholds, then
        # "A AND B -> target == val" is just subsetting A's population
        # in a way that doesn't change the prediction.  Keep it only
        # if neither component predicate alone already covers the same
        # target outcome at equal or better confidence.
        # ----------------------------------------------------------
        single_rules: Dict[str, float] = {}  # "col op val -> target == tval" : confidence
        pair_rules: List[Formalism] = []
        other_rules: List[Formalism] = []

        for f in results:
            if f.reasoning in ("combinatorial:categorical_pair", "combinatorial:cat_numeric_pair"):
                pair_rules.append(f)
            else:
                other_rules.append(f)
                # Index single rules by their consequent for fast lookup.
                # Statement format: "predicate -> target == tval"
                if " -> " in f.statement:
                    consequent = f.statement.split(" -> ", 1)[1]
                    single_rules[consequent] = max(
                        single_rules.get(consequent, 0.0), f.confidence
                    )

        filtered_pairs: List[Formalism] = []
        dropped = 0
        for f in pair_rules:
            if " -> " not in f.statement:
                filtered_pairs.append(f)
                continue
            antecedent, consequent = f.statement.split(" -> ", 1)

            # Split the pair antecedent on " AND " to get individual predicates
            parts = antecedent.split(" AND ")
            if len(parts) != 2:
                filtered_pairs.append(f)
                continue

            # Check if either component alone already has a single rule
            # for this consequent at equal or better confidence
            pred_a = f"{parts[0].strip()} -> {consequent}"
            pred_b = f"{parts[1].strip()} -> {consequent}"
            single_conf_a = single_rules.get(consequent, 0.0)

            # More targeted check: does either individual predicate exist
            # as a single rule for this exact consequent?
            dominated = False
            for single_key, single_conf in single_rules.items():
                if single_key != consequent:
                    continue
                # There's a single rule for this consequent.  Check if
                # either part of our pair matches an existing single rule's
                # antecedent exactly.
                for other_f in other_rules:
                    if other_f.reasoning not in ("combinatorial:single_categorical", "combinatorial:single_numeric"):
                        continue
                    if " -> " not in other_f.statement:
                        continue
                    o_ant, o_cons = other_f.statement.split(" -> ", 1)
                    if o_cons != consequent:
                        continue
                    if o_ant.strip() in (parts[0].strip(), parts[1].strip()):
                        if other_f.confidence >= f.confidence:
                            dominated = True
                            break
                if dominated:
                    break

            if dominated:
                dropped += 1
            else:
                filtered_pairs.append(f)

        results = other_rules + filtered_pairs
        if dropped:
            print(f"  Dropped {dropped} redundant pair rules (subsumed by single-predicate rules)")

        # Rank by quality and cap
        results.sort(key=lambda f: (f.confidence, f.support_count), reverse=True)
        if len(results) > self.max_formalisms:
            print(f"  Capped {len(results)} -> {self.max_formalisms} formalisms")
        return results[:self.max_formalisms]


# ---------------------------------------------------------------------------
# Classical tools -- reference report (not fed into axiom output)
# ---------------------------------------------------------------------------

class ClassicalToolsRunner:
    """
    Decision trees + association rules.  Decision tree results are saved as a
    reference report.  Association rules are also converted to Formalism objects
    and merged into the pipeline input for Phase 2 axiom synthesis.
    """

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.classical_dir = self.output_dir / "classical"
        self.classical_dir.mkdir(parents=True, exist_ok=True)

        # Optional deps
        try:
            from sklearn.tree import DecisionTreeClassifier, export_text  # type: ignore
            from sklearn.preprocessing import LabelEncoder  # type: ignore
            self._sklearn = (DecisionTreeClassifier, export_text, LabelEncoder)
            self.sklearn_available = True
        except Exception:
            self._sklearn = None
            self.sklearn_available = False

        try:
            from mlxtend.frequent_patterns import apriori, association_rules  # type: ignore
            from mlxtend.preprocessing import TransactionEncoder  # type: ignore
            self._mlxtend = (apriori, association_rules, TransactionEncoder)
            self.mlxtend_available = True
        except Exception:
            self._mlxtend = None
            self.mlxtend_available = False

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        results: Dict[str, Any] = {"available": {}, "decision_trees": {}, "association_rules": {}}

        results["available"]["sklearn"] = self.sklearn_available
        results["available"]["mlxtend"] = self.mlxtend_available

        # Use shared target detection so the same exclusions apply everywhere
        ccfg = self.config.get("classical", {}) or {}
        target_cols = ccfg.get("target_columns")
        if target_cols is None:
            target_cols = _detect_target_columns(df, self.config)

        if self.sklearn_available:
            for tgt in target_cols:
                try:
                    results["decision_trees"][tgt] = self._decision_tree_rules(df, tgt)
                except Exception as e:
                    results["decision_trees"][tgt] = {"error": str(e)}
        else:
            results["decision_trees"]["error"] = "sklearn not installed"

        if ccfg.get("association_rules", {}).get("enabled", True):
            if self.mlxtend_available:
                try:
                    results["association_rules"] = self._association_rules(df)
                except Exception as e:
                    results["association_rules"] = {"error": str(e)}
            else:
                results["association_rules"] = {"error": "mlxtend not installed"}

        out_path = self.classical_dir / "classical_results.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"  Saved classical tools results to {out_path}")
        return results

    def _decision_tree_rules(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        DecisionTreeClassifier, export_text, LabelEncoder = self._sklearn  # type: ignore

        ccfg = (self.config.get("classical", {}) or {}).get("decision_tree", {}) or {}
        max_depth = int(ccfg.get("max_depth", 5))
        min_samples_leaf = int(ccfg.get("min_samples_leaf", 15))
        min_samples_split = int(ccfg.get("min_samples_split", 30))

        # Exclude metadata + global exclusions + the target itself
        exclude_cols = set(_METADATA_COLUMNS)
        exclude_cols.update(self.config.get("exclude_columns", []))
        exclude_cols.add(target_col)

        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        encoders: Dict[str, Any] = {}
        for col in X.columns:
            if str(X[col].dtype) in ("object", "string") or X[col].dtype.name == "category":
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].fillna("missing").astype(str))
                encoders[col] = {"classes": le.classes_.tolist()}
            elif X[col].dtype == "bool":
                X[col] = X[col].astype(int)

        if str(y.dtype) in ("object", "string") or y.dtype.name == "category":
            le_t = LabelEncoder()
            y_enc = le_t.fit_transform(y.fillna("missing").astype(str))
            target_classes = le_t.classes_.tolist()
        else:
            y_enc = y
            target_classes = None

        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=42,
        )
        tree.fit(X, y_enc)

        rules = export_text(tree, feature_names=feature_cols)
        return {
            "target": target_col,
            "n_rows": len(df),
            "n_features": len(feature_cols),
            "accuracy_in_sample": float(tree.score(X, y_enc)),
            "depth": int(tree.get_depth()),
            "n_leaves": int(tree.get_n_leaves()),
            "rules_text": rules,
            "top_feature_importances": sorted(
                [{"feature": f, "importance": float(i)} for f, i in zip(feature_cols, tree.feature_importances_)],
                key=lambda d: d["importance"],
                reverse=True,
            )[:10],
            "encoders": encoders,
            "target_classes": target_classes,
        }

    def _association_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        apriori, association_rules, TransactionEncoder = self._mlxtend  # type: ignore
        acfg = (self.config.get("classical", {}) or {}).get("association_rules", {}) or {}
        min_support = float(acfg.get("min_support", 0.08))
        min_confidence = float(acfg.get("min_confidence", 0.55))
        max_rules = int(acfg.get("max_rules", 200))

        # Exclude metadata + global exclusions
        skip = set(_METADATA_COLUMNS)
        skip.update(self.config.get("exclude_columns", []))

        cat_cols = [
            c for c in df.select_dtypes(include=["object", "string", "category"]).columns
            if c not in skip
        ]

        transactions: List[List[str]] = []
        for _, row in df.iterrows():
            items = []
            for col in cat_cols:
                val = row[col]
                if pd.notna(val) and str(val).strip() != "" and str(val).lower() != "none":
                    items.append(f"{col}={val}")
            if items:
                transactions.append(items)

        if len(transactions) < 5:
            return {"warning": "not enough categorical transactions for association rules", "n_transactions": len(transactions)}

        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        basket = pd.DataFrame(te_ary, columns=te.columns_)

        freq = apriori(basket, min_support=min_support, use_colnames=True)
        if freq.empty:
            return {"warning": "no frequent itemsets found", "min_support": min_support}

        rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
        if rules.empty:
            return {"warning": "no rules found", "min_confidence": min_confidence}

        rules = rules.sort_values(["confidence", "lift"], ascending=False).head(max_rules)

        # serialize frozensets
        out = []
        for _, r in rules.iterrows():
            out.append({
                "antecedents": sorted(list(r["antecedents"])),
                "consequents": sorted(list(r["consequents"])),
                "support": float(r["support"]),
                "confidence": float(r["confidence"]),
                "lift": float(r["lift"]),
            })
        return {"n_rules": len(out), "rules": out}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class EFFOrchestrator:
    """
    Main EFF entry point.  See module docstring for the full pipeline flow.
    """

    def __init__(self, config_override: Optional[Dict[str, Any]] = None, llm_complete_fn=None):
        self.config = deep_merge(DEFAULT_CONFIG, config_override)
        self.llm = (
            LLMClient(self.config, complete_fn=llm_complete_fn)
            if self.config["llm"]["provider"] == "custom"
            else LLMClient(self.config)
        )

        self.output_dir: Optional[Path] = None
        self.corpus: Optional[Corpus] = None

        # initializer outputs
        self.initializer_output: Optional[InitializerOutput] = None
        self.extraction_prompts: Dict[str, str] = dict(PROMPTS["extraction"])

    def run(
        self,
        data_path: Union[str, Path],
        domain_name: str,
        description: str,
        goals: str = "Discover testable causal/structural regularities and decision dynamics.",
    ) -> Dict[str, Any]:
        # ----------------------------------------------------------------
        # Setup
        # ----------------------------------------------------------------
        self.output_dir = _make_run_dir(self.config["outputs_dir"], self.config["project_name"])
        (self.output_dir / "config.json").write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        print(f"\nRun dir: {self.output_dir}")

        # load corpus
        self.corpus = Corpus.load(data_path, domain_name=domain_name, description=description)
        (self.output_dir / "data_schema.json").write_text(self.corpus.schema_summary(), encoding="utf-8")

        # ----------------------------------------------------------------
        # Initializer -- domain adaptation
        # ----------------------------------------------------------------
        init_cfg = self.config.get("initializer", {}) or {}
        if init_cfg.get("enabled", True):
            print("\n" + "=" * 70)
            print("  INITIALIZER: DOMAIN ADAPTATION")
            print("=" * 70)
            init_agent = InitializerAgent(self.config, self.llm, self.corpus)

            try:
                self.initializer_output = init_agent.run(goals=goals)
            except Exception:
                import traceback
                err_path = self.output_dir / "initializer_error.txt"
                err_path.write_text(traceback.format_exc(), encoding="utf-8")
                print(f"  Initializer crashed. Traceback saved to: {err_path}")
                raise

            # merge config overrides
            suggested = self.initializer_output.suggested_config_overrides or {}
            (self.output_dir / "initializer_suggested_config.json").write_text(
                json.dumps(suggested, indent=2), encoding="utf-8"
            )
            self.config = deep_merge(self.config, suggested)
            (self.output_dir / "config_after_initializer.json").write_text(
                json.dumps(self.config, indent=2), encoding="utf-8"
            )

            # prompt overrides
            self.extraction_prompts = dict(self.extraction_prompts)
            self.extraction_prompts["observation_prompt_template"] = self.initializer_output.observation_prompt_template
            self.extraction_prompts["test_prompt_template"] = self.initializer_output.test_prompt_template
            (self.output_dir / "extraction_prompts.json").write_text(
                json.dumps(self.extraction_prompts, indent=2), encoding="utf-8"
            )

            # transforms
            if self.initializer_output.transform_spec:
                (self.output_dir / "transform_spec.json").write_text(
                    json.dumps(self.initializer_output.transform_spec, indent=2), encoding="utf-8"
                )
                self.corpus.apply_transform_spec(self.initializer_output.transform_spec)

            # optional python transform (guarded)
            pycode = self.initializer_output.optional_python_transform
            if pycode:
                (self.output_dir / "optional_python_transform.py").write_text(pycode, encoding="utf-8")
                if init_cfg.get("accept_generated_python_transform", False):
                    print("  Applying generated python transform (enabled by config)...")
                    self.corpus.apply_python_transform(pycode)
                else:
                    print("  Generated python transform was produced but NOT executed "
                          "(set initializer.accept_generated_python_transform=True to allow).")

        # save transformed snapshot
        (self.output_dir / "data_preview.csv").write_text(
            self.corpus.df.head(50).to_csv(index=False), encoding="utf-8"
        )

        # ----------------------------------------------------------------
        # Phase 1 -- LLM extraction
        # ----------------------------------------------------------------
        phase1 = Phase1Pipeline(
            self.corpus, self.config, self.llm, self.output_dir,
            extraction_prompts=self.extraction_prompts,
        )
        phase1_accepted = phase1.execute()

        # ----------------------------------------------------------------
        # Combinatorial search -- non-LLM rule discovery
        # ----------------------------------------------------------------
        combinatorial_formalisms: List[Formalism] = []
        if (self.config.get("combinatorial", {}) or {}).get("enabled", True):
            print("\n" + "=" * 70)
            print("  COMBINATORIAL SEARCH: SYSTEMATIC RULE DISCOVERY")
            print("=" * 70)
            searcher = CombinatorialSearcher(self.config, self.corpus)
            combinatorial_formalisms = searcher.search()
            print(f"  Found {len(combinatorial_formalisms)} rules passing thresholds")
            (self.output_dir / "combinatorial_formalisms.json").write_text(
                json.dumps([f.to_dict() for f in combinatorial_formalisms], indent=2),
                encoding="utf-8",
            )

        # ----------------------------------------------------------------
        # Classical tools -- association rules + decision trees
        # ----------------------------------------------------------------
        classical_results = None
        classical_formalisms: List[Formalism] = []
        if (self.config.get("classical", {}) or {}).get("enabled", True):
            print("\n" + "=" * 70)
            print("  CLASSICAL TOOLS: ASSOCIATION RULES + DECISION TREES")
            print("=" * 70)
            classical = ClassicalToolsRunner(self.config, self.output_dir)
            classical_results = classical.run(self.corpus.df)

            # Convert association rules to formalisms for inclusion in merge
            target_cols = _detect_target_columns(self.corpus.df, self.config)
            classical_formalisms = _association_rules_to_formalisms(
                classical_results, target_cols, len(self.corpus.df)
            )
            if classical_formalisms:
                print(f"  Converted {len(classical_formalisms)} association rules to formalisms")
                (self.output_dir / "classical_formalisms.json").write_text(
                    json.dumps([f.to_dict() for f in classical_formalisms], indent=2),
                    encoding="utf-8",
                )

        # ----------------------------------------------------------------
        # Merge Phase 1 + combinatorial + classical, dedup
        # ----------------------------------------------------------------
        all_formalisms = _merge_and_dedup(
            phase1_accepted, combinatorial_formalisms, classical_formalisms, self.config
        )
        print(
            f"\n  Merged input to Phase 2: "
            f"{len(phase1_accepted)} (Phase 1) + "
            f"{len(combinatorial_formalisms)} (combinatorial) + "
            f"{len(classical_formalisms)} (classical) -> "
            f"{len(all_formalisms)} after dedup"
        )

        # Drop rules subsumed by simpler rules with the same consequent
        # (e.g. "pentagon AND size is tiny -> oscillating" when "pentagon ->
        # oscillating" already exists at equal confidence).
        all_formalisms = _filter_subsumed(all_formalisms)
        print(f"  After subsumption filter: {len(all_formalisms)}")

        (self.output_dir / "merged_formalisms.json").write_text(
            json.dumps([f.to_dict() for f in all_formalisms], indent=2),
            encoding="utf-8",
        )

        # ----------------------------------------------------------------
        # Phase 2 -- axiom synthesis on the merged set
        # ----------------------------------------------------------------
        phase2 = Phase2Pipeline(self.corpus, self.config, self.llm, self.output_dir)
        axioms = phase2.execute(all_formalisms)

        # ----------------------------------------------------------------
        # Final quality gate -- significance filter
        # ----------------------------------------------------------------
        axioms = _filter_by_significance(axioms, self.corpus.df, self.config)
        # Overwrite axioms.json with the filtered set
        (self.output_dir / "axioms.json").write_text(
            json.dumps(axioms, indent=2), encoding="utf-8"
        )

        # ----------------------------------------------------------------
        # Final output
        # ----------------------------------------------------------------
        result: Dict[str, Any] = {
            "output_dir": str(self.output_dir),
            "phase1_accepted": [f.to_dict() for f in phase1_accepted],
            "combinatorial_formalisms": [f.to_dict() for f in combinatorial_formalisms],
            "merged_formalisms": [f.to_dict() for f in all_formalisms],
            "axioms": axioms,
            "classical": classical_results,
        }
        (self.output_dir / "run_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

        print(f"\n  Run complete. Output dir: {self.output_dir}")
        print(f"  Axioms: {len(axioms.get('axioms', []))}")
        return result
"""
EFF agents: LLM-mediated reasoning components.

- BaseAgent: convenience wrapper
- InitializerAgent: dataset-aware prompt + transform + config generation
- GodelAgent(+Pool): propose candidate formalisms from batches
- AxiomWeaver: synthesize validated formalisms into axioms
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import json

from eff_core import Corpus, Formalism, FormalismType, LLMClient, PROMPTS, deep_merge


class BaseAgent:
    def __init__(self, config: Dict[str, Any], llm: LLMClient, corpus: Corpus):
        self.config = config
        self.llm = llm
        self.corpus = corpus

    def _json(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        return self.llm.complete_json(messages, **kwargs)


@dataclass
class InitializerOutput:
    observation_prompt_template: str
    test_prompt_template: str
    transform_spec: Dict[str, Any]
    suggested_config_overrides: Dict[str, Any]
    optional_python_transform: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None


class InitializerAgent(BaseAgent):
    """
    Two-call initializer:
    1) Analyze schema + sample rows + goals
    2) Generate extraction prompts + transform spec + config overrides
    """

    def run(self, goals: str) -> InitializerOutput:
        icfg = self.config.get("initializer", {}) or {}
        sample_rows = int(icfg.get("sample_rows", 50))
        max_cols = int(icfg.get("max_columns_in_prompt", 40))

        # Strip excluded columns (ground truth, metadata) from what we send
        # to the LLM.  The initializer must not see these or it will reference
        # them in the prompts it generates.
        exclude = set(self.config.get("exclude_columns", []))
        visible_df = self.corpus.df.drop(columns=[c for c in exclude if c in self.corpus.df.columns])
        visible_cols = list(visible_df.columns)
        if len(visible_cols) > max_cols:
            visible_cols = visible_cols[:max_cols]
        schema = json.dumps({
            "n_rows": len(visible_df),
            "columns": visible_cols,
            "dtypes": {c: str(visible_df[c].dtype) for c in visible_cols},
        }, indent=2)
        n = max(1, min(sample_rows, len(visible_df)))
        sample_df = visible_df.sample(n=n, random_state=42) if len(visible_df) > 1 else visible_df
        sample = sample_df.to_json(orient="records")

        # Call 1: analysis
        p = PROMPTS["initializer"]
        messages1 = [
            {"role": "system", "content": p["analysis_system"]},
            {"role": "user", "content": p["analysis_user"].format(
                domain_name=self.corpus.domain_name,
                domain_description=self.corpus.description,
                goals=goals,
                schema=schema,
                sample_rows=sample,
            )},
        ]
        analysis = self._json(messages1)

        # Call 2: generation
        messages2 = [
            {"role": "system", "content": p["generation_system"]},
            {"role": "user", "content": p["generation_user"].format(
                analysis_json=json.dumps(analysis, indent=2),
                schema=schema,
            )},
        ]
        gen = self._json(messages2)

        # Validate returned templates.  The LLM often drops or renames the
        # required placeholders.  If they're missing, fall back to the defaults
        # rather than silently producing a broken prompt.
        obs_tmpl = str(gen.get("observation_prompt_template") or "")
        if "{batch_rows}" not in obs_tmpl:
            print("  [initializer] observation_prompt_template missing {batch_rows} placeholder -- falling back to default")
            obs_tmpl = PROMPTS["extraction"]["observation_prompt_template"]

        test_tmpl = str(gen.get("test_prompt_template") or "")
        if "{formalism_statement}" not in test_tmpl or "{rows}" not in test_tmpl:
            print("  [initializer] test_prompt_template missing required placeholders -- falling back to default")
            test_tmpl = PROMPTS["extraction"]["test_prompt_template"]

        return InitializerOutput(
            observation_prompt_template=obs_tmpl,
            test_prompt_template=test_tmpl,
            transform_spec=dict(gen.get("transform_spec") or {}),
            suggested_config_overrides=dict(gen.get("suggested_config_overrides") or {}),
            optional_python_transform=(gen.get("optional_python_transform") or None),
            analysis=analysis,
        )


class GodelAgent(BaseAgent):
    """
    Proposes candidate formalisms from a batch of rows using the extraction observation prompt.
    """

    def __init__(self, config: Dict[str, Any], llm: LLMClient, corpus: Corpus, extraction_prompts: Dict[str, str]):
        super().__init__(config, llm, corpus)
        self.prompts = extraction_prompts

    def propose(self, batch_df) -> List[Formalism]:
        # .get() takes exactly 2 args (key, default).  The JSON-strictness
        # instruction is appended separately so it always appears regardless
        # of which system prompt wins.
        p1 = self.prompts.get(
            "system_base", PROMPTS["extraction"]["system_base"]
        ) + "\nReturn STRICT JSON only. Use double quotes for all keys/strings. No trailing commas. No markdown."

        tmpl = self.prompts.get("observation_prompt_template", PROMPTS["extraction"]["observation_prompt_template"])

        batch_rows = batch_df.to_json(orient="records")
        messages = [
            {"role": "system", "content": p1},
            {"role": "user", "content": tmpl.format(
                domain_name=self.corpus.domain_name,
                domain_description=self.corpus.description,
                batch_rows=batch_rows,
            )},
        ]
        j = self._json(messages)

        out: List[Formalism] = []
        for item in (j.get("formalisms") or []):
            stmt = (item.get("statement") if isinstance(item, dict) else str(item)).strip()
            if not stmt:
                continue
            out.append(Formalism(statement=stmt, formalism_type=FormalismType.RULE))
        return out


class GodelAgentPool:
    def __init__(self, config: Dict[str, Any], llm: LLMClient, corpus: Corpus, extraction_prompts: Dict[str, str]):
        self.config = config
        self.llm = llm
        self.corpus = corpus
        self.extraction_prompts = extraction_prompts

        n_agents = int((config.get("phase1", {}) or {}).get("n_agents", 3))
        self.agents = [GodelAgent(config, llm, corpus, extraction_prompts) for _ in range(max(1, n_agents))]

    def execute_round(self, iteration: int) -> List[Formalism]:
        p1 = self.config.get("phase1", {}) or {}
        batch_size = int(p1.get("batch_size", 64))
        max_per_round = int(p1.get("max_formalisms_per_round", 12))

        formalisms: List[Formalism] = []
        for bidx, batch in enumerate(self.corpus.get_batches(batch_size)):
            agent = self.agents[bidx % len(self.agents)]
            try:
                proposed = agent.propose(batch)
            except Exception as e:
                print(f"    agent error on batch {bidx}: {e}")
                continue
            formalisms.extend(proposed)
            if len(formalisms) >= max_per_round:
                break
        return formalisms[:max_per_round]


class AxiomWeaver(BaseAgent):
    """
    Takes accepted formalisms and synthesizes them into axioms (JSON list).
    """

    def weave(self, accepted_formalisms: List[Formalism]) -> Dict[str, Any]:
        p = PROMPTS["synthesis"]
        p2cfg = self.config.get("phase2", {}) or {}
        max_tokens = int(p2cfg.get("max_tokens", 4000))

        formalisms_json = json.dumps([f.to_dict() for f in accepted_formalisms], indent=2)

        weave_prompt = p["weave_prompt"].replace("{formalisms_json}", formalisms_json)

        messages = [
            {"role": "system", "content": p["system_base"]},
            {"role": "user", "content": weave_prompt},
        ]
        candidate = self._json(messages, max_tokens=max_tokens)

        moderation_prompt = p["moderation_prompt"].replace(
            "{axioms_json}", json.dumps(candidate, indent=2)
        )

        messages2 = [
            {"role": "system", "content": p["system_base"]},
            {"role": "user", "content": moderation_prompt},
        ]
        moderated = self._json(messages2, max_tokens=max_tokens)

        return moderated
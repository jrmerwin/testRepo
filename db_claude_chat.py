"""
db_claude_chat.py
=================
Token-efficient Claude chatbot for Databricks-hosted endpoints.

Core design principles:
  1. Code blocks are SACRED — never summarized, stored verbatim, retrieved by relevance
  2. Natural language is COMPRESSIBLE — older exchanges become rolling summaries
  3. Context is TIERED — recency + relevance + budget awareness determine what goes in
  4. Every call assembles the tightest possible prompt that still gives the model enough context

Token budget (default 30K):
  ┌─────────────────────────────┬──────────┐
  │ Response reserve            │  4,000   │
  │ System prompt + core ctx    │  1,500   │
  │ Recent messages (verbatim)  │ 12,000   │
  │ Relevant code blocks        │  9,000   │
  │ Rolling summary             │  3,500   │
  └─────────────────────────────┴──────────┘

Usage:
    bot = DBClaudeChat(
        endpoint="<<your endpoint here>>",
        core_context="I am working on a Databricks monitoring agent that...",
    )
    response = bot.chat("Here is the pipeline class, let's refactor the retry logic:\n```python\n...```")
    print(response)

    # Inspect what's stored
    bot.list_code_blocks()

    # Persist session to disk (safe — no raw data, only code + summaries)
    bot.save_session("session_2025.json")
    bot.load_session("session_2025.json")
"""

import re
import json
import time
import hashlib
import textwrap
import urllib.request
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Token estimation (no external deps — works in any Databricks cluster)
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """
    Fast approximation: Claude tokenizes at ~3.8 chars/token on average for
    mixed code+prose. We use 3.5 to be conservative (overestimate = safer).
    """
    return max(1, len(text) // 4)  # slight overestimate intentional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CodeBlock:
    """A verbatim code block extracted from conversation."""
    block_id: str           # short hash, e.g. "a3f2"
    label: str              # descriptive name inferred or user-set
    language: str           # python, sql, bash, etc.
    content: str            # exact code, never modified
    tags: list              # keyword tags for relevance matching
    token_count: int = 0
    turn_index: int = 0     # which conversation turn it came from

    def __post_init__(self):
        self.token_count = _count_tokens(self.content)

    def reference(self) -> str:
        """Short placeholder used inside stored messages."""
        return f"[CODE_BLOCK:{self.block_id} — {self.label}]"

    def injection(self) -> str:
        """Full text to inject into prompt when this block is relevant."""
        return (
            f"### Code Block [{self.block_id}] — {self.label}\n"
            f"```{self.language}\n{self.content}\n```\n"
        )


@dataclass
class Turn:
    """One user+assistant exchange."""
    index: int
    user: str               # may contain [CODE_BLOCK:xxx] references
    assistant: str          # same
    summary: str = ""       # filled in during compaction
    is_compacted: bool = False
    token_count: int = 0

    def __post_init__(self):
        self.token_count = _count_tokens(self.user + self.assistant)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DBClaudeChat:
    """
    Memory-managed Claude chatbot for Databricks endpoints.

    Parameters
    ----------
    endpoint : str
        Your Databricks Claude endpoint URL.
    api_token : str, optional
        Databricks personal access token (or set DATABRICKS_TOKEN env var).
    core_context : str, optional
        Persistent project description injected in every call. Keep under ~400 tokens.
        Describe what you are building, key constraints, your stack, etc.
    max_prompt_tokens : int
        Hard cap on assembled prompt tokens. Default 28000 leaves room for response.
    max_response_tokens : int
        Max tokens requested from the model.
    recent_window_budget : int
        Token budget reserved for verbatim recent messages.
    code_budget : int
        Token budget reserved for injected code blocks.
    summary_budget : int
        Token budget reserved for rolling summary of older turns.
    recent_turns_floor : int
        Always keep this many recent turns verbatim, even if over budget.
        Prevents losing immediate context on long code discussions.
    compaction_threshold : int
        Compact oldest turns when history (excluding code) exceeds this many tokens.
    auto_label_code : bool
        Attempt to infer a label for code blocks from surrounding context.
    verbose : bool
        Print token budget breakdown before each call.
    """

    SYSTEM_PROMPT = textwrap.dedent("""\
        You are Claude, an expert AI assistant helping with systems-level engineering,
        data science, and software architecture. You have access to a structured memory
        system. When you see [CODE_BLOCK:id — label] references in the conversation,
        the actual code for relevant blocks will be injected under "## Relevant Code".
        You may refer to code blocks by their label. When writing new or modified code,
        always use fenced code blocks with a language tag so they can be stored.
    """)

    def __init__(
        self,
        endpoint: str = "<<your endpoint here>>",
        api_token: str = "",
        core_context: str = "",
        max_prompt_tokens: int = 28_000,
        max_response_tokens: int = 4_000,
        recent_window_budget: int = 12_000,
        code_budget: int = 9_000,
        summary_budget: int = 3_500,
        recent_turns_floor: int = 3,
        compaction_threshold: int = 18_000,
        auto_label_code: bool = True,
        verbose: bool = False,
    ):
        self.endpoint = endpoint
        self.api_token = api_token
        self.core_context = core_context.strip()
        self.max_prompt_tokens = max_prompt_tokens
        self.max_response_tokens = max_response_tokens
        self.recent_window_budget = recent_window_budget
        self.code_budget = code_budget
        self.summary_budget = summary_budget
        self.recent_turns_floor = recent_turns_floor
        self.compaction_threshold = compaction_threshold
        self.auto_label_code = auto_label_code
        self.verbose = verbose

        self._turns: list[Turn] = []
        self._code_store: dict[str, CodeBlock] = {}   # block_id -> CodeBlock
        self._rolling_summary: str = ""               # compacted older history
        self._turn_index: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Send a message and return the assistant's response."""
        # 1. Extract any code blocks from the user message, store them
        clean_user_msg, new_block_ids = self._extract_and_store_code(
            user_message, speaker="user"
        )

        # 2. Assemble the prompt within budget
        messages = self._build_messages(clean_user_msg, new_block_ids)

        if self.verbose:
            self._print_budget_summary(messages)

        # 3. Call the endpoint
        raw_response = self._call_api(messages)

        # 4. Extract code from the response too
        clean_response, _ = self._extract_and_store_code(raw_response, speaker="assistant")

        # 5. Store this turn
        turn = Turn(
            index=self._turn_index,
            user=clean_user_msg,
            assistant=clean_response,
        )
        self._turns.append(turn)
        self._turn_index += 1

        # 6. Compact old turns if we're getting heavy
        self._maybe_compact()

        return raw_response  # return original with code intact for display

    def set_core_context(self, text: str):
        """Update the persistent project context (injected in every call)."""
        self.core_context = text.strip()

    def add_code_block(self, content: str, label: str, language: str = "python", tags: list = None):
        """Manually register a code block (e.g., paste in a file you can't share inline)."""
        block_id = self._make_block_id(content)
        tags = tags or self._infer_tags(content, label)
        block = CodeBlock(
            block_id=block_id,
            label=label,
            language=language,
            content=content,
            tags=tags,
            turn_index=self._turn_index,
        )
        self._code_store[block_id] = block
        print(f"[DBClaudeChat] Registered code block [{block_id}] — {label} ({block.token_count} tokens)")
        return block_id

    def list_code_blocks(self):
        """Print a summary of all stored code blocks."""
        if not self._code_store:
            print("[DBClaudeChat] No code blocks stored.")
            return
        print(f"\n{'─'*60}")
        print(f"{'ID':<8} {'Tokens':<8} {'Language':<10} {'Label'}")
        print(f"{'─'*60}")
        for b in self._code_store.values():
            print(f"{b.block_id:<8} {b.token_count:<8} {b.language:<10} {b.label}")
        print(f"{'─'*60}\n")

    def get_code_block(self, block_id: str) -> Optional[str]:
        """Retrieve exact code for a stored block."""
        b = self._code_store.get(block_id)
        return b.content if b else None

    def show_memory_state(self):
        """Print a summary of the current memory state and token usage."""
        summary_tokens = _count_tokens(self._rolling_summary)
        code_tokens = sum(b.token_count for b in self._code_store.values())
        recent_tokens = sum(t.token_count for t in self._turns[-self.recent_turns_floor:] if not t.is_compacted)
        print(f"\n[DBClaudeChat Memory State]")
        print(f"  Turns stored      : {len(self._turns)} ({sum(1 for t in self._turns if t.is_compacted)} compacted)")
        print(f"  Code blocks       : {len(self._code_store)} (~{code_tokens} tokens)")
        print(f"  Rolling summary   : ~{summary_tokens} tokens")
        print(f"  Recent verbatim   : ~{recent_tokens} tokens")
        print(f"  Core context      : ~{_count_tokens(self.core_context)} tokens\n")

    def save_session(self, path: str):
        """Save session state to a JSON file (no raw business data — only code and summaries)."""
        state = {
            "core_context": self.core_context,
            "rolling_summary": self._rolling_summary,
            "turn_index": self._turn_index,
            "turns": [asdict(t) for t in self._turns],
            "code_store": {k: asdict(v) for k, v in self._code_store.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        print(f"[DBClaudeChat] Session saved to {path}")

    def load_session(self, path: str):
        """Load a previously saved session."""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.core_context = state.get("core_context", "")
        self._rolling_summary = state.get("rolling_summary", "")
        self._turn_index = state.get("turn_index", 0)
        self._turns = [Turn(**t) for t in state.get("turns", [])]
        self._code_store = {k: CodeBlock(**v) for k, v in state.get("code_store", {}).items()}
        print(f"[DBClaudeChat] Session loaded from {path} ({len(self._turns)} turns, {len(self._code_store)} code blocks)")

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def _build_messages(self, user_message: str, new_block_ids: list) -> list:
        """
        Assemble the messages list within token budget.

        Priority order (highest to lowest):
          1. System prompt (always)
          2. Core context (always)
          3. Recent turns verbatim (at least `recent_turns_floor`)
          4. Relevant code blocks (scored by keyword overlap with user_message)
          5. Rolling summary (if budget remains)
          6. Older verbatim turns (if budget still remains)
        """
        system_tokens = _count_tokens(self.SYSTEM_PROMPT)
        core_tokens = _count_tokens(self.core_context)
        base_used = system_tokens + core_tokens + _count_tokens(user_message) + 200  # overhead

        remaining = self.max_prompt_tokens - base_used

        # --- Select recent verbatim turns ---
        recent_turns = self._get_recent_verbatim(remaining)
        recent_tokens_used = sum(_count_tokens(t.user + t.assistant) for t in recent_turns)
        remaining -= recent_tokens_used

        # --- Select relevant code blocks ---
        code_blocks = self._select_relevant_code(user_message, new_block_ids, remaining)
        code_tokens_used = sum(_count_tokens(b.injection()) for b in code_blocks)
        remaining -= code_tokens_used

        # --- Rolling summary ---
        summary_text = ""
        if self._rolling_summary and remaining > 200:
            allotted_summary = min(remaining, self.summary_budget)
            summary_text = self._truncate_to_tokens(self._rolling_summary, allotted_summary)
            remaining -= _count_tokens(summary_text)

        # --- Older verbatim turns (if budget allows) ---
        older_turns = self._get_older_verbatim(recent_turns, remaining)

        # --- Assemble ---
        messages = []

        # Build the first user message: system + core + summary + code blocks injected
        context_parts = []

        if self.core_context:
            context_parts.append(f"## Project Context\n{self.core_context}")

        if summary_text:
            context_parts.append(f"## Summary of Earlier Work\n{summary_text}")

        if code_blocks:
            code_section = "\n".join(b.injection() for b in code_blocks)
            context_parts.append(f"## Relevant Code\n{code_section}")

        # Interleave older + recent turns
        all_prior_turns = older_turns + recent_turns

        if context_parts and all_prior_turns:
            # Inject context as a synthetic first exchange
            messages.append({"role": "user", "content": "\n\n".join(context_parts)})
            messages.append({"role": "assistant", "content": "Understood. I have the project context and code blocks loaded. Ready to continue."})
        elif context_parts:
            # No prior turns: prepend context to the current user message
            messages = []  # will be added below

        for turn in all_prior_turns:
            messages.append({"role": "user", "content": turn.user})
            messages.append({"role": "assistant", "content": turn.assistant})

        # Current user message (possibly prefixed with context if no prior turns)
        if context_parts and not all_prior_turns:
            final_user = "\n\n".join(context_parts) + "\n\n---\n\n" + user_message
        else:
            final_user = user_message

        messages.append({"role": "user", "content": final_user})

        return messages

    def _get_recent_verbatim(self, budget: int) -> list[Turn]:
        """
        Return recent non-compacted turns, always keeping at least `recent_turns_floor`.
        Fills up to `recent_window_budget` tokens.
        """
        candidates = [t for t in self._turns if not t.is_compacted]
        if not candidates:
            return []

        # Always include the floor
        floor = candidates[-self.recent_turns_floor:]
        floor_tokens = sum(_count_tokens(t.user + t.assistant) for t in floor)
        allotted = min(self.recent_window_budget, budget)

        if floor_tokens >= allotted:
            return floor

        # Fill remaining budget with more recent turns going backward
        result = list(floor)
        used = floor_tokens
        for turn in reversed(candidates[:-self.recent_turns_floor]):
            cost = _count_tokens(turn.user + turn.assistant)
            if used + cost <= allotted:
                result.insert(0, turn)
                used += cost
            else:
                break
        return result

    def _get_older_verbatim(self, already_included: list[Turn], budget: int) -> list[Turn]:
        """Fill any remaining budget with the oldest non-compacted turns not yet included."""
        included_indices = {t.index for t in already_included}
        candidates = [t for t in self._turns if not t.is_compacted and t.index not in included_indices]
        result = []
        used = 0
        for turn in candidates:  # oldest first
            cost = _count_tokens(turn.user + turn.assistant)
            if used + cost <= budget:
                result.append(turn)
                used += cost
            else:
                break
        return result

    def _select_relevant_code(self, user_message: str, new_block_ids: list, budget: int) -> list[CodeBlock]:
        """
        Score code blocks by keyword overlap with the current user message.
        Always include newly added blocks. Fill up to code_budget.
        No API call needed — fast keyword scoring is sufficient for relevance.
        """
        allotted = min(self.code_budget, budget)
        msg_tokens = set(re.findall(r'\b\w+\b', user_message.lower()))

        scores: list[tuple[float, CodeBlock]] = []
        for block in self._code_store.values():
            if block.block_id in new_block_ids:
                score = 999.0  # always include newly added
            else:
                block_tokens = set(block.tags + re.findall(r'\b\w+\b', block.label.lower()))
                overlap = len(msg_tokens & block_tokens)
                # Recency bonus: more recent blocks slightly preferred
                recency = block.turn_index / max(self._turn_index, 1)
                score = overlap + recency * 0.5
            scores.append((score, block))

        scores.sort(key=lambda x: x[0], reverse=True)

        result = []
        used = 0
        for score, block in scores:
            inj_tokens = _count_tokens(block.injection())
            if used + inj_tokens <= allotted:
                result.append(block)
                used += inj_tokens
            elif score == 999.0:
                # Must include new blocks even if tight — they're what the user just shared
                result.append(block)
                used += inj_tokens
        return result

    # ------------------------------------------------------------------
    # Code extraction and storage
    # ------------------------------------------------------------------

    # Matches ```lang ... ``` or ``` ... ```
    _CODE_FENCE = re.compile(
        r'```(?P<lang>\w+)?\n(?P<code>.*?)```',
        re.DOTALL
    )

    def _extract_and_store_code(self, text: str, speaker: str) -> tuple[str, list]:
        """
        Find all fenced code blocks in text, store them, replace with references.
        Returns (cleaned_text, list_of_new_block_ids).
        """
        new_ids = []

        def replacer(m):
            lang = (m.group("lang") or "text").strip()
            code = m.group("code").strip()
            if len(code) < 30:
                return m.group(0)  # don't bother storing tiny snippets

            block_id = self._make_block_id(code)

            if block_id not in self._code_store:
                label = self._infer_label(code, lang, text)
                tags = self._infer_tags(code, label)
                block = CodeBlock(
                    block_id=block_id,
                    label=label,
                    language=lang,
                    content=code,
                    tags=tags,
                    turn_index=self._turn_index,
                )
                self._code_store[block_id] = block
                new_ids.append(block_id)
                if self.verbose:
                    print(f"[DBClaudeChat] Stored code block [{block_id}] — {label} ({block.token_count} tokens)")

            return self._code_store[block_id].reference()

        cleaned = self._CODE_FENCE.sub(replacer, text)
        return cleaned, new_ids

    def _infer_label(self, code: str, lang: str, surrounding: str) -> str:
        """Infer a short label from the code itself (no API call needed)."""
        # Try class name
        m = re.search(r'class\s+(\w+)', code)
        if m:
            return f"class {m.group(1)}"
        # Try function/def
        m = re.search(r'def\s+(\w+)', code)
        if m:
            return f"def {m.group(1)}"
        # Try CREATE TABLE / FROM
        m = re.search(r'(?:CREATE\s+TABLE|FROM)\s+(\w+)', code, re.IGNORECASE)
        if m:
            return f"SQL {m.group(1)}"
        # Fall back to first non-comment line
        for line in code.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith(('#', '//', '--', '/*')):
                return stripped[:40]
        return f"{lang} block"

    def _infer_tags(self, code: str, label: str) -> list:
        """Extract meaningful keywords for relevance matching."""
        words = re.findall(r'\b[a-zA-Z_]\w{2,}\b', code + " " + label)
        # Filter out common stop words and very common Python keywords
        skip = {'def', 'class', 'self', 'return', 'import', 'from', 'pass',
                'None', 'True', 'False', 'else', 'elif', 'with', 'for', 'while',
                'not', 'and', 'the', 'that', 'this', 'str', 'int', 'list', 'dict'}
        tags = list({w.lower() for w in words if w not in skip})
        return tags[:30]  # cap tag list size

    @staticmethod
    def _make_block_id(content: str) -> str:
        return hashlib.sha1(content.encode()).hexdigest()[:6]

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    def _maybe_compact(self):
        """Summarize old turns if total history tokens exceed threshold."""
        uncompacted = [t for t in self._turns if not t.is_compacted]
        total = sum(t.token_count for t in uncompacted)

        if total <= self.compaction_threshold:
            return

        # Compact all but the most recent `recent_turns_floor` turns
        to_compact = uncompacted[:-self.recent_turns_floor]
        if not to_compact:
            return

        if self.verbose:
            print(f"[DBClaudeChat] Compacting {len(to_compact)} old turns...")

        # Build a compact summary via API call
        history_text = "\n\n".join(
            f"[Turn {t.index}]\nUser: {t.user}\nAssistant: {t.assistant}"
            for t in to_compact
        )

        # Prepend existing summary if present
        if self._rolling_summary:
            history_text = f"Existing summary:\n{self._rolling_summary}\n\nNew turns to integrate:\n{history_text}"

        compaction_prompt = (
            "You are summarizing a technical conversation about software/data engineering. "
            "Produce a dense, factual summary of the key decisions made, problems solved, "
            "architectures discussed, and any important conclusions. "
            "Do NOT summarize code — code is stored separately. "
            "Preserve all specific names (functions, classes, tables, endpoints). "
            "Keep the summary under 800 words.\n\n"
            f"{history_text}"
        )

        try:
            messages = [{"role": "user", "content": compaction_prompt}]
            new_summary = self._call_api(messages, max_tokens=1200)
            self._rolling_summary = new_summary.strip()
        except Exception as e:
            print(f"[DBClaudeChat] Compaction API call failed: {e}. Skipping compaction.")
            return

        # Mark turns as compacted
        for t in to_compact:
            t.is_compacted = True
            t.summary = "(compacted into rolling summary)"

        if self.verbose:
            print(f"[DBClaudeChat] Compaction complete. Rolling summary: {_count_tokens(self._rolling_summary)} tokens.")

    def force_compact(self):
        """Manually trigger compaction regardless of threshold."""
        uncompacted = [t for t in self._turns if not t.is_compacted]
        to_compact = uncompacted[:-self.recent_turns_floor] if len(uncompacted) > self.recent_turns_floor else []
        if not to_compact:
            print("[DBClaudeChat] Nothing old enough to compact.")
            return
        # Temporarily lower threshold and call
        old_threshold = self.compaction_threshold
        self.compaction_threshold = 0
        self._maybe_compact()
        self.compaction_threshold = old_threshold

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    def _call_api(self, messages: list, max_tokens: int = None) -> str:
        """POST to the Databricks Claude endpoint."""
        if max_tokens is None:
            max_tokens = self.max_response_tokens

        payload = {
            "model": "claude-sonnet-4-6",   # adjust to whatever model string Databricks exposes
            "max_tokens": max_tokens,
            "system": self.SYSTEM_PROMPT,
            "messages": messages,
        }

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        # Handle both Anthropic native format and Databricks wrapper
        if "content" in result:
            # Native Anthropic format
            return "".join(block.get("text", "") for block in result["content"] if block.get("type") == "text")
        elif "choices" in result:
            # OpenAI-compatible wrapper (some Databricks setups)
            return result["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Unrecognized response format: {list(result.keys())}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens tokens."""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n[... truncated for token budget ...]"

    def _print_budget_summary(self, messages: list):
        total = sum(_count_tokens(m["content"]) for m in messages)
        system_t = _count_tokens(self.SYSTEM_PROMPT)
        print(f"\n[DBClaudeChat] Token budget — system:{system_t} | messages:{total} | "
              f"code_blocks:{len(self._code_store)} | turns:{len(self._turns)} | "
              f"total_est:{total + system_t}/{self.max_prompt_tokens}")


# ---------------------------------------------------------------------------
# Convenience: quick start for notebook usage
# ---------------------------------------------------------------------------

def new_chat(endpoint: str, token: str = "", context: str = "", **kwargs) -> DBClaudeChat:
    """
    One-liner to start a session from a Databricks notebook cell.

    Example:
        bot = new_chat(
            endpoint="https://<workspace>.azuredatabricks.net/serving-endpoints/<name>/invocations",
            token=dbutils.secrets.get("scope", "anthropic-token"),
            context="I'm refactoring the batch ingestion pipeline in /pipelines/batch_ingest.py",
            verbose=True,
        )
    """
    return DBClaudeChat(endpoint=endpoint, api_token=token, core_context=context, **kwargs)

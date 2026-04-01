"""Agent framework for autonomous task execution with tool calling.

This module implements an agent that uses the OpenRouter API for LLM inference,
supporting streaming responses, tool calling, and reasoning token display.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

from framework.llm import OpenRouterClient, OpenRouterConfig, TokenUsage

ANSWER_SUBMITTED_PREFIX = "ANSWER_SUBMITTED:"

BUSINESS_TERM_PATTERNS = (
    r"\bon[- ]?time\b",
    r"\bcompleted\b",
    r"\bcurrent\b",
    r"\bactive\b",
    r"\bperforming\b",
    r"\bsevere\b",
    r"\brefund\b",
    r"\blegacy\b",
    r"\brookie\b",
    r"\bhall of fame\b",
    r"\bsession\b",
    r"\bcomparable\b",
    r"\bweighted\b",
    r"\bprivacy\b",
    r"\bheadcount\b",
    r"\bexcluding\b",
    r"\bonly count\b",
    r"\bcount only\b",
    r"\bshould be weighted\b",
)
_SCHEMA_TABLE_PATTERN = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\b")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "with",
}

_COMMAND_WORDS = {
    "show",
    "list",
    "find",
    "calculate",
    "round",
    "return",
    "display",
    "give",
    "provide",
    "report",
    "ordered",
    "order",
    "descending",
    "ascending",
    "using",
    "formula",
    "number",
    "showing",
    "showed",
    "top",
    "bottom",
    "include",
    "including",
    "exclude",
    "excluding",
    "count",
    "counts",
    "many",
    "whose",
}
_GENERIC_SINGLE_WORDS = _COMMAND_WORDS | {
    "name",
    "names",
    "value",
    "values",
    "total",
    "average",
    "amount",
    "current",
    "show",
    "player",
    "players",
    "member",
    "members",
    "employee",
    "employees",
    "rate",
    "statistics",
    "statistic",
    "formula",
    "result",
    "results",
    "table",
    "data",
    "rows",
    "record",
    "records",
}
_BUSINESS_UNIGRAM_ALLOWLIST = {
    "era",
    "whip",
    "bbwaa",
    "rookie",
    "sicilian",
    "french",
    "session",
    "legacy",
    "refund",
}
_GUIDE_QUERY_STOPWORDS = _STOPWORDS | _COMMAND_WORDS
_CATALOG_QUERY_STOPWORDS = _GUIDE_QUERY_STOPWORDS | {
    "current",
    "active",
    "performing",
    "completed",
    "severe",
    "weighted",
    "comparable",
    "legacy",
    "excluding",
    "exclude",
    "only",
}
_RETRIEVAL_TOOLS = {"search_guides", "search_catalog", "list_tables", "describe_table", "profile_columns", "preview_rows"}
_BOOTSTRAP_QUERY_HINT_LIMIT = 2
_AUXILIARY_BOOTSTRAP_PHRASE_LIMIT = 1
_MAX_BOOTSTRAP_GUIDE_QUERIES = 2
_MAX_BOOTSTRAP_CATALOG_QUERIES = 3
_GUIDE_SIGNAL_STOPWORDS = _STOPWORDS | _COMMAND_WORDS | {
    "rule", "rules", "metric", "metrics", "database", "databases", "analytics", "analysis",
    "standard", "standards", "business", "guideline", "guidelines", "classification", "classifications",
    "reporting", "report", "calculate", "calculations", "show", "return", "count", "counts",
}
_FACT_TABLE_GUIDE_PATTERNS = (
    re.compile(r"\b(always reconcile|sum of .* minus .* refunds|subtract.*refund|recomputed?|computed? from)\b", re.IGNORECASE),
    re.compile(r"\b(raw|base|event[- ]level|transaction[- ]level|line[- ]item|one row per)\b", re.IGNORECASE),
)
RAG_DIR = Path(__file__).parent.parent / ".rag"
RAG_INDEX_PATH = RAG_DIR / "guides_index.json"
_BOOTSTRAP_RAG_CACHE: dict[str, Any] | None = None


type ToolFunction = Callable[..., str]

GUIDES_DIR = Path(__file__).parent.parent / "evaluation" / "data" / "guides"


def _load_guide_names() -> list[str]:
    if not GUIDES_DIR.exists():
        return []
    return sorted(guide_path.stem for guide_path in GUIDES_DIR.glob("*.md"))


class EventType(Enum):
    GENERATION_START = auto()
    THINKING_START = auto()
    THINKING_CHUNK = auto()
    THINKING_END = auto()
    RESPONSE_CHUNK = auto()
    GENERATION_END = auto()
    BOOTSTRAP_CONTEXT_BUILT = auto()
    TOOL_CALL_START = auto()
    TOOL_CALL_PARSED = auto()
    TOOL_EXECUTION_START = auto()
    TOOL_EXECUTION_END = auto()
    ITERATION_START = auto()
    ITERATION_END = auto()
    AGENT_COMPLETE = auto()
    AGENT_ERROR = auto()


@dataclass
class AgentEvent:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.type.name}] {self.data}"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    function: ToolFunction


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    error: str | None = None


@dataclass
class Message:
    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


@dataclass
class ContextCompressionSettings:
    enabled: bool = False
    keep_recent: int = 3
    max_chars: int = 500


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)

    def to_api_format(self, compression: ContextCompressionSettings | None = None) -> list[dict[str, Any]]:
        messages_to_convert = self.messages
        if compression and compression.enabled:
            messages_to_convert = _compress_messages(
                self.messages,
                keep_recent=compression.keep_recent,
                max_chars=compression.max_chars,
            )
        result: list[dict[str, Any]] = []
        for message in messages_to_convert:
            msg: dict[str, Any] = {"role": message.role}
            if message.content is not None:
                msg["content"] = message.content
            if message.tool_calls is not None:
                msg["tool_calls"] = message.tool_calls
            if message.tool_call_id is not None:
                msg["tool_call_id"] = message.tool_call_id
            result.append(msg)
        return result


def _truncate_tool_result(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    first_line = content.split("\n")[0]
    if len(first_line) <= max_chars - 20:
        return f"[Truncated] {first_line}"
    return f"[Truncated] {content[:max_chars - 15]}..."


def _compress_messages(messages: list[Message], keep_recent: int, max_chars: int) -> list[Message]:
    tool_indices = [i for i, m in enumerate(messages) if m.role == "tool"]
    recent_tool_indices = set(tool_indices[-keep_recent:]) if tool_indices else set()
    result: list[Message] = []
    for i, msg in enumerate(messages):
        if msg.role == "tool" and i not in recent_tool_indices and msg.content:
            result.append(
                Message(role=msg.role, content=_truncate_tool_result(msg.content, max_chars), tool_call_id=msg.tool_call_id)
            )
        else:
            result.append(msg)
    return result


def _normalize_search_intent(tool_name: str, arguments: dict[str, Any]) -> str:
    if tool_name == "search_guides":
        raw = arguments.get("searchText", "")
    elif tool_name == "search_catalog":
        raw = arguments.get("searchText", "")
    elif tool_name == "describe_table":
        raw = f"{arguments.get('schemaName', '')}.{arguments.get('tableName', '')} {arguments.get('focusText', '')}"
    elif tool_name == "list_tables":
        raw = arguments.get("schemaName", "")
    elif tool_name == "profile_columns":
        cols = " ".join(arguments.get("columnNames", []) or [])
        raw = f"{arguments.get('schemaName', '')}.{arguments.get('tableName', '')} {cols}"
    else:
        raw = json.dumps(arguments, sort_keys=True)
    tokens = [token for token in re.findall(r"[a-z0-9]+", raw.lower()) if token not in _STOPWORDS]
    return f"{tool_name}:" + " ".join(sorted(tokens))


def _load_bootstrap_rag_index() -> dict[str, Any] | None:
    global _BOOTSTRAP_RAG_CACHE
    if _BOOTSTRAP_RAG_CACHE is not None:
        return _BOOTSTRAP_RAG_CACHE
    if not RAG_INDEX_PATH.exists():
        return None
    try:
        _BOOTSTRAP_RAG_CACHE = json.loads(RAG_INDEX_PATH.read_text(encoding="utf-8"))
        return _BOOTSTRAP_RAG_CACHE
    except Exception:
        return None


def _normalize_phrase(phrase: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", phrase.lower()))


def _strip_parenthetical_examples(text: str) -> str:
    text = re.sub(r"\((?:e\.?g\.?|i\.?e\.?|for example)[^)]*\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    return text


def _is_low_value_phrase(phrase: str) -> bool:
    tokens = phrase.split()
    if not tokens:
        return True
    if len(tokens) == 1:
        token = tokens[0]
        if token in _GENERIC_SINGLE_WORDS and token not in _BUSINESS_UNIGRAM_ALLOWLIST:
            return True
        if len(token) < 2 and token not in _BUSINESS_UNIGRAM_ALLOWLIST:
            return True
    if all(tok in _GENERIC_SINGLE_WORDS for tok in tokens):
        return True
    if any(tok in {"e", "g", "eg", "ie"} for tok in tokens):
        return True
    return False


def _phrase_candidates(prompt: str, purpose: str) -> list[str]:
    normalized = _strip_parenthetical_examples(prompt).replace("_", " ").replace("-", " ")
    stopwords = _GUIDE_QUERY_STOPWORDS if purpose == "guide" else _CATALOG_QUERY_STOPWORDS
    phrases: list[str] = []
    seen: set[str] = set()

    def add_phrase(value: str, allow_single: bool = False) -> None:
        cleaned = _normalize_phrase(value)
        if not cleaned:
            return
        raw_tokens = cleaned.split()
        tokens = [tok for tok in raw_tokens if tok not in stopwords]
        if not tokens:
            return
        if len(tokens) == 1:
            token = tokens[0]
            if token in _GENERIC_SINGLE_WORDS and token not in _BUSINESS_UNIGRAM_ALLOWLIST:
                return
            if not allow_single and token not in _BUSINESS_UNIGRAM_ALLOWLIST:
                return
        candidate = " ".join(tokens)
        if len(candidate) < 3 or candidate in seen:
            return
        if _is_low_value_phrase(candidate):
            return
        seen.add(candidate)
        phrases.append(candidate)

    for quoted in re.findall(r'"([^"]+)"', prompt):
        add_phrase(quoted, allow_single=False)

    lowered = prompt.lower()
    for phrase in [
        "on time performance" if ("on-time" in lowered or "on time" in lowered) else None,
        "on time" if ("on-time" in lowered or "on time" in lowered) else None,
        "hall of fame" if "hall of fame" in lowered else None,
        "completed flights" if "completed flights" in lowered else None,
        "average transaction value" if "average transaction value" in lowered else None,
        "performing loans" if "performing loans" in lowered else None,
        "legacy workforce" if "legacy workforce" in lowered else None,
        "default rate" if "default rate" in lowered else None,
        "session beer" if ("session beer" in lowered or "session beers" in lowered) else None,
        "net charge volume" if "net charge volume" in lowered else None,
    ]:
        if phrase:
            add_phrase(phrase, allow_single=False)

    tokens = [tok for tok in re.findall(r"[a-z0-9]+", normalized.lower()) if tok not in stopwords]
    for n in (4, 3, 2):
        for i in range(len(tokens) - n + 1):
            add_phrase(" ".join(tokens[i : i + n]), allow_single=False)

    for tok in tokens:
        add_phrase(tok, allow_single=True)
    return phrases


def _phrase_tfidf_score(phrase: str, rag_index: dict[str, Any] | None) -> float:
    tokens = [tok for tok in phrase.split() if tok not in _STOPWORDS]
    if not tokens:
        return 0.0
    if not rag_index:
        return float(len(tokens))
    guide_idf = rag_index.get("guide_idf", {}) or {}
    chunk_idf = rag_index.get("chunk_idf", {}) or {}
    score = 0.0
    for token in tokens:
        score += max(float(guide_idf.get(token, 0.0)), float(chunk_idf.get(token, 0.0)), 0.0)
    score /= max(1, len(tokens))
    phrase_lower = phrase.lower()
    if any(phrase_lower in guide.get("text", "").lower() for guide in rag_index.get("guides", [])):
        score += 2.5
    if any(phrase_lower in chunk.get("heading", "").lower() for chunk in rag_index.get("chunks", [])):
        score += 3.5
    if len(tokens) >= 2:
        score += 1.2
    elif tokens[0] in _BUSINESS_UNIGRAM_ALLOWLIST:
        score += 1.0
    return score


def _catalog_phrase_score(phrase: str, catalog_vocab: set[str]) -> float:
    tokens = phrase.split()
    if not tokens:
        return 0.0
    vocab_hits = sum(1 for tok in tokens if tok in catalog_vocab)
    if vocab_hits == 0:
        return 0.0
    score = 2.5 * vocab_hits / len(tokens)
    if len(tokens) >= 2:
        score += 1.0
    if any(tok in {"carrier", "route", "delay", "loan", "district", "salary", "gender", "charge", "refund", "transaction", "employee", "department", "opening", "pitcher", "season", "constructor", "capital", "country", "city", "population", "beer", "abv", "statement"} for tok in tokens):
        score += 0.9
    return score


def _select_bootstrap_phrase(prompt: str, purpose: str) -> str | None:
    rag_index = _load_bootstrap_rag_index()
    candidates = _phrase_candidates(prompt, purpose=purpose)
    if not candidates:
        return None

    scored: list[tuple[float, str]] = []
    if purpose == "guide":
        for phrase in candidates:
            tokens = phrase.split()
            score = _phrase_tfidf_score(phrase, rag_index)
            if any(re.search(pattern, phrase) for pattern in BUSINESS_TERM_PATTERNS):
                score += 2.0
            if len(tokens) >= 2:
                score += 0.75
            if len(tokens) == 1 and phrase not in _BUSINESS_UNIGRAM_ALLOWLIST:
                score -= 2.0
            if _is_low_value_phrase(phrase):
                score -= 5.0
            scored.append((score, phrase))
    else:
        from framework.database import get_catalog_vocabulary

        catalog_vocab = set(get_catalog_vocabulary())
        for phrase in candidates:
            tokens = phrase.split()
            score = _catalog_phrase_score(phrase, catalog_vocab)
            if score <= 0.0:
                continue
            if len(tokens) >= 2:
                score += 0.5
            if len(tokens) == 1 and phrase not in _BUSINESS_UNIGRAM_ALLOWLIST:
                score -= 1.5
            if _is_low_value_phrase(phrase):
                score -= 5.0
            scored.append((score, phrase))

    if not scored:
        return None

    scored.sort(key=lambda item: (-item[0], -len(item[1].split()), item[1]))
    best_score, best_phrase = scored[0]
    min_score = 4.0 if purpose == "guide" else 2.5
    if best_score < min_score:
        return None
    if len(best_phrase.split()) == 1 and best_phrase not in _BUSINESS_UNIGRAM_ALLOWLIST:
        return None
    return best_phrase

def _compact_bootstrap_results(results: list[str], section: str, limit: int) -> list[str]:
    compacted: list[str] = []
    for result in results[:limit]:
        lines = [line.rstrip() for line in result.splitlines() if line.strip()]
        if not lines:
            continue
        kept: list[str] = []
        bullet_count = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('Top '):
                continue
            if stripped[0].isdigit() and '. ' in stripped:
                if kept:
                    break
                kept.append(line)
                continue
            if stripped.startswith('- '):
                kept.append(line)
                bullet_count += 1
                if bullet_count >= 2:
                    break
                continue
            if not kept:
                kept.append(line)
        if kept:
            compacted.append("\n".join(kept))
    return compacted


def _extract_schema_info_keys(tool_name: str, arguments: dict[str, Any], result: str) -> set[str]:
    keys: set[str] = set()
    if tool_name == "list_tables":
        schema = arguments.get("schemaName", "")
        if schema:
            keys.add(f"schema:{schema}")
    elif tool_name in {"describe_table", "profile_columns", "preview_rows"}:
        schema = arguments.get("schemaName", "")
        table = arguments.get("tableName", "")
        if schema and table:
            keys.add(f"table:{schema}.{table}")
    elif tool_name == "search_catalog":
        for line in result.splitlines():
            line = line.strip()
            if not line or not line[0].isdigit() or '. ' not in line:
                continue
            body = line.split('. ', 1)[1].split(' [score=', 1)[0]
            if '.' in body:
                schema, table = body.split('.', 1)
                keys.add(f"table:{schema}.{table}")
    return keys


def _extract_tokens_for_catalog_query(text: str, max_items: int = 8) -> list[str]:
    tokens = []
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", text):
        norm = token.lower()
        if norm in _GUIDE_SIGNAL_STOPWORDS or len(norm) < 3:
            continue
        if norm in _GENERIC_SINGLE_WORDS and norm not in _BUSINESS_UNIGRAM_ALLOWLIST:
            continue
        tokens.append(norm)
    # preserve order while deduping
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= max_items:
            break
    return result


def _parse_guide_result_metadata(result_text: str) -> tuple[list[str], list[str], list[str], list[str]]:
    guide_names: list[str] = []
    headings: list[str] = []
    identifiers: list[str] = []
    codes: list[str] = []

    for match in re.finditer(r"^- ([a-z0-9_]+) :: ([^\[]+) \[score=", result_text, flags=re.MULTILINE):
        guide_names.append(match.group(1).strip())
        headings.append(match.group(2).strip())

    for ident in re.findall(r"[a-z][a-z0-9]*_[a-z0-9_]+", result_text):
        identifiers.append(ident.lower())
    for code in re.findall(r"'([A-Za-z0-9]+)'", result_text):
        codes.append(code.lower())
    for code in re.findall(r'"([A-Za-z][A-Za-z0-9 _-]{1,30})"', result_text):
        norm = _normalize_phrase(code)
        if norm and len(norm.split()) <= 3:
            codes.extend(norm.split())

    return guide_names, headings, identifiers, codes


def _build_guide_enriched_catalog_query(prompt: str, guide_contexts: list[str]) -> str | None:
    if not guide_contexts:
        return None
    prompt_tokens = _extract_tokens_for_catalog_query(prompt, max_items=6)
    if not prompt_tokens:
        return None

    guide_name_tokens: list[str] = []
    heading_tokens: list[str] = []
    identifier_tokens: list[str] = []
    code_tokens: list[str] = []
    fact_bias = False

    for context in guide_contexts:
        guide_names, headings, identifiers, codes = _parse_guide_result_metadata(context)
        for name in guide_names:
            guide_name_tokens.extend(_extract_tokens_for_catalog_query(name.replace('_', ' '), max_items=6))
        for heading in headings:
            heading_tokens.extend(_extract_tokens_for_catalog_query(heading, max_items=6))
        identifier_tokens.extend([ident for ident in identifiers if ident not in {"current_date"}])
        code_tokens.extend([code for code in codes if len(code) >= 1])
        if any(pattern.search(context) for pattern in _FACT_TABLE_GUIDE_PATTERNS):
            fact_bias = True

    pieces: list[str] = []
    pieces.extend(prompt_tokens[:4])
    pieces.extend(guide_name_tokens[:2])
    pieces.extend(heading_tokens[:3])
    pieces.extend(identifier_tokens[:4])
    pieces.extend(code_tokens[:3])
    if fact_bias:
        pieces.extend([token for token in ["charge", "loan", "transaction", "result", "flight", "statement", "account"] if token in prompt_tokens + guide_name_tokens + heading_tokens + identifier_tokens][:2])

    seen: set[str] = set()
    ordered: list[str] = []
    for piece in pieces:
        norm = _normalize_phrase(piece)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        ordered.append(norm)
        if len(ordered) >= 10:
            break
    if len(ordered) < 3:
        return None
    return " ".join(ordered)


def _build_fact_table_hint_lines(guide_contexts: list[str], catalog_contexts: list[str]) -> list[str]:
    lines: list[str] = []
    guide_blob = "\n".join(guide_contexts)
    if guide_blob:
        if any(pattern.search(guide_blob) for pattern in _FACT_TABLE_GUIDE_PATTERNS):
            lines.append("- Source-of-truth hint: prefer raw fact/event tables over pre-aggregated summary tables when the guide says a metric must reconcile, subtract refunds, or be recomputed from primitive events.")
        guide_names, headings, identifiers, codes = _parse_guide_result_metadata(guide_blob)
        important_fields = []
        for item in identifiers + codes:
            norm = _normalize_phrase(item)
            if norm and norm not in important_fields:
                important_fields.append(norm)
        if important_fields:
            lines.append("- Guide-mentioned fields/codes to verify in the chosen fact table: " + ", ".join(important_fields[:8]))
        important_headings = []
        for heading in headings:
            norm = heading.strip()
            if norm and norm not in important_headings:
                important_headings.append(norm)
        if important_headings:
            lines.append("- Relevant business-rule sections: " + "; ".join(important_headings[:3]))
    if catalog_contexts:
        lines.append("- Fact-table check: choose the table that naturally contains the metric-defining fields, business-rule codes, and event-level rows together. Avoid summary tables unless the prompt explicitly asks for a stored summary metric.")
    return lines


class Agent:
    """A simple ReAct agent built on the OpenRouter API."""

    def __init__(self, config: OpenRouterConfig, tools: dict[str, Tool]):
        self.config = config
        self.tools = tools
        self.client = OpenRouterClient(config)
        self.conversation = Conversation()
        self._compression = ContextCompressionSettings(
            enabled=config.compress_context,
            keep_recent=config.compress_keep_recent,
            max_chars=config.compress_max_chars,
        )
        self._current_prompt = ""
        self._tool_cache: dict[str, str] = {}
        self._search_intent_counts: dict[str, int] = {}
        self._consecutive_retrieval_without_sql = 0
        self._consecutive_retrieval_without_new_schema_info = 0
        self._seen_schema_info_keys: set[str] = set()
        self.reset_conversation()

    def _get_tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools.values()
        ]

    def _get_system_message(self) -> str:
        from framework.database import list_schemas, list_tables

        schemas = list_schemas()
        schema_lines = []
        for schema in schemas:
            tables = list_tables(schema) or []
            table_str = ", ".join(tables[:10])
            suffix = f" (+{len(tables)-10} more)" if len(tables) > 10 else ""
            schema_lines.append(f"  - {schema}: {table_str}{suffix}")
        schema_block = "\n".join(schema_lines)

        guide_names = _load_guide_names()
        guide_lines = "\n".join(f"  - {name}" for name in guide_names)
        guides_block = (
            "# Business Rule Guides\n\n"
            "Use search_guides(searchText) to retrieve relevant definitions, exclusions, and business rules from the markdown guides and the persisted ./.rag/ guide index.\n"
            "You do NOT need to memorize all guides; retrieve only the rules relevant to the current question.\n\n"
            f"Available guides:\n{guide_lines}"
            if guide_names
            else ""
        )

        return f"""You are an autonomous SQL agent querying a DuckDB database.
Complete every task by calling submit_answer with a valid SQL query.
Never ask for clarification — make your best judgment and proceed.

# Available Schemas

Use schema-qualified table names in all queries (e.g. schema.table).

{schema_block}

Use list_tables(schemaName) to see all tables in a schema.
Use describe_table(schemaName, tableName, focusText) to see all columns, types, and question-relevant columns. Focused columns are wrapped like __ColumnName__.
Use search_catalog(searchText) when the right schema, table, join key, or metric column is unclear.
Use search_guides(searchText) whenever the prompt contains business terms or policy-like wording.
Use profile_columns(schemaName, tableName, columnNames) to inspect nulls, distinct values, and ranges for candidate columns.
Use preview_rows(schemaName, tableName, limit) when column names are cryptic or you need example records to interpret columns.
Use run_sql(query) to test a query before submitting.

If the controller provides a "Bootstrap retrieved context" message before your first action, treat it as trusted retrieval context for the current question. You may still call tools to verify details.

# Workflow

1. Identify the likely business domain and schema.
2. If the right table or column is unclear, consult bootstrap catalog context or call search_catalog before guessing.
3. If the prompt uses business terms like current, active, completed, on-time, severe, weighted, comparable, refund, legacy, rookie, performing, Hall of Fame, or similar domain language, consult bootstrap guide context or call search_guides before finalizing SQL.
4. Use describe_table on the most likely tables with focusText based on the question to confirm exact columns and types.
5. Verify the source-of-truth fact table: prefer raw event/fact tables over summary tables when business rules mention exclusions, statuses, refunds, reconciliation, thresholds, or recomputing a metric from primitive events.
6. When two candidate columns look similar, use profile_columns to distinguish proxy indicators from the correct business metric.
7. When schema columns are cryptic or code-like, use preview_rows on the most likely table to inspect example records before choosing columns.
8. Prefer normalized dimension tables over denormalized text proxy columns for business/category filtering when both are available.
9. If the metric is defined in words, thresholds, or a formula, prefer computing it from primitive fields rather than trusting a similarly named derived column.
10. If guide context mentions explicit codes or field names (for example charge_code, statement_no, status codes, UROK, RF, WeatherDelay), verify that the chosen fact table actually contains them before finalizing SQL.
11. Use run_sql to sanity-check the logic and output shape before you submit.
12. Call submit_answer with your final SQL query.

# Pre-submit checklist

Before calling submit_answer, verify all of the following:
- Business semantics: If the prompt includes a KPI or business term, I have guide context from bootstrap retrieval or a search_guides call.
- Table grounding: I verified the source-of-truth fact table instead of stopping at the first plausible or summary table.
- Column grounding: I verified the exact metric column. If similar columns exist, I compared them explicitly.
- Constraint recovery: I checked for hidden exclusions, thresholds, date rules, sentinel values, status filters, reconciliation rules, or source-of-truth requirements from guides.
- Sanity check: I ran run_sql on the final query or a close preview query and the output shape matches the question.
- Loop control: After 2 consecutive retrieval cycles without meaningful new schema information, I must stop searching and validate the best current SQL candidate.

If the question is vague and I have not checked business-rule context, I am not ready to submit.
CRITICAL: You MUST call submit_answer to complete every task.
Do not return plain text answers.
Do not skip schema or guide retrieval when they are relevant; hidden business semantics matter.

{guides_block}"""

    def _should_bootstrap_guides(self, prompt: str) -> bool:
        from framework.database import get_catalog_vocabulary

        prompt_lower = prompt.lower()
        if any(re.search(pattern, prompt_lower) for pattern in BUSINESS_TERM_PATTERNS):
            return True
        prompt_tokens = [token for token in re.findall(r"[a-z0-9]+", prompt_lower) if token not in _STOPWORDS and len(token) > 2]
        if not prompt_tokens:
            return False
        catalog_vocab = get_catalog_vocabulary()
        ungrounded = [token for token in prompt_tokens if token not in catalog_vocab]
        semantic_markers = {
            "only", "excluding", "exclude", "current", "active", "legacy", "official", "weighted", "comparable", "severe",
            "average", "rate", "ratio", "value", "values", "net", "default", "performing", "session", "deposit", "refund",
            "reconcile", "reconciliation", "headcount", "current", "active", "completed", "status",
        }
        if any(token in semantic_markers for token in prompt_tokens):
            return True
        if len(prompt_tokens) <= 6 and any(token in {"average", "rate", "value", "net", "amount", "default", "performing", "session"} for token in prompt_tokens):
            return True
        return len(ungrounded) >= max(2, len(prompt_tokens) // 3)

    def _should_bootstrap_catalog(self, prompt: str) -> bool:
        if _SCHEMA_TABLE_PATTERN.search(prompt):
            return False
        return True

    def _bootstrap_guide_queries(self, prompt: str) -> list[str]:
        queries = [prompt]
        phrase = _select_bootstrap_phrase(prompt, purpose="guide")
        if phrase and _normalize_phrase(phrase) != _normalize_phrase(prompt):
            queries.append(phrase)
        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            norm = _normalize_phrase(query)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            deduped.append(query)
            if len(deduped) >= _MAX_BOOTSTRAP_GUIDE_QUERIES:
                break
        return deduped

    def _bootstrap_catalog_queries(self, prompt: str, guide_contexts: list[str] | None = None) -> list[str]:
        queries = [prompt]
        phrase = _select_bootstrap_phrase(prompt, purpose="catalog")
        if phrase and _normalize_phrase(phrase) != _normalize_phrase(prompt):
            queries.append(phrase)
        enriched = _build_guide_enriched_catalog_query(prompt, guide_contexts or [])
        if enriched and _normalize_phrase(enriched) not in {_normalize_phrase(q) for q in queries}:
            queries.append(enriched)
        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            norm = _normalize_phrase(query)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            deduped.append(query)
            if len(deduped) >= _MAX_BOOTSTRAP_CATALOG_QUERIES:
                break
        return deduped

    def _truncate_preview(self, value: str | None, max_chars: int = 400) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 3] + "..."

    def debug_bootstrap(self, prompt: str) -> dict[str, Any]:
        guides_triggered = self._should_bootstrap_guides(prompt) and "search_guides" in self.tools
        catalog_triggered = self._should_bootstrap_catalog(prompt) and "search_catalog" in self.tools
        guide_queries = self._bootstrap_guide_queries(prompt) if guides_triggered else []
        catalog_queries = self._bootstrap_catalog_queries(prompt) if catalog_triggered else []

        guide_contexts: list[str] = []
        guide_errors: list[str] = []
        if guides_triggered:
            for query in guide_queries:
                try:
                    result = self.tools["search_guides"].function(searchText=query, limit=4)
                except Exception as exc:
                    result = None
                    guide_errors.append(f"{query}: {exc}")
                if result and not result.startswith("No guide matches"):
                    guide_contexts.append(result)

        catalog_contexts: list[str] = []
        catalog_errors: list[str] = []
        if catalog_triggered:
            for query in catalog_queries:
                try:
                    result = self.tools["search_catalog"].function(searchText=query, limit=5)
                except Exception as exc:
                    result = None
                    catalog_errors.append(f"{query}: {exc}")
                if result and not result.startswith("No catalog matches"):
                    catalog_contexts.append(result)

        sections: list[str] = []
        if guide_contexts:
            sections.append("[Guide context]\n" + "\n\n".join(guide_contexts))
        if catalog_contexts:
            sections.append("[Catalog context]\n" + "\n\n".join(catalog_contexts))
        bootstrap_context = None
        if sections:
            bootstrap_context = "Bootstrap retrieved context for the current question:\n\n" + "\n\n".join(sections)

        return {
            "prompt": prompt,
            "guides_triggered": guides_triggered,
            "catalog_triggered": catalog_triggered,
            "guide_query_terms": guide_queries,
            "catalog_query_terms": catalog_queries,
            "guide_context_preview": self._truncate_preview("\n\n".join(guide_contexts) if guide_contexts else None),
            "catalog_context_preview": self._truncate_preview("\n\n".join(catalog_contexts) if catalog_contexts else None),
            "guide_result_count": len(guide_contexts),
            "catalog_result_count": len(catalog_contexts),
            "guide_errors": guide_errors,
            "catalog_errors": catalog_errors,
            "bootstrap_context_present": bootstrap_context is not None,
            "bootstrap_context_preview": self._truncate_preview(bootstrap_context, max_chars=700),
            "bootstrap_context": bootstrap_context,
        }

    def _build_bootstrap_context(self, prompt: str) -> tuple[str | None, dict[str, Any]]:
        debug = self.debug_bootstrap(prompt)
        return debug["bootstrap_context"], debug

    def _augment_tool_arguments(self, tool_call: ToolCall) -> ToolCall:
        if tool_call.name == "describe_table" and "focusText" not in tool_call.arguments:
            updated_args = dict(tool_call.arguments)
            updated_args["focusText"] = self._current_prompt
            return ToolCall(id=tool_call.id, name=tool_call.name, arguments=updated_args, error=tool_call.error)
        return tool_call

    def _execute_tool(self, tool_call: ToolCall) -> str:
        tool_call = self._augment_tool_arguments(tool_call)
        if tool_call.error:
            return f"Error parsing arguments for tool '{tool_call.name}': {tool_call.error}"
        if tool_call.name not in self.tools:
            return f"Error: Unknown tool '{tool_call.name}'"

        cache_key = f"{tool_call.name}:{json.dumps(tool_call.arguments, sort_keys=True, default=str)}"
        from_cache = cache_key in self._tool_cache
        if from_cache:
            result = self._tool_cache[cache_key]
        else:
            try:
                result = self.tools[tool_call.name].function(**tool_call.arguments)
            except Exception as e:
                result = f"Error executing {tool_call.name}: {e}"
            self._tool_cache[cache_key] = result

        if tool_call.name in {"run_sql", "submit_answer"}:
            self._consecutive_retrieval_without_sql = 0
            self._consecutive_retrieval_without_new_schema_info = 0
        elif tool_call.name in _RETRIEVAL_TOOLS:
            self._consecutive_retrieval_without_sql += 1
            new_schema_keys = _extract_schema_info_keys(tool_call.name, tool_call.arguments, result)
            discovered_new_info = any(key not in self._seen_schema_info_keys for key in new_schema_keys)
            self._seen_schema_info_keys.update(new_schema_keys)
            if discovered_new_info:
                self._consecutive_retrieval_without_new_schema_info = 0
            else:
                self._consecutive_retrieval_without_new_schema_info += 1

        if tool_call.name in _RETRIEVAL_TOOLS:
            intent_key = _normalize_search_intent(tool_call.name, tool_call.arguments)
            self._search_intent_counts[intent_key] = self._search_intent_counts.get(intent_key, 0) + 1
            notes: list[str] = []
            if from_cache:
                notes.append("[Loop guard] Reused cached result for an identical retrieval call.")
            if self._search_intent_counts[intent_key] >= 2:
                notes.append("[Loop guard] This retrieval intent has already been used. Prefer choosing the best candidate table/columns and validating with run_sql instead of repeating similar searches.")
            if self._consecutive_retrieval_without_new_schema_info >= 2:
                notes.append("[Loop guard] You have made 2 consecutive retrieval moves without discovering meaningful new schema information. Stop searching and validate the best current SQL candidate with run_sql.")
            elif self._consecutive_retrieval_without_sql >= 3:
                notes.append("[Loop guard] You have done several retrieval steps without validating a SQL query. Stop searching unless the intent changes substantially; move to run_sql on the best current candidate.")
            if notes:
                result = result + "\n\n" + "\n".join(notes)

        return result

    def _generate_response(self, conversation: Conversation) -> Iterator[AgentEvent]:
        yield AgentEvent(type=EventType.GENERATION_START)
        messages = conversation.to_api_format(compression=self._compression)
        tools = self._get_tool_definitions() if self.tools else None
        full_content = ""
        tool_calls: list[dict[str, Any]] = []
        in_thinking = False
        finish_reason: str | None = None
        usage: TokenUsage | None = None

        for chunk in self.client.chat_completion_stream(messages, tools):
            if chunk.reasoning_details:
                for detail in chunk.reasoning_details:
                    if detail.get("type") == "reasoning.text":
                        text = detail.get("text", "")
                        if text:
                            if not in_thinking:
                                in_thinking = True
                                yield AgentEvent(type=EventType.THINKING_START)
                            yield AgentEvent(type=EventType.THINKING_CHUNK, data={"chunk": text})
            if chunk.content:
                if in_thinking:
                    in_thinking = False
                    yield AgentEvent(type=EventType.THINKING_END)
                full_content += chunk.content
                yield AgentEvent(type=EventType.RESPONSE_CHUNK, data={"chunk": chunk.content})
            if chunk.tool_calls:
                tool_calls = chunk.tool_calls
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if chunk.usage:
                usage = chunk.usage

        if in_thinking:
            yield AgentEvent(type=EventType.THINKING_END)

        event_data: dict[str, Any] = {
            "full_response": full_content,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
        }
        if usage:
            event_data["usage"] = usage
        yield AgentEvent(type=EventType.GENERATION_END, data=event_data)

    def run(self, prompt: str) -> Iterator[AgentEvent]:
        self._current_prompt = prompt
        self.conversation.messages.append(Message(role="user", content=prompt))
        bootstrap_context, bootstrap_debug = self._build_bootstrap_context(prompt)
        yield AgentEvent(type=EventType.BOOTSTRAP_CONTEXT_BUILT, data=bootstrap_debug)
        if bootstrap_context:
            self.conversation.messages.append(Message(role="assistant", content=bootstrap_context))

        total_usage = TokenUsage()
        empty_response_count = 0

        for iteration in range(self.config.max_iterations):
            yield AgentEvent(type=EventType.ITERATION_START, data={"iteration": iteration + 1})
            full_response = ""
            tool_calls_data: list[dict[str, Any]] = []

            for event in self._generate_response(self.conversation):
                yield event
                if event.type == EventType.GENERATION_END:
                    full_response = event.data.get("full_response", "")
                    tool_calls_data = event.data.get("tool_calls", [])
                    if "usage" in event.data and event.data["usage"]:
                        total_usage = total_usage + event.data["usage"]

            tool_calls = _parse_tool_calls_from_api(tool_calls_data)

            if not tool_calls:
                is_empty = not full_response or not full_response.strip()
                if is_empty:
                    empty_response_count += 1
                    if empty_response_count >= 3:
                        yield AgentEvent(type=EventType.AGENT_ERROR, data={"error": "Too many empty generations without tool calls", "usage": total_usage})
                        return
                    self.conversation.messages.append(Message(role="assistant", content=""))
                    self.conversation.messages.append(Message(role="user", content="You must use the submit_answer tool to complete this task. Call it now with a SQL query."))
                    continue

                self.conversation.messages.append(Message(role="assistant", content=full_response))
                self.conversation.messages.append(Message(role="user", content="You must call submit_answer with a SQL query to complete this task."))
                continue

            empty_response_count = 0
            yield AgentEvent(type=EventType.TOOL_CALL_START, data={"count": len(tool_calls)})
            self.conversation.messages.append(Message(role="assistant", content=full_response if full_response else None, tool_calls=tool_calls_data))

            for tool_call in tool_calls:
                tool_call = self._augment_tool_arguments(tool_call)
                yield AgentEvent(type=EventType.TOOL_CALL_PARSED, data={"name": tool_call.name, "arguments": tool_call.arguments})
                yield AgentEvent(type=EventType.TOOL_EXECUTION_START, data={"name": tool_call.name})
                tool_result = self._execute_tool(tool_call)
                yield AgentEvent(type=EventType.TOOL_EXECUTION_END, data={"name": tool_call.name, "result": tool_result})
                self.conversation.messages.append(Message(role="tool", content=tool_result, tool_call_id=tool_call.id))
                if tool_result.startswith(ANSWER_SUBMITTED_PREFIX):
                    yield AgentEvent(type=EventType.AGENT_COMPLETE, data={"reason": "answer_submitted", "tool": tool_call.name, "usage": total_usage})
                    return

            yield AgentEvent(type=EventType.ITERATION_END, data={"iteration": iteration + 1})

        yield AgentEvent(type=EventType.AGENT_ERROR, data={"error": "Max iterations reached", "usage": total_usage})

    def reset_conversation(self) -> None:
        self.conversation = Conversation()
        self._tool_cache = {}
        self._search_intent_counts = {}
        self._consecutive_retrieval_without_sql = 0
        self._consecutive_retrieval_without_new_schema_info = 0
        self._seen_schema_info_keys = set()
        self._current_prompt = ""
        self.conversation.messages.append(Message(role="system", content=self._get_system_message()))


def _parse_tool_calls_from_api(tool_calls_data: list[dict[str, Any]]) -> list[ToolCall]:
    tool_calls: list[ToolCall] = []
    for tc in tool_calls_data:
        tc_id = tc.get("id", "")
        function = tc.get("function", {})
        name = function.get("name", "")
        arguments_str = function.get("arguments", "{}")
        try:
            arguments = json.loads(arguments_str)
            error = None
        except json.JSONDecodeError as e:
            arguments = {}
            error = f"Invalid JSON arguments: {e}"
        tool_calls.append(ToolCall(id=tc_id, name=name, arguments=arguments, error=error))
    return tool_calls

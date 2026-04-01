"""Tools for SQL execution, schema exploration, and business-rule retrieval."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import framework.agent as agentFramework
import framework.database as database

SQL_VALID_PREFIX = "SQL_VALID:"
GUIDES_DIR = Path(__file__).parent.parent / "evaluation" / "data" / "guides"
RAG_DIR = Path(__file__).parent.parent / ".rag"
RAG_VERSION = "guides_tfidf_v1"
RAG_INDEX_PATH = RAG_DIR / "guides_index.json"
MAX_GUIDE_MATCHES = 8
MAX_GUIDE_SNIPPET_CHARS = 450
_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
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
_GENERIC_GUIDE_TOKENS = {"data", "time", "management", "performance", "system", "process", "project"}


@dataclass
class _SearchVector:
    weights: dict[str, float]
    norm: float


_GUIDE_INDEX_CACHE: dict[str, Any] | None = None


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in _STOPWORDS]


def _phrase_candidates(text: str) -> list[str]:
    lowered = _normalize_text(text.replace("_", " ").replace("-", " "))
    phrases: set[str] = set()
    quoted = re.findall(r'"([^"]+)"', text)
    phrases.update(_normalize_text(item) for item in quoted if item.strip())
    tokens = [token for token in _tokenize(lowered) if token not in _GENERIC_GUIDE_TOKENS]
    for n in (2, 3):
        for idx in range(len(tokens) - n + 1):
            phrases.add(" ".join(tokens[idx : idx + n]))
    if "on time" in lowered:
        phrases.add("on time")
    if "hall of fame" in lowered:
        phrases.add("hall of fame")
    if "average transaction value" in lowered:
        phrases.add("average transaction value")
    return [phrase for phrase in phrases if phrase]


def _build_tfidf(doc_tokens_list: list[list[str]]) -> tuple[list[dict[str, float]], dict[str, float]]:
    num_docs = len(doc_tokens_list)
    doc_freq: Counter[str] = Counter()
    for tokens in doc_tokens_list:
        doc_freq.update(set(tokens))
    idf = {token: math.log((1 + num_docs) / (1 + freq)) + 1.0 for token, freq in doc_freq.items()}
    vectors: list[dict[str, float]] = []
    for tokens in doc_tokens_list:
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        vector = {token: (count / total) * idf[token] for token, count in counts.items() if token in idf}
        vectors.append(vector)
    return vectors, idf


def _vectorize_query(query: str, idf: dict[str, float]) -> _SearchVector:
    counts = Counter(_tokenize(query))
    total = sum(counts.values()) or 1
    weights = {token: (count / total) * idf[token] for token, count in counts.items() if token in idf}
    norm = math.sqrt(sum(value * value for value in weights.values()))
    return _SearchVector(weights=weights, norm=norm)


def _cosine_similarity(query_vec: _SearchVector, doc_vec: dict[str, float]) -> float:
    if not query_vec.weights or not doc_vec:
        return 0.0
    dot = sum(weight * doc_vec.get(token, 0.0) for token, weight in query_vec.weights.items())
    doc_norm = math.sqrt(sum(value * value for value in doc_vec.values()))
    if doc_norm == 0.0 or query_vec.norm == 0.0:
        return 0.0
    return dot / (doc_norm * query_vec.norm)


def _guide_manifest() -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    if not GUIDES_DIR.exists():
        return manifest
    for guide_path in sorted(GUIDES_DIR.glob("*.md")):
        stat = guide_path.stat()
        manifest.append(
            {
                "name": guide_path.name,
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            }
        )
    return manifest


def _split_guide_chunks(guide_name: str, raw_text: str) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    current_heading = "Overview"
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        content = "\n".join(buffer).strip()
        if content:
            chunks.append({"guide_name": guide_name, "heading": current_heading, "content": content})
        buffer = []

    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            flush()
            current_heading = stripped.lstrip("#").strip() or "Overview"
        else:
            buffer.append(line)
    flush()
    return chunks


def _build_guide_index() -> dict[str, Any]:
    guide_docs: list[dict[str, Any]] = []
    chunk_docs: list[dict[str, Any]] = []

    for guide_path in sorted(GUIDES_DIR.glob("*.md")):
        raw_text = guide_path.read_text(encoding="utf-8")
        chunk_dicts = _split_guide_chunks(guide_path.stem, raw_text)
        headings = [chunk["heading"] for chunk in chunk_dicts]
        guide_text = "\n".join(headings) + "\n" + raw_text
        guide_docs.append(
            {
                "guide_name": guide_path.stem,
                "title": guide_path.stem.replace("_", " "),
                "text": guide_text,
                "tokens": _tokenize(guide_text),
            }
        )
        for chunk in chunk_dicts:
            searchable = f"{chunk['guide_name']}\n{chunk['heading']}\n{chunk['content']}"
            chunk_docs.append(
                {
                    **chunk,
                    "text": searchable,
                    "tokens": _tokenize(searchable),
                }
            )

    guide_vectors, guide_idf = _build_tfidf([doc["tokens"] for doc in guide_docs]) if guide_docs else ([], {})
    chunk_vectors, chunk_idf = _build_tfidf([doc["tokens"] for doc in chunk_docs]) if chunk_docs else ([], {})

    for index, vector in enumerate(guide_vectors):
        guide_docs[index]["tfidf"] = vector
    for index, vector in enumerate(chunk_vectors):
        chunk_docs[index]["tfidf"] = vector

    return {
        "version": RAG_VERSION,
        "manifest": _guide_manifest(),
        "guides": guide_docs,
        "chunks": chunk_docs,
        "guide_idf": guide_idf,
        "chunk_idf": chunk_idf,
    }


def _save_guide_index(index: dict[str, Any]) -> None:
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAG_INDEX_PATH, "w", encoding="utf-8") as handle:
        json.dump(index, handle)


def _load_guide_index() -> dict[str, Any]:
    global _GUIDE_INDEX_CACHE
    if _GUIDE_INDEX_CACHE is not None:
        return _GUIDE_INDEX_CACHE

    current_manifest = _guide_manifest()
    if not GUIDES_DIR.exists():
        _GUIDE_INDEX_CACHE = {
            "version": RAG_VERSION,
            "manifest": current_manifest,
            "guides": [],
            "chunks": [],
            "guide_idf": {},
            "chunk_idf": {},
        }
        return _GUIDE_INDEX_CACHE

    if RAG_INDEX_PATH.exists():
        try:
            with open(RAG_INDEX_PATH, encoding="utf-8") as handle:
                loaded = json.load(handle)
            if loaded.get("version") == RAG_VERSION and loaded.get("manifest") == current_manifest:
                _GUIDE_INDEX_CACHE = loaded
                return loaded
        except Exception:
            pass

    rebuilt = _build_guide_index()
    _save_guide_index(rebuilt)
    _GUIDE_INDEX_CACHE = rebuilt
    return rebuilt


def _guide_name_tokens(index: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for guide in index.get("guides", []):
        tokens.update(_tokenize(guide.get("guide_name", "")))
        tokens.update(_tokenize(guide.get("title", "")))
    return tokens


def _score_phrases(text: str, phrases: list[str]) -> float:
    text_lower = text.lower()
    score = 0.0
    for phrase in phrases:
        if phrase and phrase in text_lower:
            score += 3.0 if len(phrase.split()) >= 2 else 1.5
    return score


def _score_guide_doc(doc: dict[str, Any], query: str, query_vec: _SearchVector, phrases: list[str], guide_name_hint_tokens: set[str]) -> float:
    title = doc.get("title", "")
    text = doc.get("text", "")
    tokens = set(doc.get("tokens", []))
    score = 8.0 * _cosine_similarity(query_vec, doc.get("tfidf", {}))
    score += _score_phrases(title + "\n" + text[:250], phrases)
    lowered_query = _normalize_text(query)
    title_lower = title.lower()
    if lowered_query and lowered_query in title_lower:
        score += 8.0
    if any(token in tokens for token in guide_name_hint_tokens):
        score += 2.5
    if any(token in tokens for token in _tokenize(query) if token not in _GENERIC_GUIDE_TOKENS):
        score += 1.0
    return score


def _score_chunk_doc(doc: dict[str, Any], query: str, query_vec: _SearchVector, phrases: list[str], guide_prior: float) -> float:
    heading = doc.get("heading", "")
    text = doc.get("text", "")
    tokens = set(doc.get("tokens", []))
    score = guide_prior
    score += 10.0 * _cosine_similarity(query_vec, doc.get("tfidf", {}))
    score += _score_phrases(heading + "\n" + text, phrases)
    lowered_query = _normalize_text(query)
    if lowered_query and lowered_query in heading.lower():
        score += 6.0
    if not any(token in tokens for token in _tokenize(query) if token not in _GENERIC_GUIDE_TOKENS):
        score -= 2.0
    return score


def runSql(query: str, previewRows: int = 20) -> str:
    previewRows = max(1, previewRows)

    if re.search(r"\b(delete|drop|truncate|alter|update|insert|create|replace)\b", query, flags=re.IGNORECASE):
        return f"{SQL_VALID_PREFIX}ERROR\nSafety error: write statements are not allowed."

    validation = database.validate_query(query)
    if not validation.is_valid:
        return f"{SQL_VALID_PREFIX}ERROR\nSyntax error: {validation.error_message}"

    result = database.execute_query(query)
    if not result.is_success:
        return f"{SQL_VALID_PREFIX}ERROR\nExecution error: {result.error_message}"

    df = result.dataframe
    if df is None:
        return f"{SQL_VALID_PREFIX}ERROR\nQuery returned no dataframe."

    preview = df.head(previewRows)
    lines = [
        f"{SQL_VALID_PREFIX}OK",
        f"rowCount={df.height}",
        f"columnCount={df.width}",
        f"columns={', '.join(df.columns)}",
        "preview:",
        preview.write_csv(),
    ]
    return "\n".join(lines)


def listTables(schemaName: str) -> str:
    tables = database.list_tables(schemaName)
    if not tables:
        return f"No tables found for schema '{schemaName}'. Check the schema name."
    return f"Tables in {schemaName}:\n- " + "\n- ".join(tables)


def _format_profile(profile: database.TableColumnProfiles) -> list[str]:
    lines: list[str] = []
    for item in profile.profiles:
        null_pct = (item.null_count / item.row_count * 100.0) if item.row_count else 0.0
        lines.append(f"- {item.column_name} ({item.data_type})")
        lines.append(
            f"  rows={item.row_count}, nulls={item.null_count} ({null_pct:.1f}%), "
            f"approx_distinct={item.approx_distinct if item.approx_distinct is not None else 'unknown'}"
        )
        if item.min_value is not None or item.max_value is not None:
            lines.append(f"  min={item.min_value}, max={item.max_value}")
        if item.top_values:
            formatted = ", ".join(f"{value} ({count})" for value, count in item.top_values)
            lines.append(f"  top_values: {formatted}")
        elif item.example_values:
            lines.append(f"  example_values: {', '.join(item.example_values)}")
    return lines


def describeTable(schemaName: str, tableName: str, focusText: str | None = None) -> str:
    description = database.describe_table(schemaName, tableName, focus_text=focusText)
    if description is None or not description.columns:
        return f"No columns found for {schemaName}.{tableName}. Check schema and table names."

    lines = [f"Columns in {schemaName}.{tableName}:"]
    if description.focused_columns:
        lines.append("Focused columns for this question:")
        for name in description.focused_columns:
            lines.append(f"- __{name}__")

    lines.append("All columns:")
    for column in description.columns:
        display_name = f"__{column.column_name}__" if column.is_focused else column.column_name
        nullable_str = ", nullable" if column.is_nullable else ""
        lines.append(f"- {display_name} ({column.data_type}{nullable_str})")

    if description.suggested_profile_columns:
        lines.append(
            "Potentially ambiguous similarly named columns detected: "
            + ", ".join(description.suggested_profile_columns)
        )
        profile = database.profile_columns(
            schema_name=schemaName,
            table_name=tableName,
            column_names=description.suggested_profile_columns,
        )
        if profile and profile.profiles:
            lines.append("Auto-profiled ambiguous columns:")
            lines.extend(_format_profile(profile))

    return "\n".join(lines)


def searchCatalog(searchText: str, limit: int = 10) -> str:
    matches = database.search_catalog(searchText, limit=limit)
    if not matches:
        return f"No catalog matches found for '{searchText}'. Try different keywords."

    lines = [f"Top catalog matches for '{searchText}':"]
    for index, match in enumerate(matches, start=1):
        columns = ", ".join(match.matched_columns[:8])
        lines.append(f"{index}. {match.schema_name}.{match.table_name} [score={match.score}]")
        if columns:
            lines.append(f"   matched columns: {columns}")
    return "\n".join(lines)


def searchGuides(searchText: str, limit: int = MAX_GUIDE_MATCHES) -> str:
    index = _load_guide_index()
    guides = index.get("guides", [])
    chunks = index.get("chunks", [])
    if not guides or not chunks:
        return "No business-rule guides are available."

    guide_query_vec = _vectorize_query(searchText, index.get("guide_idf", {}))
    chunk_query_vec = _vectorize_query(searchText, index.get("chunk_idf", {}))
    phrases = _phrase_candidates(searchText)
    guide_name_hint_tokens = _guide_name_tokens(index)

    guide_scores: dict[str, float] = {}
    for guide in guides:
        score = _score_guide_doc(guide, searchText, guide_query_vec, phrases, guide_name_hint_tokens)
        if score > 0:
            guide_scores[guide["guide_name"]] = score

    if not guide_scores:
        return f"No guide matches found for '{searchText}'. Try synonyms or narrower keywords."

    top_guides = sorted(guide_scores.items(), key=lambda item: (-item[1], item[0].lower()))[:4]
    top_guide_names = {name for name, _ in top_guides}

    chunk_scores: list[tuple[float, dict[str, Any]]] = []
    for chunk in chunks:
        if chunk["guide_name"] not in top_guide_names:
            continue
        score = _score_chunk_doc(
            chunk,
            searchText,
            chunk_query_vec,
            phrases,
            guide_prior=guide_scores.get(chunk["guide_name"], 0.0),
        )
        if score > 0:
            chunk_scores.append((score, chunk))

    if not chunk_scores:
        return f"No guide matches found for '{searchText}'. Try synonyms or narrower keywords."

    chunk_scores.sort(key=lambda item: (-item[0], item[1]["guide_name"].lower(), item[1]["heading"].lower()))
    lines = [f"Top guide matches for '{searchText}':"]
    seen: set[tuple[str, str]] = set()
    count = 0
    for score, chunk in chunk_scores:
        key = (chunk["guide_name"], chunk["heading"])
        if key in seen:
            continue
        seen.add(key)
        snippet = " ".join(chunk["content"].split())
        if len(snippet) > MAX_GUIDE_SNIPPET_CHARS:
            snippet = snippet[: MAX_GUIDE_SNIPPET_CHARS - 3] + "..."
        lines.append(f"- {chunk['guide_name']} :: {chunk['heading']} [score={score:.2f}]\n  {snippet}")
        count += 1
        if count >= max(1, limit):
            break
    return "\n".join(lines)


def profileColumns(schemaName: str, tableName: str, columnNames: list[str], topK: int = 8) -> str:
    profile = database.profile_columns(schema_name=schemaName, table_name=tableName, column_names=columnNames, top_k=topK)
    if profile is None or not profile.profiles:
        return f"Could not profile columns for {schemaName}.{tableName}. Check the schema, table, and column names."

    lines = [f"Column profile for {profile.schema_name}.{profile.table_name}:"]
    lines.extend(_format_profile(profile))
    return "\n".join(lines)


def previewRows(schemaName: str, tableName: str, limit: int = 5) -> str:
    df = database.preview_rows(schema_name=schemaName, table_name=tableName, limit=limit)
    if df is None:
        return f"Could not preview rows for {schemaName}.{tableName}. Check the schema and table names."
    lines = [
        f"Row preview for {schemaName}.{tableName}:",
        f"rowCount={df.height}",
        f"columnCount={df.width}",
        "preview:",
        df.write_csv(),
    ]
    return "\n".join(lines)


RUN_SQL = agentFramework.Tool(
    name="run_sql",
    description=(
        "Execute a SQL query against the DuckDB database to validate and preview results. "
        "Use this to test your query before calling submit_answer. "
        "Returns row count, column names, and a data preview."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL query to execute. Use schema-qualified table names (e.g. schema.table).",
            },
            "previewRows": {
                "type": "integer",
                "description": "Number of rows to preview (default 20).",
                "default": 20,
            },
        },
        "required": ["query"],
    },
    function=runSql,
)

LIST_TABLES = agentFramework.Tool(
    name="list_tables",
    description="List all tables in a given schema.",
    parameters={
        "type": "object",
        "properties": {"schemaName": {"type": "string", "description": "The schema name."}},
        "required": ["schemaName"],
    },
    function=listTables,
)

DESCRIBE_TABLE = agentFramework.Tool(
    name="describe_table",
    description=(
        "Describe a table's columns and types. Provide focusText when you want relevant columns highlighted. "
        "Focused columns are wrapped like __ColumnName__. Similar candidate metric columns may be auto-profiled."
    ),
    parameters={
        "type": "object",
        "properties": {
            "schemaName": {"type": "string", "description": "The schema name."},
            "tableName": {"type": "string", "description": "The table name."},
            "focusText": {
                "type": "string",
                "description": "Optional question text or keyword hint used to highlight relevant columns.",
            },
        },
        "required": ["schemaName", "tableName"],
    },
    function=describeTable,
)

SEARCH_CATALOG = agentFramework.Tool(
    name="search_catalog",
    description=(
        "Search the database catalog across schemas, tables, and columns to find likely tables and columns for a business concept. "
        "Results are ranked at the table level and favor richer fact tables over lookup tables."
    ),
    parameters={
        "type": "object",
        "properties": {
            "searchText": {
                "type": "string",
                "description": "Keywords describing the business concept, metric, table, or column you are looking for.",
            },
            "limit": {"type": "integer", "description": "Maximum number of catalog matches to return.", "default": 10},
        },
        "required": ["searchText"],
    },
    function=searchCatalog,
)

SEARCH_GUIDES = agentFramework.Tool(
    name="search_guides",
    description=(
        "Retrieve relevant business-rule guide snippets using the persisted guide RAG index in ./.rag/. "
        "Use this whenever a prompt contains business terms like on-time, active, current, completed, rookie, performing, severe, session, legacy, refund, or similar domain language."
    ),
    parameters={
        "type": "object",
        "properties": {
            "searchText": {
                "type": "string",
                "description": "Keywords or a short phrase describing the business rule you need to recover.",
            },
            "limit": {"type": "integer", "description": "Maximum number of guide snippets to return.", "default": 8},
        },
        "required": ["searchText"],
    },
    function=searchGuides,
)

PROFILE_COLUMNS = agentFramework.Tool(
    name="profile_columns",
    description=(
        "Profile candidate columns in a table to inspect null rates, distinct values, common values, and min/max ranges. "
        "Use this when similar columns exist and you need to distinguish an indicator/proxy column from the correct business metric."
    ),
    parameters={
        "type": "object",
        "properties": {
            "schemaName": {"type": "string", "description": "The schema name."},
            "tableName": {"type": "string", "description": "The table name."},
            "columnNames": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One or more column names to profile.",
            },
            "topK": {"type": "integer", "description": "How many top/common values to show for low-cardinality columns.", "default": 8},
        },
        "required": ["schemaName", "tableName", "columnNames"],
    },
    function=profileColumns,
)

PREVIEW_ROWS = agentFramework.Tool(
    name="preview_rows",
    description=(
        "Preview example rows from a table. Use this when column names are cryptic, code-like, or when you need to see example records to interpret columns correctly."
    ),
    parameters={
        "type": "object",
        "properties": {
            "schemaName": {"type": "string", "description": "The schema name."},
            "tableName": {"type": "string", "description": "The table name."},
            "limit": {"type": "integer", "description": "How many rows to preview (default 5).", "default": 5},
        },
        "required": ["schemaName", "tableName"],
    },
    function=previewRows,
)

VALIDATE_SQL_BUNDLE = agentFramework.Tool(
    name="validate_sql_bundle",
    description=(
        "Validate a SQL query: checks syntax, executes it, and previews results. "
        "Use run_sql instead — this is kept for backwards compatibility."
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question being answered (unused, kept for compatibility)."},
            "query": {"type": "string", "description": "SQL query to validate."},
        },
        "required": ["question", "query"],
    },
    function=lambda question="", query="", **_: runSql(query),
)

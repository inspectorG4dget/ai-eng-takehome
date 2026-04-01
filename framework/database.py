"""Database tooling for executing SQL queries against the consolidated DuckDB database.

All CTU Relational databases are consolidated into a single DuckDB file (hecks.duckdb)
with each original database as its own schema. Queries use schema.table syntax.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import duckdb
import polars as pl
import sqlglot
from sqlglot.errors import ParseError

# Path to the consolidated database file
DATABASE_PATH = Path(__file__).parent.parent / "hecks.duckdb"

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
    "per",
    "the",
    "to",
    "what",
    "which",
    "with",
}
_LOOKUP_HINTS = ("l_", "lookup", "group", "groups", "type", "types", "code", "codes", "xref", "dim")
_FACT_HINTS = (
    "transaction",
    "trans",
    "charge",
    "payment",
    "performance",
    "flight",
    "result",
    "results",
    "salary",
    "employee",
    "dept",
    "department",
    "loan",
    "account",
    "order",
    "sale",
)
_NUMERIC_TYPE_HINTS = ("int", "double", "float", "decimal", "numeric", "real", "hugeint", "bigint", "smallint", "tinyint")
_TEMPORAL_TYPE_HINTS = ("date", "time", "timestamp")
_COMMON_SUFFIXES = ("minutes", "minute", "amount", "amt", "flag", "code", "status", "count", "pct", "percentage", "id")


@dataclass
class QueryValidationResult:
    is_valid: bool
    error_message: str | None = None


@dataclass
class CatalogTableMatch:
    schema_name: str
    table_name: str
    score: int
    matched_columns: list[str] = field(default_factory=list)


@dataclass
class ColumnProfile:
    column_name: str
    data_type: str
    row_count: int
    null_count: int
    approx_distinct: int | None
    min_value: str | None = None
    max_value: str | None = None
    top_values: list[tuple[str, int]] = field(default_factory=list)
    example_values: list[str] = field(default_factory=list)


@dataclass
class TableColumnProfiles:
    schema_name: str
    table_name: str
    profiles: list[ColumnProfile] = field(default_factory=list)


@dataclass
class ColumnDescription:
    column_name: str
    data_type: str
    is_nullable: bool
    is_focused: bool = False


@dataclass
class TableDescription:
    schema_name: str
    table_name: str
    columns: list[ColumnDescription] = field(default_factory=list)
    focused_columns: list[str] = field(default_factory=list)
    suggested_profile_columns: list[str] = field(default_factory=list)


@dataclass
class QueryExecutionResult:
    dataframe: pl.DataFrame | None
    error_message: str | None = None

    @property
    def is_success(self) -> bool:
        return self.error_message is None

    @property
    def is_empty(self) -> bool:
        return self.is_success and self.dataframe is not None and self.dataframe.is_empty()


@dataclass(frozen=True)
class _CatalogRow:
    schema_name: str
    table_name: str
    column_name: str
    data_type: str
    is_nullable: str
    ordinal_position: int


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token and token not in _STOPWORDS]


def _identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _is_numeric_type(data_type: str) -> bool:
    return any(hint in data_type.lower() for hint in _NUMERIC_TYPE_HINTS)


def _is_temporal_type(data_type: str) -> bool:
    return any(hint in data_type.lower() for hint in _TEMPORAL_TYPE_HINTS)


def _identifier_tokens(name: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return _tokenize(spaced.replace("_", " "))


def _normalize_identifier_compact(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _strip_common_suffixes(compact_name: str) -> str:
    root = compact_name
    for suffix in _COMMON_SUFFIXES:
        if root.endswith(suffix) and len(root) > len(suffix) + 2:
            root = root[: -len(suffix)]
    root = re.sub(r"\d+$", "", root)
    return root


def _common_prefix_len(a: str, b: str) -> int:
    max_len = min(len(a), len(b))
    idx = 0
    while idx < max_len and a[idx] == b[idx]:
        idx += 1
    return idx


def _pair_is_similar(a: str, b: str) -> bool:
    comp_a = _normalize_identifier_compact(a)
    comp_b = _normalize_identifier_compact(b)
    if not comp_a or not comp_b or a == b:
        return False
    root_a = _strip_common_suffixes(comp_a)
    root_b = _strip_common_suffixes(comp_b)
    if root_a and root_a == root_b and len(root_a) >= 3:
        return True
    if len(root_a) >= 4 and len(root_b) >= 4 and (root_a.startswith(root_b) or root_b.startswith(root_a)):
        return True
    if _common_prefix_len(comp_a, comp_b) >= 5:
        return True
    return False


def _group_similar_columns(columns: list[str], limit: int = 4) -> list[str]:
    if len(columns) < 2:
        return []
    groups: list[list[str]] = []
    for name in columns:
        placed = False
        for group in groups:
            if any(_pair_is_similar(name, existing) for existing in group):
                group.append(name)
                placed = True
                break
        if not placed:
            groups.append([name])
    candidate_groups = [group for group in groups if len(group) >= 2]
    if not candidate_groups:
        return []
    best_group = sorted(candidate_groups, key=lambda g: (-len(g), g[0].lower()))[0]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in best_group:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def _focus_seed_columns(schema_name: str, table_name: str, focus_text: str | None) -> list[str]:
    if not focus_text:
        return []
    matches = search_catalog(focus_text, limit=20)
    for match in matches:
        if match.schema_name == schema_name and match.table_name == table_name:
            return match.matched_columns[:8]
    return []


@lru_cache(maxsize=1)
def _catalog_rows() -> tuple[_CatalogRow, ...]:
    conn: duckdb.DuckDBPyConnection | None = None
    try:
        conn = duckdb.connect(str(DATABASE_PATH), read_only=True)
        rows = conn.execute(
            """
            SELECT table_schema, table_name, column_name, data_type, is_nullable, ordinal_position
            FROM information_schema.columns
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name, ordinal_position
            """
        ).fetchall()
        return tuple(_CatalogRow(*row) for row in rows)
    finally:
        if conn is not None:
            conn.close()


@lru_cache(maxsize=1)
def get_catalog_vocabulary() -> frozenset[str]:
    vocab: set[str] = set()
    for row in _catalog_rows():
        vocab.update(_identifier_tokens(row.schema_name))
        vocab.update(_identifier_tokens(row.table_name))
        vocab.update(_identifier_tokens(row.column_name))
    return frozenset(vocab)


def validate_query(query: str) -> QueryValidationResult:
    try:
        _ = sqlglot.parse_one(query, read="duckdb")
    except ParseError as e:
        return QueryValidationResult(is_valid=False, error_message=str(e))
    return QueryValidationResult(is_valid=True)


def execute_query(query: str) -> QueryExecutionResult:
    conn: duckdb.DuckDBPyConnection | None = None
    try:
        conn = duckdb.connect(str(DATABASE_PATH), read_only=True)
        result = conn.execute(query)
        df = pl.DataFrame(result.fetch_arrow_table())
        return QueryExecutionResult(dataframe=df)
    except duckdb.Error as e:
        return QueryExecutionResult(dataframe=None, error_message=f"DuckDB error: {e}")
    except Exception as e:
        return QueryExecutionResult(dataframe=None, error_message=str(e))
    finally:
        if conn is not None:
            conn.close()


def list_schemas() -> list[str]:
    return sorted({row.schema_name for row in _catalog_rows()})


def list_tables(schema_name: str) -> list[str] | None:
    tables = sorted({row.table_name for row in _catalog_rows() if row.schema_name == schema_name})
    return tables


def _score_identifier_tokens(query_tokens: list[str], identifier_tokens: list[str], compact_name: str) -> int:
    if not query_tokens or not identifier_tokens:
        return 0
    identifier_token_set = set(identifier_tokens)
    score = 0
    for token in query_tokens:
        if token in identifier_token_set:
            score += 10 if len(token) <= 4 else 8
        elif len(token) > 4 and any(id_token.startswith(token) or token.startswith(id_token) for id_token in identifier_token_set):
            score += 3
        elif len(token) > 4 and token in compact_name:
            score += 2
    return score


def search_catalog(search_text: str, limit: int = 10) -> list[CatalogTableMatch]:
    cleaned = _normalize_text(search_text)
    if not cleaned:
        return []

    tokens = _tokenize(search_text)
    if not tokens:
        return []

    grouped_rows: dict[tuple[str, str], list[_CatalogRow]] = defaultdict(list)
    for row in _catalog_rows():
        grouped_rows[(row.schema_name, row.table_name)].append(row)

    ranked_matches: list[CatalogTableMatch] = []
    for (schema_name, table_name), rows in grouped_rows.items():
        schema_tokens = _identifier_tokens(schema_name)
        table_tokens = _identifier_tokens(table_name)
        schema_compact = _normalize_identifier_compact(schema_name)
        table_compact = _normalize_identifier_compact(table_name)

        base_score = 0
        if cleaned == _normalize_text(schema_name):
            base_score += 12
        if cleaned == _normalize_text(table_name):
            base_score += 20
        if cleaned.replace(" ", "") == table_compact:
            base_score += 18

        base_score += _score_identifier_tokens(tokens, schema_tokens, schema_compact)
        base_score += _score_identifier_tokens(tokens, table_tokens, table_compact)

        column_matches: list[tuple[str, int]] = []
        for row in rows:
            column_tokens = _identifier_tokens(row.column_name)
            column_compact = _normalize_identifier_compact(row.column_name)
            col_score = _score_identifier_tokens(tokens, column_tokens, column_compact)
            if cleaned == _normalize_text(row.column_name):
                col_score += 14
            if cleaned.replace(" ", "") == column_compact:
                col_score += 12
            if col_score > 0:
                column_matches.append((row.column_name, col_score))

        if base_score <= 0 and not column_matches:
            continue

        sorted_columns = sorted(column_matches, key=lambda item: (-item[1], item[0].lower()))
        unique_columns: list[str] = []
        seen_cols: set[str] = set()
        top_column_scores: list[int] = []
        for column_name, score in sorted_columns:
            if column_name in seen_cols:
                continue
            seen_cols.add(column_name)
            unique_columns.append(column_name)
            top_column_scores.append(score)
            if len(unique_columns) >= 8:
                break

        table_name_lower = table_name.lower()
        schema_name_lower = schema_name.lower()
        final_score = base_score
        final_score += sum(top_column_scores[:4])
        final_score += min(16, len(unique_columns) * 4)
        if any(hint in table_name_lower for hint in _FACT_HINTS):
            final_score += 8
        if any(hint in table_name_lower for hint in _LOOKUP_HINTS):
            final_score -= 10
        if any(hint in schema_name_lower for hint in _LOOKUP_HINTS):
            final_score -= 4
        if len(unique_columns) >= 5:
            final_score += 6

        ranked_matches.append(
            CatalogTableMatch(
                schema_name=schema_name,
                table_name=table_name,
                score=final_score,
                matched_columns=unique_columns,
            )
        )

    ranked_matches.sort(key=lambda match: (-match.score, match.schema_name.lower(), match.table_name.lower()))
    return ranked_matches[: max(1, limit)]


def describe_table(schema_name: str, table_name: str, focus_text: str | None = None) -> TableDescription | None:
    rows = [row for row in _catalog_rows() if row.schema_name == schema_name and row.table_name == table_name]
    if not rows:
        return None

    seed_columns = _focus_seed_columns(schema_name, table_name, focus_text)
    focus_tokens = _tokenize(focus_text or "")
    focused_columns: list[str] = []
    columns: list[ColumnDescription] = []
    for row in rows:
        col_tokens = _identifier_tokens(row.column_name)
        compact = _normalize_identifier_compact(row.column_name)
        score = 0
        if row.column_name in seed_columns:
            score += 20
        score += _score_identifier_tokens(focus_tokens, col_tokens, compact)
        is_focused = score > 0
        if is_focused:
            focused_columns.append(row.column_name)
        columns.append(
            ColumnDescription(
                column_name=row.column_name,
                data_type=row.data_type,
                is_nullable=(row.is_nullable == "YES"),
                is_focused=is_focused,
            )
        )

    focused_set = set(focused_columns)
    ordered_columns = sorted(
        columns,
        key=lambda col: (
            0 if col.column_name in focused_set else 1,
            next((r.ordinal_position for r in rows if r.column_name == col.column_name), 0),
        ),
    )
    suggested_profile_columns = _group_similar_columns([col for col in focused_columns if col])

    return TableDescription(
        schema_name=schema_name,
        table_name=table_name,
        columns=ordered_columns,
        focused_columns=focused_columns,
        suggested_profile_columns=suggested_profile_columns,
    )


def preview_rows(schema_name: str, table_name: str, limit: int = 5) -> pl.DataFrame | None:
    limit = max(1, min(20, int(limit)))
    rows = [row for row in _catalog_rows() if row.schema_name == schema_name and row.table_name == table_name]
    if not rows:
        return None

    conn: duckdb.DuckDBPyConnection | None = None
    try:
        conn = duckdb.connect(str(DATABASE_PATH), read_only=True)
        qualified_table = f"{_identifier(schema_name)}.{_identifier(table_name)}"
        result = conn.execute(f"SELECT * FROM {qualified_table} LIMIT {limit}")
        return pl.DataFrame(result.fetch_arrow_table())
    except Exception:
        return None
    finally:
        if conn is not None:
            conn.close()


def profile_columns(schema_name: str, table_name: str, column_names: list[str], top_k: int = 8) -> TableColumnProfiles | None:
    requested = [name for name in column_names if name]
    if not requested:
        return None

    rows = [row for row in _catalog_rows() if row.schema_name == schema_name and row.table_name == table_name]
    type_map = {row.column_name: row.data_type for row in rows}
    selected = [name for name in requested if name in type_map]
    if not selected:
        return None

    conn: duckdb.DuckDBPyConnection | None = None
    try:
        conn = duckdb.connect(str(DATABASE_PATH), read_only=True)
        qualified_table = f"{_identifier(schema_name)}.{_identifier(table_name)}"
        profiles: list[ColumnProfile] = []
        for column_name in selected:
            quoted_col = _identifier(column_name)
            data_type = type_map[column_name]
            stats_query = f"""
                SELECT
                    COUNT(*) AS row_count,
                    SUM(CASE WHEN {quoted_col} IS NULL THEN 1 ELSE 0 END) AS null_count,
                    APPROX_COUNT_DISTINCT({quoted_col}) AS approx_distinct,
                    CAST(MIN({quoted_col}) AS VARCHAR) AS min_value,
                    CAST(MAX({quoted_col}) AS VARCHAR) AS max_value
                FROM {qualified_table}
            """
            row_count, null_count, approx_distinct, min_value, max_value = conn.execute(stats_query).fetchone()

            top_values: list[tuple[str, int]] = []
            example_values: list[str] = []
            should_show_top_values = (
                approx_distinct is not None and approx_distinct <= max(10, top_k * 2)
            ) or (not _is_numeric_type(data_type) and not _is_temporal_type(data_type))

            if should_show_top_values:
                value_rows = conn.execute(
                    f"""
                    SELECT CAST({quoted_col} AS VARCHAR) AS value, COUNT(*) AS frequency
                    FROM {qualified_table}
                    WHERE {quoted_col} IS NOT NULL
                    GROUP BY 1
                    ORDER BY frequency DESC, value
                    LIMIT {max(1, top_k)}
                    """
                ).fetchall()
                top_values = [(str(value), int(frequency)) for value, frequency in value_rows]
            else:
                example_rows = conn.execute(
                    f"""
                    SELECT DISTINCT CAST({quoted_col} AS VARCHAR) AS value
                    FROM {qualified_table}
                    WHERE {quoted_col} IS NOT NULL
                    LIMIT {max(1, min(5, top_k))}
                    """
                ).fetchall()
                example_values = [str(value) for (value,) in example_rows]

            if not (_is_numeric_type(data_type) or _is_temporal_type(data_type)):
                min_value = None
                max_value = None

            profiles.append(
                ColumnProfile(
                    column_name=column_name,
                    data_type=data_type,
                    row_count=int(row_count or 0),
                    null_count=int(null_count or 0),
                    approx_distinct=(int(approx_distinct) if approx_distinct is not None else None),
                    min_value=min_value,
                    max_value=max_value,
                    top_values=top_values,
                    example_values=example_values,
                )
            )

        return TableColumnProfiles(schema_name=schema_name, table_name=table_name, profiles=profiles)
    except Exception:
        return None
    finally:
        if conn is not None:
            conn.close()

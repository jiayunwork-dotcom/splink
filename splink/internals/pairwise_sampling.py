"""Stratified sampling of pairwise predictions for quality evaluation.

The review budget for clerically checking predicted record pairs is limited
and deliberately uneven across the score range.  This module implements a
reproducible, label-agnostic stratified sampling design over a table of
generated pairwise predictions so that later, weighted evaluation can
estimate match quality over the whole *generated* prediction population.

Scope / population boundary
----------------------------
The population is exactly the prediction table handed in by the caller.
Splink only generates pairs that pass its blocking rules, so nothing here
says anything about pairs that were never generated, and no confidence
intervals are produced.

Design summary
--------------
* strata are half-open intervals of ``match_probability`` defined by the
  caller-supplied ``score_bins`` (the final stratum includes ``1``);
* every non-empty stratum is guaranteed at least one sampled pair;
* the remaining budget is apportioned proportionally to stratum size using
  the largest-remainder (Hamilton) method - decided before any label exists,
  independent of row order, table name and the clerical outcomes;
* within a stratum, simple random sampling without replacement at equal
  inclusion probability ``n_h / N_h`` is realised by ranking canonical,
  seed-salted hashes with a deterministic tie breaker (see
  ``docs/topic_guides/evaluation/weighted_sampling_design.md`` for the
  comparison with ``random()`` ordering, Bernoulli and reservoir designs).

All pair-level data stays inside DuckDB.  Python only ever receives the
per-stratum counts and the final sample (at most ``budget`` rows).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from splink.internals.duckdb.database_api import DuckDBAPI
from splink.internals.misc import ascii_uid
from splink.internals.pipeline import CTEPipeline
from splink.internals.splink_dataframe import SplinkDataFrame
from splink.internals.vertically_concatenate import compute_df_concat_with_tf

if TYPE_CHECKING:
    from splink.internals.linker import Linker

# Fixed output metadata columns.  Leading double underscores are avoided here
# because these columns are part of the public sampling output and must be
# safe to hand back to the user / register as an ordinary table.
SAMPLE_STRATUM_COLUMN = "sample_stratum"
STRATUM_POPULATION_COLUMN = "stratum_population"
STRATUM_SAMPLE_SIZE_COLUMN = "stratum_sample_size"
INCLUSION_PROBABILITY_COLUMN = "inclusion_probability"

SAMPLING_METADATA_COLUMNS = [
    SAMPLE_STRATUM_COLUMN,
    STRATUM_POPULATION_COLUMN,
    STRATUM_SAMPLE_SIZE_COLUMN,
    INCLUSION_PROBABILITY_COLUMN,
]

# Max seed is exclusive: valid seeds are ints in [0, 2**32).
MAX_SEED = 2**32


@dataclass(frozen=True)
class StratumAllocation:
    """Per-stratum sampling bookkeeping (small: one instance per stratum)."""

    stratum: int
    population: int
    sample_size: int

    @property
    def inclusion_probability(self) -> float:
        return self.sample_size / self.population


def _is_real_number(value: Any) -> bool:
    """A real (non-bool) Python int or float, excluding booleans."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_sampling_arguments(
    score_bins: Any, budget: Any, seed: Any
) -> list[float]:
    if not isinstance(score_bins, list) or len(score_bins) < 2:
        raise ValueError(
            "score_bins must be a list of at least two real numbers with "
            "first element 0 and last element 1"
        )
    bins: list[float] = []
    for b in score_bins:
        if not _is_real_number(b):
            raise ValueError(
                "score_bins must contain only real numbers (int or float, "
                "not bool)"
            )
        b = float(b)
        if not math.isfinite(b):
            raise ValueError("score_bins must contain only finite numbers")
        bins.append(b)

    if bins[0] != 0.0 or bins[-1] != 1.0:
        raise ValueError("score_bins must start at 0 and end at 1")
    if any(bins[i] >= bins[i + 1] for i in range(len(bins) - 1)):
        raise ValueError(
            "score_bins must be strictly increasing between 0 and 1"
        )

    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError("budget must be a positive, non-bool Python int")

    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not (0 <= seed < MAX_SEED)
    ):
        raise ValueError(
            f"seed must be a non-bool Python int in the range [0, {MAX_SEED})"
        )

    return bins


def _identity_expr(
    source_dataset_col: str | None, unique_id_col: str, side: str
) -> str:
    """Full-identity SQL expression for one side of a pairwise row.

    Mirrors :func:`splink.internals.labelled_pairwise._label_identity_sql`
    so sampling and label normalisation share exactly one identity notion.
    """
    uid = f"{unique_id_col}_{side}"
    if source_dataset_col is not None:
        sds = f"{source_dataset_col}_{side}"
        return f"(cast({sds} as varchar) || '-__-' || cast({uid} as varchar))"
    return f"cast({uid} as varchar)"


def _stratum_case_sql(score_expr: str, bins: list[float]) -> str:
    """Deterministic half-open stratum index expression.

    ``[b_h, b_{h+1})`` for every stratum except the last, which additionally
    contains the upper endpoint ``1``.  Bins are rendered as explicit float
    literals so the SQL is self-contained and independent of row order.
    """
    whens = []
    for h in range(len(bins) - 1):
        low = bins[h]
        high = bins[h + 1]
        if h == len(bins) - 2:
            # final stratum is closed at the top (contains score == 1)
            cond = f"{score_expr} >= {low} and {score_expr} <= {high}"
        else:
            cond = f"{score_expr} >= {low} and {score_expr} < {high}"
        whens.append(f"when {cond} then {h}")
    return "case\n" + "\n".join(whens) + "\nend"


def largest_remainder_allocation(
    stratum_populations: list[int], budget: int
) -> list[int]:
    """Allocate ``budget`` draws across strata, each capped at its population.

    Every non-empty stratum first receives one slot (the caller rejects the
    call when the budget is smaller than the number of non-empty strata).
    Remaining slots are apportioned proportionally to stratum population using
    the largest-remainder (Hamilton) method.  Strata that would be allocated
    more than their population are saturated and their unspent slots are
    redistributed among the remaining strata under the same rule.

    The result never depends on input row order: ties in fractional
    remainders resolve by ascending stratum index.
    """
    populations = list(stratum_populations)
    n_strata = len(populations)
    total_population = sum(populations)
    if budget >= total_population:
        return list(populations)

    allocations = [0] * n_strata
    active = [h for h, n_h in enumerate(populations) if n_h > 0]

    # Floor: one pair per non-empty stratum.
    for h in active:
        allocations[h] = 1
    remaining = budget - len(active)

    # Iteratively apply largest-remainder apportionment, saturating any
    # stratum whose proportional share exceeds its population and
    # redistributing its unspent slots.
    saturated: set[int] = set()
    while remaining > 0 and active:
        active = [h for h in active if allocations[h] < populations[h]]
        if not active:
            break
        active_population = sum(populations[h] - allocations[h] for h in active)
        if active_population <= remaining:
            for h in active:
                allocations[h] = populations[h]
            remaining -= active_population
            break

        # Proportional shares of the remaining budget over remaining capacity.
        raw = {}
        for h in active:
            capacity = populations[h] - allocations[h]
            raw[h] = remaining * capacity / active_population
        floors = {h: int(math.floor(raw[h])) for h in active}
        # never hand out more capacity than exists
        floors = {h: min(f, populations[h] - allocations[h]) for h, f in floors.items()}
        handed_out = sum(floors.values())
        leftover = remaining - handed_out

        # Largest fractional remainders first; ties broken by stratum index.
        order = sorted(active, key=lambda h: (- (raw[h] - math.floor(raw[h])), h))
        extras: dict[int, int] = {h: 0 for h in active}
        for h in order:
            if leftover == 0:
                break
            if allocations[h] + floors[h] + extras[h] < populations[h]:
                extras[h] += 1
                leftover -= 1

        newly_saturated = []
        for h in active:
            add = floors[h] + extras[h]
            allocations[h] += add
            remaining -= add
            if allocations[h] == populations[h]:
                newly_saturated.append(h)
        saturated.update(newly_saturated)

        if not newly_saturated and leftover == 0:
            break
        # If something could not be allocated (only possible when capped),
        # loop again with the remaining budget over remaining capacity.

    return allocations


def sample_pairwise_predictions(
    linker: "Linker",
    predictions: str | SplinkDataFrame,
    *,
    budget: int,
    score_bins: list[float],
    seed: int = 0,
) -> SplinkDataFrame:
    """Stratified, reproducible sampling of pairwise predictions.

    See the module docstring and
    ``docs/topic_guides/evaluation/weighted_sampling_design.md`` for the full
    design, the allocation rationale and the population boundary.
    """
    if not isinstance(linker._db_api, DuckDBAPI):
        raise ValueError(
            "sample_pairwise_predictions is currently only implemented for "
            "the DuckDB backend"
        )

    bins = _validate_sampling_arguments(score_bins, budget, seed)

    db_api = linker._db_api
    settings = linker._settings_obj
    column_info = settings.column_info_settings
    unique_id_col = column_info.unique_id_column_name
    source_dataset_col = column_info.source_dataset_column_name

    if isinstance(predictions, SplinkDataFrame):
        input_name = predictions.physical_name
    elif isinstance(predictions, str):
        input_name = predictions
    else:
        raise ValueError(
            "predictions must be a SplinkDataFrame or the name of a "
            "registered table"
        )

    if not db_api.table_exists_in_database(input_name):
        raise ValueError(f"Predictions table '{input_name}' does not exist")

    input_sdf = linker._table_to_splink_dataframe(input_name, input_name)
    raw_columns = [c.unquote().name for c in input_sdf.columns]

    uid_l = f"{unique_id_col}_l"
    uid_r = f"{unique_id_col}_r"
    required = [uid_l, uid_r, "match_probability", "match_weight"]
    if source_dataset_col is not None:
        sds_l = f"{source_dataset_col}_l"
        sds_r = f"{source_dataset_col}_r"
        required = [sds_l, sds_r, *required]
    missing = [c for c in required if c not in raw_columns]
    if missing:
        raise ValueError(
            "Predictions table is missing required column(s): "
            + ", ".join(missing)
        )

    con = db_api._con
    run_uid = ascii_uid(8)
    canonical_table = f"__splink__sampling_canonical_{run_uid}"
    stratum_table = f"__splink__sampling_strata_{run_uid}"
    output_table = f"__splink__sampled_predictions_{run_uid}"
    temp_tables = [canonical_table, stratum_table, output_table]

    def _cleanup() -> None:
        for name in temp_tables:
            try:
                db_api.delete_table_from_database(name)
            except Exception:
                # Cleanup must never mask the original error.
                pass

    try:
        id_l = _identity_expr(source_dataset_col, unique_id_col, "l")
        id_r = _identity_expr(source_dataset_col, unique_id_col, "r")
        key_l = f"least({id_l}, {id_r})"
        key_r = f"greatest({id_l}, {id_r})"
        score_expr = "cast(match_probability as float8)"
        stratum_expr = _stratum_case_sql(score_expr, bins)

        # Empty population: return the full output schema, zero rows.
        n_rows = con.sql(f"select count(*) from {input_name}").fetchall()[0][0]
        if n_rows == 0:
            con.sql(
                f"""
                create table {output_table} as
                select *,
                    cast(null as bigint) as {SAMPLE_STRATUM_COLUMN},
                    cast(null as bigint) as {STRATUM_POPULATION_COLUMN},
                    cast(null as bigint) as {STRATUM_SAMPLE_SIZE_COLUMN},
                    cast(null as float8) as {INCLUSION_PROBABILITY_COLUMN}
                from {input_name}
                where false
                """
            )
            empty_sdf = db_api.table_to_splink_dataframe(
                "__splink__sampled_predictions", output_table
            )
            empty_sdf.created_by_splink = True
            return empty_sdf

        # --- SQL-side validation: all checks are aggregate scans returning a
        # single row; nothing is pulled into Python but the counts. ---
        validation_sql = f"""
            select
                sum(case when {score_expr} is null
                         or {score_expr} <> {score_expr}
                         or abs({score_expr}) = 1.0/0.0
                         or {score_expr} < 0.0 or {score_expr} > 1.0
                    then 1 else 0 end) as n_bad_score,
                sum(case when {id_l} is null or {id_r} is null
                    then 1 else 0 end) as n_null_id,
                sum(case when {id_l} = {id_r} then 1 else 0 end) as n_self_pair,
                sum(case when {stratum_expr} is null then 1 else 0 end)
                    as n_unbinned
            from {input_name}
        """
        (
            n_bad_score,
            n_null_id,
            n_self_pair,
            n_unbinned,
        ) = con.sql(validation_sql).fetchall()[0]
        if n_bad_score:
            raise ValueError(
                f"match_probability must be finite and within [0, 1]; "
                f"{n_bad_score} offending row(s)"
            )
        if n_null_id:
            raise ValueError(
                f"Pair identity columns must not contain nulls; "
                f"{n_null_id} offending row(s)"
            )
        if n_self_pair:
            raise ValueError(
                f"Self-pairs (the same record on both sides) are not allowed; "
                f"{n_self_pair} offending row(s)"
            )
        if n_unbinned:
            raise ValueError(
                "Internal error: some match_probability values did not map to "
                "a score stratum"
            )

        # Canonicalise orientation once; the table is materialised so every
        # later step shares one canonical representation.
        con.sql(
            f"""
            create table {canonical_table} as
            select *,
                {key_l} as __splink_canonical_key_l,
                {key_r} as __splink_canonical_key_r,
                {stratum_expr} as {SAMPLE_STRATUM_COLUMN}
            from {input_name}
            """
        )

        # Duplicate unoriented pairs (including reversed duplicates).
        n_duplicates = con.sql(
            f"""
            select count(*) from (
                select __splink_canonical_key_l, __splink_canonical_key_r
                from {canonical_table}
                group by 1, 2
                having count(*) > 1
            )
            """
        ).fetchall()[0][0]
        if n_duplicates:
            raise ValueError(
                f"Duplicate unoriented pairs are not allowed; {n_duplicates} "
                "pair(s) appear more than once (including reversed pairs)"
            )

        # Unknown nodes: every canonical identity must exist in the concat
        # table of input records.
        pipeline = CTEPipeline()
        nodes_with_tf = compute_df_concat_with_tf(linker, pipeline)
        node_id = _node_identity_from_settings(settings)
        missing_check_sql = f"""
            with referenced as (
                select __splink_canonical_key_l as k from {canonical_table}
                union
                select __splink_canonical_key_r as k from {canonical_table}
            ),
            nodes as (
                select distinct {node_id} as k
                from {nodes_with_tf.physical_name}
            )
            select count(*) from referenced r
            left join nodes n on r.k = n.k
            where n.k is null
        """
        n_unknown = con.sql(missing_check_sql).fetchall()[0][0]
        if n_unknown:
            raise ValueError(
                f"{n_unknown} pair row(s) reference records that do not exist "
                "in the linker's input data"
            )

        # --- Stratum populations (one small row per stratum into Python). ---
        count_records = con.sql(
            f"""
            select {SAMPLE_STRATUM_COLUMN} as stratum,
                   count(*) as population
            from {canonical_table}
            group by {SAMPLE_STRATUM_COLUMN}
            order by stratum
            """
        ).fetchall()
        non_empty_strata = [int(h) for h, _ in count_records]
        populations = {int(h): int(pop) for h, pop in count_records}
        n_non_empty = len(non_empty_strata)

        if budget < n_non_empty:
            raise ValueError(
                f"budget ({budget}) is smaller than the number of non-empty "
                f"strata ({n_non_empty}); every non-empty stratum must "
                "receive at least one sampled pair"
            )

        total_population = sum(populations.values())
        sample_target = min(budget, total_population)
        ordered_populations = [populations[h] for h in non_empty_strata]
        ordered_sizes = largest_remainder_allocation(
            ordered_populations, sample_target
        )
        allocations = {
            h: StratumAllocation(
                stratum=h,
                population=populations[h],
                sample_size=ordered_sizes[i],
            )
            for i, h in enumerate(non_empty_strata)
        }

        # Materialise the small allocation table inside DuckDB so the final
        # join needs no Python-side parameter expansion over pairs.
        values_sql = ", ".join(
            f"({a.stratum}, {a.population}, {a.sample_size}, "
            f"cast({a.sample_size} as float8) / cast({a.population} as float8))"
            for a in allocations.values()
        )
        con.sql(
            f"""
            create table {stratum_table} as
            select * from (values {values_sql})
            as t({SAMPLE_STRATUM_COLUMN}, {STRATUM_POPULATION_COLUMN},
                 {STRATUM_SAMPLE_SIZE_COLUMN}, {INCLUSION_PROBABILITY_COLUMN})
            """
        )

        # --- Within-stratum equal-probability SRSWOR via seeded, canonical
        # hash ranking with a total deterministic tie breaker. ---
        hash_expr = (
            f"hash(cast({seed} as varchar) || chr(1) || "
            "__splink_canonical_key_l || chr(1) || __splink_canonical_key_r)"
        )
        con.sql(
            f"""
            create table {output_table} as
            with ranked as (
                select *,
                    row_number() over (
                        partition by {SAMPLE_STRATUM_COLUMN}
                        order by {hash_expr},
                                 __splink_canonical_key_l,
                                 __splink_canonical_key_r
                    ) as __splink_within_stratum_rank
                from {canonical_table}
            )
            select
                r.* exclude (__splink_within_stratum_rank,
                             __splink_canonical_key_l,
                             __splink_canonical_key_r),
                s.{STRATUM_POPULATION_COLUMN},
                s.{STRATUM_SAMPLE_SIZE_COLUMN},
                s.{INCLUSION_PROBABILITY_COLUMN}
            from ranked r
            inner join {stratum_table} s
                on r.{SAMPLE_STRATUM_COLUMN} = s.{SAMPLE_STRATUM_COLUMN}
            where r.__splink_within_stratum_rank
                <= s.{STRATUM_SAMPLE_SIZE_COLUMN}
            """
        )

        sampled = db_api.table_to_splink_dataframe(
            "__splink__sampled_predictions", output_table
        )
        sampled.created_by_splink = True
        sampled.sql_used_to_create = None
        # Canonical / stratum scratch tables are ours and no longer needed;
        # the output table is the returned SplinkDataFrame and is kept.
        db_api.delete_table_from_database(canonical_table)
        db_api.delete_table_from_database(stratum_table)
        temp_tables = [output_table]
        return sampled
    except Exception:
        _cleanup()
        raise


def _node_identity_from_settings(settings) -> str:
    """Full-identity expression over the linker's concat table (no suffix)."""
    column_info = settings.column_info_settings
    unique_id_col = column_info.unique_id_input_column.name
    source_dataset_col = column_info.source_dataset_input_column
    if source_dataset_col is not None:
        return (
            f"cast({source_dataset_col.name} as varchar) || '-__-' || "
            f"cast({unique_id_col} as varchar)"
        )
    return f"cast({unique_id_col} as varchar)"

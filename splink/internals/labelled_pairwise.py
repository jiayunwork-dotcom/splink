"""Handling of weighted clerical pairwise labels.

This module contains the shared machinery used by both the direct
``estimate_m_from_pairwise_labels`` path and the expectation-maximisation path
so that the same piece of clerical evidence always has the same meaning.

A normalised label is an unoriented pair of *fully identified* records (so
source dataset and unique id together identify each side) together with a
``clerical_match_score`` in ``[0, 1]`` (the proportion of matching quality,
i.e. 1 = confirmed match, 0 = confirmed non-match, fractional values retain
uncertainty) and a positive ``label_weight`` (a frequency / importance
weight).

All normalisation and validation is performed inside the SQL backend - no
pairwise data is pulled into Python.  Only small, parameter-level summaries
are ever returned to Python.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from splink.internals.pipeline import CTEPipeline

if TYPE_CHECKING:
    from splink.internals.linker import Linker
    from splink.internals.splink_dataframe import SplinkDataFrame

logger = logging.getLogger(__name__)

# Internal, fixed column names used for the normalised label table.  These
# never collide with user data because of the leading double underscore.
LABEL_WEIGHT_COLUMN = "__splink_label_weight"
LABEL_SCORE_COLUMN = "clerical_match_score"
LABEL_KEY_L = "__splink_label_key_l"
LABEL_KEY_R = "__splink_label_key_r"

# Columns that must exist on a label table when score mode is switched on.
SCORE_MODE_REQUIRED_COLUMNS = [
    "unique_id_l",
    "unique_id_r",
]
SOURCE_DATASET_LABEL_COLUMNS = [
    "source_dataset_l",
    "source_dataset_r",
]


@dataclass
class LabelTrainingSummary:
    """Small, parameter-level diagnostics for a labelled training run.

    Nothing here contains the per-pair labels themselves - only aggregate
    counts and masses, so this is safe to attach to training sessions and
    serialise into logs (but not model JSON).
    """

    labelled_pair_count: int
    unlabelled_pair_count: Optional[int]
    label_weight_sum: float
    positive_mass_sum: float
    negative_mass_sum: float
    stop_reason: str
    not_estimated: Optional[list[dict[str, str]]] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "labelled_pair_count": self.labelled_pair_count,
            "unlabelled_pair_count": self.unlabelled_pair_count,
            "label_weight_sum": self.label_weight_sum,
            "positive_mass_sum": self.positive_mass_sum,
            "negative_mass_sum": self.negative_mass_sum,
            "stop_reason": self.stop_reason,
            "not_estimated": self.not_estimated or [],
        }


def _label_identity_sql(
    source_dataset_col: Optional[str],
    unique_id_col: str,
    side: str,
) -> str:
    """SQL expression for the full identity of one side of a label.

    The source dataset is part of the identity so that equal unique ids in
    different datasets cannot be confused.  ``cast(... as varchar)`` keeps the
    comparison robust to integer / string id column types.
    """
    uid_col = f"{unique_id_col}_{side}"
    if source_dataset_col is not None:
        sds_col = f"{source_dataset_col}_{side}"
        return f"cast({sds_col} as varchar) || '-__-' || cast({uid_col} as varchar)"
    return f"cast({uid_col} as varchar)"


def normalise_pairwise_labels(
    linker: "Linker",
    labels_tablename: str,
    *,
    use_label_scores: bool,
    label_weight_column: Optional[str],
) -> tuple[Optional["SplinkDataFrame"], LabelTrainingSummary]:
    """Validate and normalise a table of pairwise labels.

    The returned SplinkDataFrame (a materialised table in the SQL backend)
    has one row per distinct unoriented record pair and the fixed columns
    ``__splink_label_key_l``, ``__splink_label_key_r`` (varchar composite
    identities, with the smaller key on the left), ``clerical_match_score``
    (float in [0, 1]) and ``__splink_label_weight`` (positive float).

    If ``use_label_scores`` is False this function performs no work and
    returns ``(None, empty_summary)`` so callers keep the legacy behaviour.

    Raises:
        ValueError: for any contract violation (missing columns, illegal
            scores/weights, self pairs, pairs violating link_type, unknown
            records, conflicting duplicate evidence, ...).
    """
    if not use_label_scores:
        if label_weight_column is not None:
            raise ValueError(
                "label_weight_column can only be supplied when "
                "use_label_scores=True"
            )
        return None, LabelTrainingSummary(
            labelled_pair_count=0,
            unlabelled_pair_count=None,
            label_weight_sum=0.0,
            positive_mass_sum=0.0,
            negative_mass_sum=0.0,
            stop_reason="not_used",
        )

    db_api = linker._db_api
    settings = linker._settings_obj
    column_info = settings.column_info_settings
    unique_id_col = column_info.unique_id_column_name
    source_dataset_col = (
        column_info.source_dataset_column_name
        if column_info.source_dataset_column_name is not None
        else None
    )
    link_type = settings._link_type

    labels_df = linker._table_to_splink_dataframe(
        labels_tablename, labels_tablename
    )
    raw_colnames = [c.unquote().name for c in labels_df.columns]

    # --- column presence checks (raised from Python, with clear messages) ---
    required = [f"{unique_id_col}_l", f"{unique_id_col}_r", LABEL_SCORE_COLUMN]
    if source_dataset_col is not None:
        required += [f"{source_dataset_col}_l", f"{source_dataset_col}_r"]
    missing = [c for c in required if c not in raw_colnames]
    if missing:
        raise ValueError(
            "Pairwise label table is missing required column(s): "
            f"{', '.join(missing)}"
        )
    if label_weight_column is not None and label_weight_column not in raw_colnames:
        raise ValueError(
            f"label_weight_column '{label_weight_column}' not found on the "
            "pairwise label table"
        )

    weight_expr = (
        f"cast({label_weight_column} as float8)"
        if label_weight_column is not None
        else "cast(1.0 as float8)"
    )

    id_l = _label_identity_sql(source_dataset_col, unique_id_col, "l")
    id_r = _label_identity_sql(source_dataset_col, unique_id_col, "r")

    # An empty (but schema-valid) label table is allowed for EM (means: behave
    # exactly as if no labels were supplied).  For direct m estimation the
    # later "no positive mass" check raises a ValueError.
    count_sql = f"select count(*) as n from {labels_df.physical_name}"
    n_rows = db_api._execute_sql_against_backend(count_sql).fetchall()[0][0]
    if n_rows == 0:
        empty_summary = LabelTrainingSummary(
            labelled_pair_count=0,
            unlabelled_pair_count=None,
            label_weight_sum=0.0,
            positive_mass_sum=0.0,
            negative_mass_sum=0.0,
            stop_reason="empty_label_table",
        )
        return None, empty_summary

    # Everything from here on is SQL-side validation + normalisation.  Each
    # check is a separate small query returning at most a handful of rows.
    def _fail_count(check_sql: str, message: str) -> None:
        n_bad = db_api._execute_sql_against_backend(check_sql).fetchall()[0][0]
        if n_bad > 0:
            raise ValueError(f"{message} ({n_bad} offending row(s))")

    score_expr = f"cast({LABEL_SCORE_COLUMN} as float8)"
    # Portable finiteness checks.  NaN is the only value not equal to itself;
    # 1.0/0.0 is the positive-infinity literal each supported backend renders
    # for its own dialect (avoids dialect-specific 'double infinity' syntax).
    nan_test = "{e} <> {e}"
    inf_test = "abs({e}) = 1.0/0.0"
    _fail_count(
        f"""
        select count(*) from {labels_df.physical_name}
        where {score_expr} is null
            or {nan_test.format(e=score_expr)}
            or {inf_test.format(e=score_expr)}
            or {score_expr} < 0.0 or {score_expr} > 1.0
        """,
        "clerical_match_score must be a finite numeric value between 0 and 1",
    )
    _fail_count(
        f"""
        select count(*) from {labels_df.physical_name}
        where {weight_expr} is null
            or {nan_test.format(e=weight_expr)}
            or {inf_test.format(e=weight_expr)}
            or {weight_expr} <= 0.0
        """,
        "label weights must be finite, strictly positive numbers",
    )
    _fail_count(
        f"""
        select count(*) from {labels_df.physical_name}
        where {id_l} = {id_r}
        """,
        "Self-pairs (the same record on both sides) are not allowed in "
        "pairwise labels",
    )
    if link_type == "link_only":
        # link_only: pairs must span different source datasets
        _fail_count(
            f"""
            select count(*) from {labels_df.physical_name}
            where {f'{source_dataset_col}_l'} = {f'{source_dataset_col}_r'}
            """,
            f"link_type '{link_type}' requires labels to reference different "
            "source datasets on the left and right",
        )
    if link_type == "dedupe_only":
        # dedupe_only: there is a single dataset.  If source dataset columns
        # are present they must agree.
        if source_dataset_col is not None:
            _fail_count(
                f"""
                select count(*) from {labels_df.physical_name}
                where {f'{source_dataset_col}_l'} <> {f'{source_dataset_col}_r'}
                """,
                f"link_type '{link_type}' requires labels within a single "
                "source dataset",
            )

    # Canonicalise orientation: smaller full identity goes to the left.
    key_l = f"least({id_l}, {id_r})"
    key_r = f"greatest({id_l}, {id_r})"

    # Check every referenced record exists in the (cached) concat table.
    pipeline = CTEPipeline()
    from splink.internals.vertically_concatenate import compute_df_concat_with_tf

    nodes_with_tf = compute_df_concat_with_tf(linker, pipeline)
    concat_name = nodes_with_tf.physical_name
    node_id = _node_identity_sql(settings)

    # Build the distinct set of referenced identities as its own CTE first, so
    # the resulting SQL never contains a nested leading WITH.
    pipeline = CTEPipeline([nodes_with_tf])
    pipeline.enqueue_sql(
        f"""
        select distinct k from (
            select {key_l} as k from {labels_df.physical_name}
            union
            select {key_r} as k from {labels_df.physical_name}
        )
        """,
        "__splink__label_referenced_keys",
    )
    pipeline.enqueue_sql(
        f"""
        select count(*) as n_missing
        from __splink__label_referenced_keys as lab
        left join (
            select distinct {node_id} as k from {concat_name}
        ) nodes
        on lab.k = nodes.k
        where nodes.k is null
        """,
        "__splink__label_missing_records_check",
    )
    missing_records = db_api.sql_pipeline_to_splink_dataframe(pipeline)
    n_missing = missing_records.as_record_dict()[0]["n_missing"]
    missing_records.drop_table_from_database_and_remove_from_cache()
    if n_missing > 0:
        raise ValueError(
            f"{n_missing} label row(s) reference records that do not exist in "
            "the input data"
        )

    # Normalise: one row per (canonical pair, score, weight) first, so that
    # identical repeated submissions simply add frequency mass.  Then detect
    # conflicting evidence for the same canonical pair.
    pipeline = CTEPipeline([nodes_with_tf])
    pipeline.enqueue_sql(
        f"""
        select
            {key_l} as {LABEL_KEY_L},
            {key_r} as {LABEL_KEY_R},
            {score_expr} as {LABEL_SCORE_COLUMN},
            {weight_expr} as {LABEL_WEIGHT_COLUMN}
        from {labels_df.physical_name}
        """,
        "__splink__labels_canonical_raw",
    )
    pipeline.enqueue_sql(
        f"""
        select
            {LABEL_KEY_L},
            {LABEL_KEY_R},
            {LABEL_SCORE_COLUMN},
            sum({LABEL_WEIGHT_COLUMN}) as {LABEL_WEIGHT_COLUMN}
        from __splink__labels_canonical_raw
        group by 1, 2, 3
        """,
        "__splink__labels_canonical_grouped",
    )
    pipeline.enqueue_sql(
        f"""
        select {LABEL_KEY_L}, {LABEL_KEY_R},
               count(*) as n_distinct_evidences
        from __splink__labels_canonical_grouped
        group by 1, 2
        having count(*) > 1
        """,
        "__splink__label_conflict_check",
    )
    conflicts = db_api.sql_pipeline_to_splink_dataframe(pipeline)
    n_conflicts = len(conflicts.as_pandas_dataframe())
    conflicts.drop_table_from_database_and_remove_from_cache()
    if n_conflicts > 0:
        raise ValueError(
            f"Conflicting clerical evidence for {n_conflicts} record pair(s): "
            "the same pair was supplied with different clerical_match_score "
            "and/or label_weight values"
        )

    # Materialise the final, deduplicated normalised label table.  Conflicts
    # have already been rejected, so grouping by the canonical pair alone is
    # safe and the score is constant within each group.
    pipeline = CTEPipeline([nodes_with_tf])
    pipeline.enqueue_sql(
        f"""
        select
            {key_l} as {LABEL_KEY_L},
            {key_r} as {LABEL_KEY_R},
            min({score_expr}) as {LABEL_SCORE_COLUMN},
            sum({weight_expr}) as {LABEL_WEIGHT_COLUMN}
        from {labels_df.physical_name}
        group by 1, 2
        """,
        "__splink__labels_normalised",
    )
    normalised = db_api.sql_pipeline_to_splink_dataframe(pipeline)

    summary_records = db_api._execute_sql_against_backend(
        f"""
        select
            count(*) as labelled_pair_count,
            coalesce(sum({LABEL_WEIGHT_COLUMN}), 0.0) as label_weight_sum,
            coalesce(
                sum({LABEL_WEIGHT_COLUMN} * {LABEL_SCORE_COLUMN}), 0.0
            ) as positive_mass_sum,
            coalesce(
                sum({LABEL_WEIGHT_COLUMN} * (1.0 - {LABEL_SCORE_COLUMN})), 0.0
            ) as negative_mass_sum
        from {normalised.physical_name}
        """
    ).fetchall()
    labelled_pair_count, weight_sum, pos_mass, neg_mass = summary_records[0]

    summary = LabelTrainingSummary(
        labelled_pair_count=int(labelled_pair_count),
        unlabelled_pair_count=None,
        label_weight_sum=float(weight_sum),
        positive_mass_sum=float(pos_mass),
        negative_mass_sum=float(neg_mass),
        stop_reason="normalised",
    )
    return normalised, summary


def _node_identity_sql(settings) -> str:
    """Full-identity expression for rows of the concat table."""
    column_info = settings.column_info_settings
    unique_id_col = column_info.unique_id_column_name
    source_dataset_col = (
        column_info.source_dataset_column_name
        if column_info.source_dataset_column_name is not None
        else None
    )
    if source_dataset_col is not None:
        return (
            f"cast({source_dataset_col} as varchar) || '-__-' || "
            f"cast({unique_id_col} as varchar)"
        )
    return f"cast({unique_id_col} as varchar)"


def label_join_key_sql(settings, side: str) -> str:
    """Composite key expression for a blocked/comparison-vector row.

    ``side`` is "l" or "r".  The comparison-vector table exposes columns such
    as ``"unique_id_l"`` and (for link jobs) ``"source_dataset_l"``; the
    suffix is part of the quoted identifier, hence the ``name_l``/``name_r``
    InputColumn properties (rather than manually appending ``_l``).
    """
    column_info = settings.column_info_settings
    uid = column_info.unique_id_input_column
    cols = []
    sds = column_info.source_dataset_input_column
    if sds is not None:
        side_col = sds.name_l if side == "l" else sds.name_r
        cols.append(f"cast({side_col} as varchar)")
    side_uid = uid.name_l if side == "l" else uid.name_r
    cols.append(f"cast({side_uid} as varchar)")
    return " || '-__-' || ".join(cols)


def _node_identity_alias_sql(settings, side: str) -> str:
    """Full-identity expression for the ``l``/``r`` aliases in a blocked
    join (i.e. references into the aliased concat table, like
    ``l."unique_id"``)."""
    column_info = settings.column_info_settings
    uid = column_info.unique_id_input_column
    cols = []
    sds = column_info.source_dataset_input_column
    if sds is not None:
        side_col = f"{side}.{sds.name}"
        cols.append(f"cast({side_col} as varchar)")
    cols.append(f"cast({side}.{uid.name} as varchar)")
    return " || '-__-' || ".join(cols)


def block_from_normalised_labels_sqls(
    linker: "Linker", normalised_labels_table: str
) -> list[dict[str, str]]:
    """Build the blocked pairwise comparisons for a normalised label table.

    Unlike :func:`block_from_labels` this consumes the internal, canonical
    table (composite key columns) and carries both the clerical match score
    and the label weight through to the comparison-vector stage.

    The output ``__splink__df_blocked`` table has the same shape as the
    ordinary blocking output, plus ``clerical_match_score`` and the internal
    weight column.
    """
    settings = linker._settings_obj
    columns_to_select = settings._columns_to_select_for_blocking
    sql_select_expr = ", ".join(columns_to_select)
    concat_name = "__splink__df_concat_with_tf"

    l_key = _node_identity_alias_sql(settings, "l")
    r_key = _node_identity_alias_sql(settings, "r")

    sql = f"""
    select
        {sql_select_expr},
        'from_labels' as match_key,
        lab.{LABEL_SCORE_COLUMN} as {LABEL_SCORE_COLUMN},
        lab.{LABEL_WEIGHT_COLUMN} as {LABEL_WEIGHT_COLUMN}
    from {normalised_labels_table} as lab
    inner join {concat_name} as l
        on {l_key} = lab.{LABEL_KEY_L}
    inner join {concat_name} as r
        on {r_key} = lab.{LABEL_KEY_R}
    """
    return [
        {
            "sql": sql,
            "output_table_name": "__splink__df_blocked",
        }
    ]


def split_labelled_and_unlabelled_cvv_sqls(
    linker: "Linker",
    cvv_table_name: str,
    normalised_labels_table: str,
) -> list[dict[str, str]]:
    """Split a comparison-vector table into labelled and unlabelled rows.

    The label table is joined on the full canonical pair identity.  Two
    tables are produced:

    * ``__splink__df_comparison_vectors_labelled`` - the labelled rows with
      ``clerical_match_score`` and the internal weight column attached;
    * ``__splink__df_comparison_vectors_unlabelled`` - every other row, with
      the original comparison-vector columns only.

    The contract that labels may not enlarge the training block is enforced
    separately via :func:`check_all_labels_are_in_training_block`.
    """
    settings = linker._settings_obj
    key_l = label_join_key_sql(settings, "l")
    key_r = label_join_key_sql(settings, "r")

    labelled_sql = f"""
    select cvv.*,
           lab.{LABEL_SCORE_COLUMN} as {LABEL_SCORE_COLUMN},
           lab.{LABEL_WEIGHT_COLUMN} as {LABEL_WEIGHT_COLUMN}
    from {cvv_table_name} as cvv
    inner join {normalised_labels_table} as lab
    on least({key_l}, {key_r}) = lab.{LABEL_KEY_L}
    and greatest({key_l}, {key_r}) = lab.{LABEL_KEY_R}
    """

    unlabelled_sql = f"""
    select cvv.*
    from {cvv_table_name} as cvv
    left join {normalised_labels_table} as lab
    on least({key_l}, {key_r}) = lab.{LABEL_KEY_L}
    and greatest({key_l}, {key_r}) = lab.{LABEL_KEY_R}
    where lab.{LABEL_KEY_L} is null
    """

    return [
        {
            "sql": labelled_sql,
            "output_table_name": "__splink__df_comparison_vectors_labelled",
        },
        {
            "sql": unlabelled_sql,
            "output_table_name": "__splink__df_comparison_vectors_unlabelled",
        },
    ]


def check_all_labels_are_in_training_block(
    linker: "Linker",
    cvv_table_name: str,
    normalised_labels_table: str,
) -> None:
    """Raise ValueError if any normalised label falls outside the training
    block generated by the blocking rule.

    Labels must never enlarge a training block - they only reweight evidence
    for pairs the blocking rule already produces.
    """
    settings = linker._settings_obj
    key_l = label_join_key_sql(settings, "l")
    key_r = label_join_key_sql(settings, "r")
    db_api = linker._db_api

    sql = f"""
    select count(*) as n_outside
    from {normalised_labels_table} as lab
    left join {cvv_table_name} as cvv
    on least({key_l}, {key_r}) = lab.{LABEL_KEY_L}
    and greatest({key_l}, {key_r}) = lab.{LABEL_KEY_R}
    where cvv.{settings.column_info_settings.unique_id_input_column.name_l}
        is null
    """
    n_outside = db_api._execute_sql_against_backend(sql).fetchall()[0][0]
    if n_outside > 0:
        raise ValueError(
            f"{n_outside} labelled pair(s) are not generated by the training "
            "blocking rule. Pairwise labels may only reweight candidate "
            "pairs inside the training block; they cannot enlarge it. Choose "
            "a blocking rule that covers all labelled pairs or remove the "
            "offending labels."
        )


def labelled_mass_counts_sql(comparisons) -> str:
    """Per-gamma weighted match / non-match mass from the labelled rows.

    The labelled contribution to the m count of level k is
    ``sum(weight * score)`` and to the u count is
    ``sum(weight * (1 - score))``.  These quantities are fixed across EM
    iterations because the clerical score is held constant, so the result
    can be materialised once and reused every iteration.
    """
    union_sqls = []
    for cc in comparisons:
        gamma_col = cc._gamma_column_name
        union_sqls.append(
            f"""
            select
                {gamma_col} as comparison_vector_value,
                sum({LABEL_WEIGHT_COLUMN} * {LABEL_SCORE_COLUMN}) as m_count,
                sum(
                    {LABEL_WEIGHT_COLUMN}
                    * (1.0 - {LABEL_SCORE_COLUMN})
                ) as u_count,
                '{cc.output_column_name}' as output_column_name
            from __splink__df_comparison_vectors_labelled
            group by {gamma_col}
            """
        )

    sql = f"""
    select
        0 as comparison_vector_value,
        sum({LABEL_WEIGHT_COLUMN} * {LABEL_SCORE_COLUMN}) as m_count,
        sum({LABEL_WEIGHT_COLUMN} * (1.0 - {LABEL_SCORE_COLUMN})) as u_count,
        '_probability_two_random_records_match' as output_column_name
    from __splink__df_comparison_vectors_labelled
    """
    union_sqls.append(sql)
    return " union all ".join(union_sqls)


def labelled_total_weight_sql() -> str:
    """Total labelled weight, used for the training-block prior estimate."""
    return f"""
    select sum({LABEL_WEIGHT_COLUMN}) as total_weight,
           sum({LABEL_WEIGHT_COLUMN} * {LABEL_SCORE_COLUMN}) as positive_mass,
           sum(
               {LABEL_WEIGHT_COLUMN}
               * (1.0 - {LABEL_SCORE_COLUMN})
           ) as negative_mass
    from __splink__df_comparison_vectors_labelled
    """


def labelled_summary_counts_sql() -> str:
    """Aggregate diagnostics for the labelled evidence."""
    return f"""
    select
        count(*) as labelled_pair_count,
        coalesce(sum({LABEL_WEIGHT_COLUMN}), 0.0) as label_weight_sum,
        coalesce(
            sum({LABEL_WEIGHT_COLUMN} * {LABEL_SCORE_COLUMN}), 0.0
        ) as positive_mass_sum,
        coalesce(
            sum({LABEL_WEIGHT_COLUMN} * (1.0 - {LABEL_SCORE_COLUMN})), 0.0
        ) as negative_mass_sum
    from __splink__df_comparison_vectors_labelled
    """

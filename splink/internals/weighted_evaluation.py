"""Weighted quality evaluation from a stratified clerical review.

When the reviewed pairs come from
:meth:`LinkerEvalution.sample_pairwise_predictions`, each pair was drawn with
a known, stratum-specific inclusion probability ``pi``.  Every reviewed pair
then represents ``1 / pi`` pairs of the original prediction population (its
Horvitz-Thompson mass, called ``evaluation_weight``).

This module contains:

* validation of the labels table on the weighted path (probability column
  contract, on top of the existing identity / self-pair / duplicate checks);
* the weighted truth-space SQL generator, which mirrors the unweighted
  implementation in :mod:`splink.internals.accuracy` but accumulates
  evaluation mass instead of row counts;
* helpers threading the probability column through the existing
  blocking -> comparison vectors -> prediction chain.

The unweighted path is untouched: callers that do not name an
``inclusion_probability_column`` get exactly the legacy SQL and outputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from splink.internals.pipeline import CTEPipeline
from splink.internals.vertically_concatenate import compute_df_concat_with_tf

if TYPE_CHECKING:
    from splink.internals.linker import Linker

# Public evaluation weight column added to the weighted error table.
EVALUATION_WEIGHT_COLUMN = "evaluation_weight"
# Weighted truth-space diagnostics.
SAMPLED_PAIR_COUNT_COLUMN = "sampled_pair_count"
ESTIMATED_PAIR_COUNT_COLUMN = "estimated_pair_count"


def validate_inclusion_probability_labels(
    linker: "Linker",
    labels_tablename: str,
    inclusion_probability_column: str,
) -> None:
    """Validate a pairwise labels table on the weighted evaluation path.

    Checks the probability column exists and contains only finite values in
    the open-closed interval ``(0, 1]``, then reuses the canonical identity
    machinery to reject self-pairs, unknown nodes and duplicated unoriented
    label pairs.  Raises :class:`ValueError` on any contract violation.

    Nothing is dropped or rewritten: a failure aborts before evaluation SQL
    is materialised and leaves the input table, model parameters and any
    cached state untouched.
    """
    db_api = linker._db_api
    settings = linker._settings_obj
    column_info = settings.column_info_settings
    unique_id_col = column_info.unique_id_column_name
    source_dataset_col = column_info.source_dataset_column_name

    labels_sdf = linker._table_to_splink_dataframe(
        labels_tablename, labels_tablename
    )
    raw_columns = [c.unquote().name for c in labels_sdf.columns]
    required = [
        f"{unique_id_col}_l",
        f"{unique_id_col}_r",
        "clerical_match_score",
        inclusion_probability_column,
    ]
    if source_dataset_col is not None:
        required = [
            f"{source_dataset_col}_l",
            f"{source_dataset_col}_r",
            *required,
        ]
    missing = [c for c in required if c not in raw_columns]
    if missing:
        raise ValueError(
            "Weighted evaluation labels table is missing required column(s): "
            + ", ".join(missing)
        )

    con = db_api._con
    table = labels_sdf.physical_name

    if source_dataset_col is not None:
        id_l = (
            f"(cast({source_dataset_col}_l as varchar) || '-__-' || "
            f"cast({unique_id_col}_l as varchar))"
        )
        id_r = (
            f"(cast({source_dataset_col}_r as varchar) || '-__-' || "
            f"cast({unique_id_col}_r as varchar))"
        )
    else:
        id_l = f"cast({unique_id_col}_l as varchar)"
        id_r = f"cast({unique_id_col}_r as varchar)"
    key_l = f"least({id_l}, {id_r})"
    key_r = f"greatest({id_l}, {id_r})"
    # TRY_CAST turns non-numeric values into null rather than raising, so the
    # generic null/non-finite check below can reject them with a clear
    # ValueError instead of letting a backend conversion error escape.
    prob_expr = f"try_cast({inclusion_probability_column} as float8)"
    score_expr = "try_cast(clerical_match_score as float8)"

    n_rows = con.sql(f"select count(*) from {table}").fetchall()[0][0]
    if n_rows == 0:
        raise ValueError(
            "Weighted evaluation requires at least one clerically reviewed "
            "pair, but the labels table is empty"
        )

    checks = con.sql(
        f"""
        select
            sum(case when {prob_expr} is null
                          or {prob_expr} <> {prob_expr}
                          or abs({prob_expr}) = 1.0/0.0
                     then 1 else 0 end) as n_bad_prob,
            sum(case when {prob_expr} <= 0.0 or {prob_expr} > 1.0
                     then 1 else 0 end) as n_out_of_range_prob,
            sum(case when {score_expr} is null
                          or {score_expr} <> {score_expr}
                          or abs({score_expr}) = 1.0/0.0
                          or {score_expr} < 0.0 or {score_expr} > 1.0
                     then 1 else 0 end) as n_bad_score,
            sum(case when {id_l} is null or {id_r} is null
                     then 1 else 0 end) as n_null_id,
            sum(case when {id_l} = {id_r} then 1 else 0 end) as n_self_pair
        from {table}
        """
    ).fetchall()[0]
    (
        n_bad_prob,
        n_out_of_range_prob,
        n_bad_score,
        n_null_id,
        n_self_pair,
    ) = checks

    if n_bad_prob:
        raise ValueError(
            f"{inclusion_probability_column} must be numeric and finite; "
            f"{n_bad_prob} null, non-numeric, NaN or infinite value(s) found"
        )
    if n_out_of_range_prob:
        raise ValueError(
            f"{inclusion_probability_column} values must lie in the interval "
            f"(0, 1]; {n_out_of_range_prob} offending value(s) found"
        )
    if n_bad_score:
        raise ValueError(
            "clerical_match_score must be a finite numeric value between 0 "
            f"and 1; {n_bad_score} offending value(s) found"
        )
    if n_null_id:
        raise ValueError(
            "Pair identity columns must not contain nulls; "
            f"{n_null_id} offending row(s)"
        )
    if n_self_pair:
        raise ValueError(
            "Self-pairs (the same record on both sides) are not allowed in "
            f"evaluation labels; {n_self_pair} offending row(s)"
        )

    # Duplicated unoriented pairs (including reversed submissions).
    n_duplicates = con.sql(
        f"""
        select count(*) from (
            select {key_l}, {key_r}
            from {table}
            group by 1, 2
            having count(*) > 1
        )
        """
    ).fetchall()[0][0]
    if n_duplicates:
        raise ValueError(
            f"Duplicate unoriented label pairs are not allowed; "
            f"{n_duplicates} pair(s) appear more than once (including "
            "reversed pairs)"
        )

    # Every referenced node must exist in the linker's input records.
    pipeline = CTEPipeline()
    nodes_with_tf = compute_df_concat_with_tf(linker, pipeline)
    from splink.internals.pairwise_sampling import _node_identity_from_settings

    node_id = _node_identity_from_settings(settings)
    n_unknown = con.sql(
        f"""
        with referenced as (
            select {key_l} as k from {table}
            union
            select {key_r} as k from {table}
        ),
        nodes as (
            select distinct {node_id} as k
            from {nodes_with_tf.physical_name}
        )
        select count(*) from referenced r
        left join nodes n on r.k = n.k
        where n.k is null
        """
    ).fetchall()[0][0]
    if n_unknown:
        raise ValueError(
            f"{n_unknown} label row(s) reference records that do not exist in "
            "the linker's input data"
        )


def weighted_truth_space_table_sqls(
    *,
    threshold_actual: float,
    match_weight_round_to_nearest: float | None,
    inclusion_probability_column: str,
) -> list[dict[str, str]]:
    """Weighted version of the truth-space threshold sweep.

    Mirrors :func:`truth_space_table_from_labels_with_predictions_sqls` but
    every row contributes its ``evaluation_weight = 1 / pi`` mass rather than
    a count of one.  Threshold rounding, the blocking-rule zero override and
    all zero-denominator conventions stay identical to the unweighted path.
    """
    if match_weight_round_to_nearest is not None:
        truth_thres_expr = f"""
            cast({match_weight_round_to_nearest} as float) *
            (round(match_weight/{match_weight_round_to_nearest}))
        """
    else:
        truth_thres_expr = "match_weight"

    weight_expr = f"(1.0 / cast({inclusion_probability_column} as float8))"

    sqls = []

    sql = f"""
    select
    *,
    {weight_expr} as {EVALUATION_WEIGHT_COLUMN},
    {truth_thres_expr} as truth_threshold,
    case when clerical_match_score >= {threshold_actual} then 1
    else 0
    end
    as clerical_positive,
    case when clerical_match_score >= {threshold_actual} then 0
    else 1
    end
    as clerical_negative
    from __splink__labels_with_predictions
    order by match_weight
    """
    sqls.append(
        {"sql": sql, "output_table_name": "__splink__labels_with_pos_neg"}
    )

    truth_threshold = """CASE
                        WHEN found_by_blocking_rules then truth_threshold
                        ELSE cast(-999 as float8)
                        END"""
    sql = f"""
    select *, {truth_threshold} as truth_threshold_adj
    from __splink__labels_with_pos_neg
    """
    sqls.append(
        {
            "sql": sql,
            "output_table_name": "__splink__labels_with_pos_neg_tt_adj",
        }
    )

    # Per-threshold weighted masses.  An explicit row-order-independent tie
    # breaker on the canonical pair keys keeps the window semantics identical
    # for every row sharing a threshold, mirroring the determinism the
    # unweighted aggregation relies on via its upstream ordering.
    sql = f"""
    select
        truth_threshold_adj as truth_threshold,
        sum({EVALUATION_WEIGHT_COLUMN}) as num_records_in_row,
        sum({EVALUATION_WEIGHT_COLUMN} * clerical_positive) as clerical_positive,
        sum({EVALUATION_WEIGHT_COLUMN} * clerical_negative) as clerical_negative
    from
    __splink__labels_with_pos_neg_tt_adj
    group by truth_threshold_adj
    order by truth_threshold_adj
    """
    sqls.append(
        {
            "sql": sql,
            "output_table_name": "__splink__labels_with_pos_neg_grouped",
        }
    )

    sql = f"""
    select
    truth_threshold,

    (sum(clerical_positive) over (order by truth_threshold desc))
        as cumulative_clerical_positives_at_or_above_threshold,

    (sum(clerical_negative) over (order by truth_threshold))
        - clerical_negative
        as cumulative_clerical_negatives_below_threshold,

    (select sum(clerical_positive) from __splink__labels_with_pos_neg_grouped)
        as total_clerical_positives,

    (select sum(clerical_negative) from __splink__labels_with_pos_neg_grouped)
        as total_clerical_negatives,

    (select sum(num_records_in_row) from __splink__labels_with_pos_neg_grouped)
        as total_clerical_labels,

    -num_records_in_row + sum(num_records_in_row) over (order by truth_threshold)
        as num_labels_scored_below_threshold,

    sum(num_records_in_row)
        over (order by truth_threshold desc) as num_labels_scored_at_or_above_threshold

    from __splink__labels_with_pos_neg_grouped
    order by  truth_threshold
    """
    sqls.append(
        {
            "sql": sql,
            "output_table_name": "__splink__labels_with_pos_neg_grouped_with_stats",
        }
    )

    # No ghost / implicit labels on the pairwise-table path.
    sql = """
    select * from __splink__labels_with_pos_neg_grouped_with_stats
    """
    sqls.append(
        {
            "sql": sql,
            "output_table_name": "__splink__labels_with_pos_neg_grouped_with_stats_adj",
        }
    )

    sql = """
    select
    truth_threshold,
    total_clerical_labels,
    total_clerical_positives as P,
    total_clerical_negatives as N,

    num_labels_scored_at_or_above_threshold -
        cumulative_clerical_positives_at_or_above_threshold as FP,

    cumulative_clerical_positives_at_or_above_threshold as TP,

    num_labels_scored_below_threshold - cumulative_clerical_negatives_below_threshold
        as FN,

    cumulative_clerical_negatives_below_threshold as TN

    from __splink__labels_with_pos_neg_grouped_with_stats_adj
    """
    sqls.append(
        {
            "sql": sql,
            "output_table_name": "__splink__labels_with_pos_neg_grouped_with_truth_stats",
        }
    )

    # Actual number of reviewed pairs (constant across thresholds) and the
    # estimated population mass (fractional, and never called a row count).
    sql = f"""
    select
        truth_threshold,
        power(2, truth_threshold) / (1 + power(2, truth_threshold))
            as match_probability,
        cast(total_clerical_labels as float8) as total_clerical_labels,
        cast(P as float8) as p,
        cast(N as float8) as n,
        cast(TP as float8) as tp,
        cast(TN as float8) as tn,
        cast(FP as float8) as fp,
        cast(FN as float8) as fn,
        cast(P/total_clerical_labels as float8) as P_rate,
        cast(N as float)/total_clerical_labels as N_rate,
        cast(TP as float8)/P as tp_rate,
        cast(TN as float8)/N as tn_rate,
        cast(FP as float8)/N as fp_rate,
        cast(FN as float8)/P as fn_rate,
        case when TP+FP=0 then 1 else cast(TP as float8)/(TP+FP) end as precision,
        cast(TP as float8)/P as recall,
        cast(TN as float8)/N as specificity,
        case when TN+FN=0 then 1 else cast(TN as float8)/(TN+FN) end as npv,
        cast(TP+TN as float8)/(P+N) as accuracy,
        cast(2.0*TP/(2*TP + FN + FP) as float8) as f1,
        cast(5.0*TP/(5*TP + 4*FN + FP) as float8) as f2,
        cast(1.25*TP/(1.25*TP + 0.25*FN + FP) as float8) as f0_5,
        cast(4.0*TP*TN/((4.0*TP*TN) + ((TP + TN)*(FP + FN))) as float8) as p4,
        case when TN+FN=0 or TP+FP=0 or P=0 or N=0 then 0
            else cast((TP*TN)-(FP*FN) as float8)/sqrt((TP+FP)*P*N*(TN+FN)) end as phi,
        (select count(*) from __splink__labels_with_pos_neg)
            as {SAMPLED_PAIR_COUNT_COLUMN},
        (select sum({EVALUATION_WEIGHT_COLUMN})
            from __splink__labels_with_pos_neg)
            as {ESTIMATED_PAIR_COUNT_COLUMN}

    from __splink__labels_with_pos_neg_grouped_with_truth_stats
    where truth_threshold >= cast(-998 as float8)
    """
    sqls.append(
        {"sql": sql, "output_table_name": "__splink__truth_space_table"}
    )
    return sqls

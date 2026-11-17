from __future__ import annotations

import logging
import time
from typing import Any, List, cast

import pandas as pd

from splink.internals.comparison import Comparison
from splink.internals.comparison_level import ComparisonLevel
from splink.internals.constants import LEVEL_NOT_OBSERVED_TEXT
from splink.internals.input_column import InputColumn
from splink.internals.labelled_pairwise import (
    LabelTrainingSummary,
    labelled_mass_counts_sql,
)
from splink.internals.m_u_records_to_parameters import m_u_records_to_lookup_dict
from splink.internals.pipeline import CTEPipeline
from splink.internals.predict import (
    predict_from_agreement_pattern_counts_sqls,
    predict_from_comparison_vectors_sqls,
)
from splink.internals.settings import CoreModelSettings, TrainingSettings
from splink.internals.splink_dataframe import SplinkDataFrame

from .database_api import DatabaseAPISubClass

logger = logging.getLogger(__name__)


def count_agreement_patterns_sql(comparisons: List[Comparison]) -> str:
    """Count how many times each realized agreement pattern
    was observed across the blocked dataset."""
    gamma_cols = [cc._gamma_column_name for cc in comparisons]
    gamma_cols_expr = ",".join(gamma_cols)

    sql = f"""
    select
    {gamma_cols_expr},
    count(*) as agreement_pattern_count
    from __splink__df_comparison_vectors
    group by {gamma_cols_expr}
    """

    return sql


def compute_new_parameters_sql(
    estimate_without_term_frequencies: bool, comparisons: List[Comparison]
) -> str:
    """compute m and u counts from the results of predict"""
    if estimate_without_term_frequencies:
        agreement_pattern_count = "agreement_pattern_count"
    else:
        agreement_pattern_count = "1"

    sql_template = """
    select
    {gamma_column} as comparison_vector_value,
    sum(match_probability * {agreement_pattern_count}) as m_count,
    sum((1-match_probability) * {agreement_pattern_count}) as u_count,
    '{output_column_name}' as output_column_name
    from __splink__df_predict
    group by {gamma_column}
    """
    union_sqls = [
        sql_template.format(
            gamma_column=cc._gamma_column_name,
            output_column_name=cc.output_column_name,
            agreement_pattern_count=agreement_pattern_count,
        )
        for cc in comparisons
    ]

    # Probability of two random records matching
    sql = f"""
    select 0 as comparison_vector_value,
           sum(match_probability * {agreement_pattern_count}) /
               sum({agreement_pattern_count}) as m_count,
           sum((1-match_probability) * {agreement_pattern_count}) /
               sum({agreement_pattern_count}) as u_count,
           '_probability_two_random_records_match' as output_column_name
    from __splink__df_predict
    """
    union_sqls.append(sql)

    sql = " union all ".join(union_sqls)

    return sql


def compute_proportions_for_new_parameters_sql(table_name):
    """Using the results from compute_new_parameters_sql, compute
    m and u
    """

    sql = f"""
    select
        comparison_vector_value,
        output_column_name,
        m_count/sum(m_count) over (PARTITION BY output_column_name)
            as m_probability,
        u_count/sum(u_count) over (PARTITION BY output_column_name)
            as u_probability
    from {table_name}
    where comparison_vector_value != -1
    and output_column_name != '_probability_two_random_records_match'

    union all

    select
        comparison_vector_value,
        output_column_name,
        m_count as m_probability,
        u_count as u_probability
    from {table_name}
    where output_column_name = '_probability_two_random_records_match'
    order by output_column_name, comparison_vector_value asc
    """

    return sql


def compute_proportions_for_new_parameters_pandas(
    m_u_df: pd.DataFrame,
) -> List[dict[str, Any]]:
    data = m_u_df.copy()
    m_prob = "m_probability"
    u_prob = "u_probability"
    data.rename(columns={"m_count": m_prob, "u_count": u_prob}, inplace=True)

    random_records = data[
        data.output_column_name == "_probability_two_random_records_match"
    ]
    data = data[data.output_column_name != "_probability_two_random_records_match"]

    data = data[data.comparison_vector_value != -1]
    index = data.index.tolist()

    m_probs = data.loc[index, m_prob] / data.groupby("output_column_name")[
        m_prob
    ].transform("sum")
    u_probs = data.loc[index, u_prob] / data.groupby("output_column_name")[
        u_prob
    ].transform("sum")

    data.loc[index, m_prob] = m_probs
    data.loc[index, u_prob] = u_probs

    data = pd.concat([random_records, data])

    return data.to_dict("records")


def compute_proportions_for_new_parameters(
    m_u_df: pd.DataFrame,
) -> List[dict[str, Any]]:
    # Execute with duckdb if installed, otherwise default to pandas
    try:
        import duckdb

        sql = compute_proportions_for_new_parameters_sql("m_u_df")
        return duckdb.query(sql).to_df().to_dict("records")
    except (ImportError, ModuleNotFoundError):
        return compute_proportions_for_new_parameters_pandas(m_u_df)


def populate_m_u_from_lookup(
    training_fixed_probabilities: set[str],
    comparison_level: ComparisonLevel,
    output_column_name: str,
    m_u_records_lookup: dict[str, dict[int, Any]],
) -> None:
    cl = comparison_level

    if not cl._fix_m_probability and "m" not in training_fixed_probabilities:
        try:
            m_probability = m_u_records_lookup[output_column_name][
                cl.comparison_vector_value
            ]["m_probability"]

        except KeyError:
            m_probability = LEVEL_NOT_OBSERVED_TEXT
            cc_n = output_column_name
            cl_n = cl.label_for_charts
            if not cl._m_warning_sent:
                logger.warning(
                    "WARNING:\n"
                    f"Level {cl_n} on comparison {cc_n} not observed in dataset, "
                    "unable to train m value\n"
                )
                cl._m_warning_sent = True
        cl.m_probability = m_probability

    if not cl._fix_u_probability and "u" not in training_fixed_probabilities:
        try:
            u_probability = m_u_records_lookup[output_column_name][
                cl.comparison_vector_value
            ]["u_probability"]

        except KeyError:
            u_probability = LEVEL_NOT_OBSERVED_TEXT

            cc_n = output_column_name
            cl_n = cl.label_for_charts
            if not cl._u_warning_sent:
                logger.warning(
                    "WARNING:\n"
                    f"Level {cl_n} on comparison {cc_n} not observed in dataset, "
                    "unable to train u value\n"
                )
                cl._u_warning_sent = True

        cl.u_probability = u_probability


def maximisation_step(
    training_fixed_probabilities: set[str],
    core_model_settings: CoreModelSettings,
    param_records: List[dict[str, Any]],
) -> CoreModelSettings:
    core_model_settings = core_model_settings.copy()

    m_u_records = []
    for r in param_records:
        if r["output_column_name"] == "_probability_two_random_records_match":
            prop_record = r
        else:
            m_u_records.append(r)

    if "lambda" not in training_fixed_probabilities:
        core_model_settings.probability_two_random_records_match = prop_record[
            "m_probability"
        ]

    m_u_records_lookup = m_u_records_to_lookup_dict(m_u_records)
    for cc in core_model_settings.comparisons:
        for cl in cc._comparison_levels_excluding_null:
            populate_m_u_from_lookup(
                training_fixed_probabilities,
                cl,
                cc.output_column_name,
                m_u_records_lookup,
            )

    return core_model_settings


def expectation_maximisation(
    db_api: DatabaseAPISubClass,
    training_settings: TrainingSettings,
    estimate_without_term_frequencies: bool,
    core_model_settings: CoreModelSettings,
    unique_id_input_columns: List[InputColumn],
    training_fixed_probabilities: set[str],
    df_comparison_vector_values: SplinkDataFrame,
) -> List[CoreModelSettings]:
    """In the expectation step, we use the current model parameters to estimate
    the probability of match for each pairwise record comparison

    In the maximisation step, we use these predicted probabilities to re-compute
    the parameters of the model
    """
    # initial values of parameters
    core_model_settings_history = [core_model_settings.copy()]

    sql_infinity_expression = db_api.sql_dialect.infinity_expression

    max_iterations = training_settings.max_iterations
    em_convergence = training_settings.em_convergence
    logger.info("")  # newline

    if estimate_without_term_frequencies:
        sql = count_agreement_patterns_sql(core_model_settings.comparisons)
        pipeline = CTEPipeline([df_comparison_vector_values])
        pipeline.enqueue_sql(sql, "__splink__agreement_pattern_counts")
        agreement_pattern_counts = db_api.sql_pipeline_to_splink_dataframe(pipeline)

    for i in range(1, max_iterations + 1):
        pipeline = CTEPipeline()
        probability_two_random_records_match = (
            core_model_settings.probability_two_random_records_match
        )
        start_time = time.time()

        # Expectation step
        if estimate_without_term_frequencies:
            sqls = predict_from_agreement_pattern_counts_sqls(
                core_model_settings.comparisons,
                probability_two_random_records_match,
                sql_infinity_expression=db_api.sql_dialect.infinity_expression,
            )
        else:
            sqls = predict_from_comparison_vectors_sqls(
                unique_id_input_columns=unique_id_input_columns,
                core_model_settings=core_model_settings,
                training_mode=True,
                sql_infinity_expression=sql_infinity_expression,
            )

        for sql_info in sqls:
            pipeline.enqueue_sql(sql_info["sql"], sql_info["output_table_name"])

        sql = compute_new_parameters_sql(
            estimate_without_term_frequencies,
            core_model_settings.comparisons,
        )
        pipeline.enqueue_sql(sql, "__splink__m_u_counts")
        if estimate_without_term_frequencies:
            pipeline.append_input_dataframe(agreement_pattern_counts)
            df_params = db_api.sql_pipeline_to_splink_dataframe(pipeline)
        else:
            pipeline.append_input_dataframe(df_comparison_vector_values)
            df_params = db_api.sql_pipeline_to_splink_dataframe(pipeline)
        param_records = df_params.as_pandas_dataframe()
        param_records = compute_proportions_for_new_parameters(param_records)

        df_params.drop_table_from_database_and_remove_from_cache()

        core_model_settings = maximisation_step(
            training_fixed_probabilities=training_fixed_probabilities,
            core_model_settings=core_model_settings,
            param_records=param_records,
        )
        core_model_settings_history.append(core_model_settings)

        max_change_dict = _max_change_in_parameters_comparison_levels(
            core_model_settings_history
        )
        logger.info(f"Iteration {i}: {max_change_dict['message']}")
        end_time = time.time()
        logger.log(15, f"    Iteration time: {end_time - start_time} seconds")

        if max_change_dict["max_abs_change_value"] < em_convergence:
            break

    logger.info(f"\nEM converged after {i} iterations")
    return core_model_settings_history


def _max_change_message(max_change_dict):
    message = "Largest change in params was"

    if max_change_dict["max_change_type"] == "probability_two_random_records_match":
        message = (
            f"{message} {max_change_dict['max_change_value']:,.3g} in "
            "probability_two_random_records_match"
        )
    else:
        cl = max_change_dict["current_comparison_level"]
        m_u = max_change_dict["max_change_type"]
        cc_name = max_change_dict["output_column_name"]

        cl_label = cl.label_for_charts
        level_text = f"{cc_name}, level `{cl_label}`"

        message = (
            f"{message} {max_change_dict['max_change_value']:,.3g} in "
            f"the {m_u} of {level_text}"
        )

    return message


def _max_change_in_parameters_comparison_levels(
    core_model_settings_history: List[CoreModelSettings],
) -> dict[str, Any]:
    previous_iteration = core_model_settings_history[-2]
    this_iteration = core_model_settings_history[-1]
    max_change = -0.1

    max_change_levels: dict[str, Any] = {
        "previous_iteration": None,
        "this_iteration": None,
        "max_change_type": None,
        "max_change_value": None,
    }
    comparisons = zip(previous_iteration.comparisons, this_iteration.comparisons)
    for comparison in comparisons:
        prev_cc = comparison[0]
        this_cc = comparison[1]
        z_cls = zip(prev_cc.comparison_levels, this_cc.comparison_levels)
        for z_cl in z_cls:
            if z_cl[0].is_null_level:
                continue
            prev_cl = z_cl[0]
            this_cl = z_cl[1]

            prev_m_prob = cast(float, prev_cl.m_probability)
            this_m_prob = cast(float, this_cl.m_probability)
            prev_u_prob = cast(float, prev_cl.u_probability)
            this_u_prob = cast(float, this_cl.u_probability)
            change_m = this_m_prob - prev_m_prob
            change_u = this_u_prob - prev_u_prob

            change = max(abs(change_m), abs(change_u))
            change_type = (
                "m_probability" if abs(change_m) > abs(change_u) else "u_probability"
            )
            change_value = change_m if abs(change_m) > abs(change_u) else change_u
            if change > max_change:
                max_change = change
                max_change_levels["prev_comparison_level"] = prev_cl
                max_change_levels["current_comparison_level"] = this_cl
                max_change_levels["max_change_type"] = change_type
                max_change_levels["max_change_value"] = change_value
                max_change_levels["max_abs_change_value"] = abs(change_value)
                max_change_levels["output_column_name"] = this_cc.output_column_name

    change_probability_two_random_records_match = (
        this_iteration.probability_two_random_records_match
        - previous_iteration.probability_two_random_records_match
    )

    if abs(change_probability_two_random_records_match) > max_change:
        max_change = abs(change_probability_two_random_records_match)
        max_change_levels["prev_comparison_level"] = None
        max_change_levels["current_comparison_level"] = None
        max_change_levels["max_change_type"] = "probability_two_random_records_match"
        max_change_levels["max_change_value"] = (
            change_probability_two_random_records_match
        )
        max_change_levels["max_abs_change_value"] = abs(
            change_probability_two_random_records_match
        )

    max_change_levels["message"] = _max_change_message(max_change_levels)

    return max_change_levels


# ---------------------------------------------------------------------------
# Weighted, partially-labelled expectation maximisation
#
# The evidence contract shared with the direct m estimator:
#   * labelled rows contribute fixed per-pair mass: match mass
#     w * s and non-match mass w * (1 - s), with s the clerical match score;
#   * unlabelled rows keep their model-inferred match_probability with
#     implicit weight 1;
#   * every comparison's own null level (gamma = -1) is excluded from that
#     comparison's m/u normalisation;
#   * where a comparison's effective m (or u) mass is zero, its previous
#     parameter is retained and the level/comparison is marked not estimated;
#   * labels never touch predict() - they only influence these parameter
#     updates.
# ---------------------------------------------------------------------------


def _unlabelled_raw_counts_sql(
    comparisons: List[Comparison], estimate_without_term_frequencies: bool
) -> str:
    """Raw (unnormalised) m/u numerators from the *unlabelled* rows only.

    Returned columns mirror :func:`compute_new_parameters_sql` so the two
    pieces of evidence can be summed in Python on parameter-level data.
    """
    if estimate_without_term_frequencies:
        weight_expr = "agreement_pattern_count"
        from_clause = "__splink__df_predict"
    else:
        weight_expr = "1"
        from_clause = "__splink__df_predict"

    sql_template = """
    select
    {gamma_column} as comparison_vector_value,
    sum(match_probability * {weight_expr}) as m_count,
    sum((1-match_probability) * {weight_expr}) as u_count,
    '{output_column_name}' as output_column_name
    from {from_clause}
    group by {gamma_column}
    """
    union_sqls = [
        sql_template.format(
            gamma_column=cc._gamma_column_name,
            output_column_name=cc.output_column_name,
            weight_expr=weight_expr,
            from_clause=from_clause,
        )
        for cc in comparisons
    ]
    sql = f"""
    select 0 as comparison_vector_value,
           sum(match_probability * {weight_expr}) as m_count,
           sum((1-match_probability) * {weight_expr}) as u_count,
           '_probability_two_random_records_match' as output_column_name
    from {from_clause}
    """
    union_sqls.append(sql)
    return " union all ".join(union_sqls)


def combine_labelled_and_unlabelled_parameters(
    unlabelled_df: pd.DataFrame,
    labelled_df: pd.DataFrame,
    core_model_settings: CoreModelSettings,
    training_fixed_probabilities: set[str],
    labelled_total_weight: float,
) -> tuple[List[dict[str, Any]], List[dict[str, str]]]:
    """Combine the two pieces of evidence on parameter-level data.

    All normalisation happens here in pandas; the tables pulled from the
    backend have at most a handful of rows (one per comparison level plus the
    prior row), never the full pairwise dataset.

    Returns the parameter records (same shape as
    :func:`compute_proportions_for_new_parameters`) and a list of
    ``not_estimated`` diagnostic records describing comparisons whose m or u
    effective mass was zero (previous parameters must be retained for those).
    """
    unlabelled = unlabelled_df.copy()
    labelled = labelled_df.copy()
    for df in (unlabelled, labelled):
        df["m_count"] = df["m_count"].astype(float)
        df["u_count"] = df["u_count"].astype(float)

    # Labelled prior row additionally carries total_weight.
    labelled_lambda = labelled[
        labelled["output_column_name"]
        == "_probability_two_random_records_match"
    ]
    unlabelled_lambda = unlabelled[
        unlabelled["output_column_name"]
        == "_probability_two_random_records_match"
    ]
    u_lam = unlabelled_lambda.iloc[0] if len(unlabelled_lambda) else None
    l_lam = labelled_lambda.iloc[0] if len(labelled_lambda) else None

    if u_lam is not None:
        unlabelled_n = float(u_lam["m_count"] + u_lam["u_count"])
    else:
        unlabelled_n = 0.0
    labelled_weight = float(labelled_total_weight)
    total_weight = unlabelled_n + labelled_weight

    if total_weight <= 0.0:
        # No evidence whatsoever - retain previous prior.
        lambda_value = core_model_settings.probability_two_random_records_match
    else:
        labelled_mass = float(l_lam["m_count"]) if l_lam is not None else 0.0
        unlabelled_mass = float(u_lam["m_count"]) if u_lam is not None else 0.0
        lambda_value = (labelled_mass + unlabelled_mass) / total_weight

    records = [
        {
            "comparison_vector_value": 0,
            "output_column_name": "_probability_two_random_records_match",
            "m_probability": lambda_value,
            "u_probability": lambda_value,
        }
    ]
    not_estimated: List[dict[str, str]] = []

    unlabelled_mu = unlabelled[
        unlabelled["output_column_name"]
        != "_probability_two_random_records_match"
    ]
    labelled_mu = labelled[
        labelled["output_column_name"]
        != "_probability_two_random_records_match"
    ]

    for cc in core_model_settings.comparisons:
        name = cc.output_column_name
        u_rows = unlabelled_mu[unlabelled_mu["output_column_name"] == name]
        l_rows = labelled_mu[labelled_mu["output_column_name"] == name]
        # Null levels (gamma = -1) never enter the normalisation.
        u_rows = u_rows[u_rows["comparison_vector_value"] != -1]
        l_rows = l_rows[l_rows["comparison_vector_value"] != -1]

        merged = pd.merge(
            u_rows[["comparison_vector_value", "m_count", "u_count"]],
            l_rows[["comparison_vector_value", "m_count", "u_count"]],
            on="comparison_vector_value",
            how="outer",
            suffixes=("_u", "_l"),
        ).fillna(0.0)
        merged["m_total"] = merged["m_count_u"] + merged["m_count_l"]
        merged["u_total"] = merged["u_count_u"] + merged["u_count_l"]

        m_denom = float(merged["m_total"].sum())
        u_denom = float(merged["u_total"].sum())

        if m_denom <= 0.0 and "m" not in training_fixed_probabilities:
            not_estimated.append(
                {"output_column_name": name, "parameter": "m", "reason": "zero_mass"}
            )
        if u_denom <= 0.0 and "u" not in training_fixed_probabilities:
            not_estimated.append(
                {"output_column_name": name, "parameter": "u", "reason": "zero_mass"}
            )

        for _, row in merged.iterrows():
            m_prob = (
                float(row["m_total"]) / m_denom if m_denom > 0.0 else float("nan")
            )
            u_prob = (
                float(row["u_total"]) / u_denom if u_denom > 0.0 else float("nan")
            )
            records.append(
                {
                    "comparison_vector_value": int(row["comparison_vector_value"]),
                    "output_column_name": name,
                    "m_probability": m_prob,
                    "u_probability": u_prob,
                }
            )

    return records, not_estimated


def labelled_maximisation_step(
    training_fixed_probabilities: set[str],
    core_model_settings: CoreModelSettings,
    param_records: List[dict[str, Any]],
    not_estimated: List[dict[str, str]],
) -> CoreModelSettings:
    """Maximisation step that retains previous parameters where the
    effective mass for m (or u) was zero, instead of writing NaN."""
    core_model_settings = core_model_settings.copy()

    prop_record = next(
        r
        for r in param_records
        if r["output_column_name"] == "_probability_two_random_records_match"
    )
    if "lambda" not in training_fixed_probabilities:
        core_model_settings.probability_two_random_records_match = prop_record[
            "m_probability"
        ]

    not_estimated_set = {
        (d["output_column_name"], d["parameter"]) for d in not_estimated
    }
    m_u_records = [
        r
        for r in param_records
        if r["output_column_name"] != "_probability_two_random_records_match"
    ]
    m_u_records_lookup = m_u_records_to_lookup_dict(m_u_records)

    for cc in core_model_settings.comparisons:
        name = cc.output_column_name
        skip_m = (
            (name, "m") in not_estimated_set
            or "m" in training_fixed_probabilities
        )
        skip_u = (
            (name, "u") in not_estimated_set
            or "u" in training_fixed_probabilities
        )
        for cl in cc._comparison_levels_excluding_null:
            if not skip_m:
                try:
                    cl.m_probability = m_u_records_lookup[name][
                        cl.comparison_vector_value
                    ]["m_probability"]
                except KeyError:
                    # Level not observed at all in this iteration's evidence:
                    # retain the previous value rather than inventing one.
                    logger.info(
                        f"m probability not trained for {name} - "
                        f"{cl.label_for_charts} (comparison vector value: "
                        f"{cl.comparison_vector_value}); previous value retained"
                    )
            if not skip_u:
                try:
                    cl.u_probability = m_u_records_lookup[name][
                        cl.comparison_vector_value
                    ]["u_probability"]
                except KeyError:
                    logger.info(
                        f"u probability not trained for {name} - "
                        f"{cl.label_for_charts} (comparison vector value: "
                        f"{cl.comparison_vector_value}); previous value retained"
                    )

    return core_model_settings


def expectation_maximisation_labelled(
    db_api: DatabaseAPISubClass,
    training_settings: TrainingSettings,
    estimate_without_term_frequencies: bool,
    core_model_settings: CoreModelSettings,
    unique_id_input_columns: List[InputColumn],
    training_fixed_probabilities: set[str],
    df_comparison_vector_values_labelled: SplinkDataFrame,
    df_comparison_vector_values_unlabelled: SplinkDataFrame,
    label_summary: LabelTrainingSummary,
) -> tuple[List[CoreModelSettings], LabelTrainingSummary, List[dict[str, str]]]:
    """EM over a partially labelled blocked dataset.

    Labelled rows hold the clerical score fixed in every iteration; unlabelled
    rows are scored by the current model.  The labelled mass counts are
    computed once (they do not depend on the model parameters) and only the
    small, parameter-level count tables ever cross into Python.
    """
    core_model_settings_history = [core_model_settings.copy()]
    sql_infinity_expression = db_api.sql_dialect.infinity_expression
    max_iterations = training_settings.max_iterations
    em_convergence = training_settings.em_convergence
    logger.info("")

    # Fixed labelled contribution (materialised once, reused every iteration).
    pipeline = CTEPipeline([df_comparison_vector_values_labelled])
    pipeline.enqueue_sql(
        labelled_mass_counts_sql(core_model_settings.comparisons),
        "__splink__labelled_m_u_counts",
    )
    labelled_counts_sdf = db_api.sql_pipeline_to_splink_dataframe(pipeline)
    labelled_counts = labelled_counts_sdf.as_pandas_dataframe()

    # Total labelled weight (fixed across iterations), needed for the prior.
    pipeline = CTEPipeline([df_comparison_vector_values_labelled])
    from splink.internals.labelled_pairwise import labelled_total_weight_sql

    pipeline.enqueue_sql(
        labelled_total_weight_sql(), "__splink__labelled_total_weight"
    )
    total_weight_sdf = db_api.sql_pipeline_to_splink_dataframe(pipeline)
    total_weight_row = total_weight_sdf.as_record_dict()[0]
    total_weight_sdf.drop_table_from_database_and_remove_from_cache()
    labelled_total_weight = float(total_weight_row["total_weight"])

    # Labelled diagnostics (also fixed across iterations).
    pipeline = CTEPipeline([df_comparison_vector_values_labelled])
    from splink.internals.labelled_pairwise import labelled_summary_counts_sql

    pipeline.enqueue_sql(
        labelled_summary_counts_sql(), "__splink__labelled_summary"
    )
    labelled_summary_sdf = db_api.sql_pipeline_to_splink_dataframe(pipeline)
    labelled_summary_row = labelled_summary_sdf.as_record_dict()[0]
    labelled_summary_sdf.drop_table_from_database_and_remove_from_cache()

    # Count unlabelled rows cheaply without materialising the whole frame in
    # Python: the table already exists in the backend.
    unlabelled_count_sql = (
        f"select count(*) as n from "
        f"{df_comparison_vector_values_unlabelled.physical_name}"
    )
    unlabelled_pair_count = db_api._execute_sql_against_backend(
        unlabelled_count_sql
    ).fetchall()[0][0]

    label_summary.labelled_pair_count = int(
        labelled_summary_row["labelled_pair_count"]
    )
    label_summary.unlabelled_pair_count = int(unlabelled_pair_count)
    label_summary.label_weight_sum = float(labelled_summary_row["label_weight_sum"])
    label_summary.positive_mass_sum = float(
        labelled_summary_row["positive_mass_sum"]
    )
    label_summary.negative_mass_sum = float(
        labelled_summary_row["negative_mass_sum"]
    )

    if estimate_without_term_frequencies and unlabelled_pair_count > 0:
        # Compress unlabelled rows to agreement-pattern counts.  Labelled rows
        # are NOT folded in here: identical comparison vectors with different
        # clerical scores must remain distinct evidence.
        pipeline = CTEPipeline([df_comparison_vector_values_unlabelled])
        pipeline.enqueue_sql(
            count_agreement_patterns_sql(core_model_settings.comparisons),
            "__splink__agreement_pattern_counts_unlabelled",
        )
        agreement_pattern_counts = db_api.sql_pipeline_to_splink_dataframe(pipeline)
        # The compressed predict SQLs read a CTE named exactly
        # __splink__agreement_pattern_counts.
        agreement_pattern_counts.templated_name = (
            "__splink__agreement_pattern_counts"
        )

    # Empty-count frame reused when there are no unlabelled rows: in that case
    # the model scores nothing, so the unlabelled contribution to every count
    # is zero and the predict SQL is skipped entirely (this also avoids NaN
    # priors being rendered into SQL on an empty scoring table).
    empty_unlabelled_counts = pd.DataFrame(
        {
            "comparison_vector_value": pd.Series(dtype="int64"),
            "m_count": pd.Series(dtype="float64"),
            "u_count": pd.Series(dtype="float64"),
            "output_column_name": pd.Series(dtype="object"),
        }
    )

    all_not_estimated: List[dict[str, str]] = []
    stop_reason = "max_iterations_reached"
    i = 0
    for i in range(1, max_iterations + 1):
        start_time = time.time()

        if unlabelled_pair_count == 0:
            unlabelled_counts = empty_unlabelled_counts.copy()
        else:
            pipeline = CTEPipeline()
            probability_two_random_records_match = (
                core_model_settings.probability_two_random_records_match
            )

            if estimate_without_term_frequencies:
                sqls = predict_from_agreement_pattern_counts_sqls(
                    core_model_settings.comparisons,
                    probability_two_random_records_match,
                    sql_infinity_expression=db_api.sql_dialect.infinity_expression,
                )
                pipeline.append_input_dataframe(agreement_pattern_counts)
            else:
                sqls = predict_from_comparison_vectors_sqls(
                    unique_id_input_columns=unique_id_input_columns,
                    core_model_settings=core_model_settings,
                    training_mode=True,
                    sql_infinity_expression=sql_infinity_expression,
                )
                pipeline.append_input_dataframe(
                    df_comparison_vector_values_unlabelled
                )

            for sql_info in sqls:
                pipeline.enqueue_sql(sql_info["sql"], sql_info["output_table_name"])

            pipeline.enqueue_sql(
                _unlabelled_raw_counts_sql(
                    core_model_settings.comparisons,
                    estimate_without_term_frequencies,
                ),
                "__splink__unlabelled_m_u_counts",
            )
            unlabelled_counts_sdf = db_api.sql_pipeline_to_splink_dataframe(pipeline)
            unlabelled_counts = unlabelled_counts_sdf.as_pandas_dataframe()
            unlabelled_counts_sdf.drop_table_from_database_and_remove_from_cache()

        param_records, not_estimated = combine_labelled_and_unlabelled_parameters(
            unlabelled_counts,
            labelled_counts,
            core_model_settings,
            training_fixed_probabilities,
            labelled_total_weight,
        )
        for ne in not_estimated:
            if ne not in all_not_estimated:
                all_not_estimated.append(ne)

        core_model_settings = labelled_maximisation_step(
            training_fixed_probabilities=training_fixed_probabilities,
            core_model_settings=core_model_settings,
            param_records=param_records,
            not_estimated=not_estimated,
        )
        core_model_settings_history.append(core_model_settings)

        max_change_dict = _max_change_in_parameters_comparison_levels(
            core_model_settings_history
        )
        logger.info(f"Iteration {i}: {max_change_dict['message']}")
        end_time = time.time()
        logger.log(15, f"    Iteration time: {end_time - start_time} seconds")

        if max_change_dict["max_abs_change_value"] < em_convergence:
            stop_reason = "converged"
            break

    labelled_counts_sdf.drop_table_from_database_and_remove_from_cache()
    if (
        estimate_without_term_frequencies
        and unlabelled_pair_count > 0
    ):
        agreement_pattern_counts.drop_table_from_database_and_remove_from_cache()

    label_summary.stop_reason = stop_reason
    label_summary.not_estimated = all_not_estimated
    logger.info(
        f"\nLabelled EM finished after {i} iterations (stop reason: {stop_reason})"
    )
    return core_model_settings_history, label_summary, all_not_estimated

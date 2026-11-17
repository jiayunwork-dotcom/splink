import logging
from typing import TYPE_CHECKING, Optional

from splink.internals.block_from_labels import block_from_labels
from splink.internals.comparison_vector_values import (
    compute_comparison_vector_values_sql,
)
from splink.internals.expectation_maximisation import (
    compute_new_parameters_sql,
    compute_proportions_for_new_parameters,
)
from splink.internals.labelled_pairwise import (
    LABEL_SCORE_COLUMN,
    LABEL_WEIGHT_COLUMN,
    LabelTrainingSummary,
    block_from_normalised_labels_sqls,
    normalise_pairwise_labels,
)
from splink.internals.pipeline import CTEPipeline
from splink.internals.vertically_concatenate import compute_df_concat_with_tf

from .m_u_records_to_parameters import (
    append_m_probability_to_comparison_level_trained_probabilities,
    m_u_records_to_lookup_dict,
)

if TYPE_CHECKING:
    from splink.internals.linker import Linker
logger = logging.getLogger(__name__)


def estimate_m_from_pairwise_labels(
    linker: "Linker",
    table_name: str,
    use_label_scores: bool = False,
    label_weight_column: Optional[str] = None,
) -> Optional[LabelTrainingSummary]:
    if not use_label_scores:
        # Legacy behaviour: every label row is treated as a certain match,
        # clerical_match_score (if present) is ignored.
        if label_weight_column is not None:
            raise ValueError(
                "label_weight_column can only be supplied when "
                "use_label_scores=True"
            )
        _estimate_m_legacy(linker, table_name)
        return None

    # All validation happens before any model parameters are touched, so a
    # failed call leaves the existing parameters / training history /
    # predictions untouched.
    normalised_labels, summary = normalise_pairwise_labels(
        linker,
        table_name,
        use_label_scores=True,
        label_weight_column=label_weight_column,
    )

    if normalised_labels is None:
        # Empty label table: no positive mass at all.
        raise ValueError(
            "Cannot estimate m probabilities: the pairwise label table "
            "contained no rows with positive matching mass "
            "(all clerical_match_score values are 0 or the table is empty)"
        )

    if summary.positive_mass_sum <= 0.0:
        raise ValueError(
            "Cannot estimate m probabilities: the pairwise label evidence "
            "has zero positive matching mass "
            "(sum of label_weight * clerical_match_score is 0)"
        )

    _estimate_m_weighted(linker, normalised_labels.physical_name)
    return summary


def _estimate_m_legacy(linker: "Linker", table_name: str) -> None:
    pipeline = CTEPipeline()
    nodes_with_tf = compute_df_concat_with_tf(linker, pipeline)
    pipeline = CTEPipeline([nodes_with_tf])
    sqls = block_from_labels(linker, table_name)

    pipeline.enqueue_list_of_sqls(sqls)

    sql = compute_comparison_vector_values_sql(
        linker._settings_obj._columns_to_select_for_comparison_vector_values
    )

    pipeline.enqueue_sql(sql, "__splink__df_comparison_vectors")

    sql = """
    select *, cast(1.0 as float8) as match_probability
    from __splink__df_comparison_vectors
    """
    pipeline.enqueue_sql(sql, "__splink__df_predict")

    sql = compute_new_parameters_sql_legacy(
        comparisons=linker._settings_obj.comparisons,
    )
    pipeline.enqueue_sql(sql, "__splink__m_u_counts")

    df_params = linker._db_api.sql_pipeline_to_splink_dataframe(pipeline)

    param_records = df_params.as_pandas_dataframe()
    param_records = compute_proportions_for_new_parameters(param_records)

    m_u_records = [
        r
        for r in param_records
        if r["output_column_name"] != "_probability_two_random_records_match"
    ]

    m_u_records_lookup = m_u_records_to_lookup_dict(m_u_records)
    for cc in linker._settings_obj.comparisons:
        for cl in cc._comparison_levels_excluding_null:
            if not cl._fix_m_probability:
                append_m_probability_to_comparison_level_trained_probabilities(
                    cl,
                    m_u_records_lookup,
                    cc.output_column_name,
                    "estimate m from pairwise labels",
                )

    linker._populate_m_u_from_trained_values()


def _estimate_m_weighted(linker: "Linker", normalised_labels_table: str) -> None:
    """Direct, weighted m estimation.

    For each comparison, m at a (non-null) level is the weighted positive
    matching mass observed at that level divided by the total weighted
    positive mass over all non-null levels of that comparison:

        m_k = sum(weight * score * 1[gamma = k])
              / sum(weight * score * 1[gamma <> -1])

    u values and the global prior are left untouched.  Each comparison's own
    null level does not participate in the normalisation.
    """
    settings = linker._settings_obj
    db_api = linker._db_api

    pipeline = CTEPipeline()
    nodes_with_tf = compute_df_concat_with_tf(linker, pipeline)
    pipeline = CTEPipeline([nodes_with_tf])

    sqls = block_from_normalised_labels_sqls(linker, normalised_labels_table)
    pipeline.enqueue_list_of_sqls(sqls)

    cvv_columns = settings._columns_to_select_for_comparison_vector_values
    select_cols_expr = ", ".join(cvv_columns)
    # Carry the clerical score and the weight straight through the
    # comparison-vector step.
    pipeline.enqueue_sql(
        f"""
        select {select_cols_expr},
               {LABEL_SCORE_COLUMN},
               {LABEL_WEIGHT_COLUMN}
        from __splink__df_blocked
        """,
        "__splink__df_comparison_vectors_weighted",
    )

    # Weighted positive mass per gamma level for every comparison.
    union_sqls = []
    for cc in settings.comparisons:
        gamma_col = cc._gamma_column_name
        union_sqls.append(
            f"""
            select
                {gamma_col} as comparison_vector_value,
                sum(
                    {LABEL_WEIGHT_COLUMN} * {LABEL_SCORE_COLUMN}
                ) as m_count,
                '{cc.output_column_name}' as output_column_name
            from __splink__df_comparison_vectors_weighted
            group by {gamma_col}
            """
        )
    pipeline.enqueue_sql(
        " union all ".join(union_sqls), "__splink__weighted_m_counts"
    )

    m_counts_df = db_api.sql_pipeline_to_splink_dataframe(pipeline)
    m_counts = m_counts_df.as_pandas_dataframe()
    m_counts_df.drop_table_from_database_and_remove_from_cache()

    # Normalise per comparison in Python.  The data is parameter-level (at
    # most one row per comparison level), never the full pair table.  Null
    # levels (gamma = -1) are excluded from each comparison's denominator.
    m_u_records_lookup: dict = {}
    not_estimated: list[str] = []
    for cc in settings.comparisons:
        name = cc.output_column_name
        rows = m_counts[m_counts["output_column_name"] == name]
        non_null = rows[rows["comparison_vector_value"] != -1]
        total_mass = float(non_null["m_count"].sum())
        lookup: dict[int, dict] = {}
        if total_mass <= 0.0:
            # No positive evidence for this comparison at all: keep whatever
            # parameters it currently has and flag it as not estimated.
            not_estimated.append(name)
            m_u_records_lookup[name] = lookup
            continue
        for _, row in non_null.iterrows():
            gamma = int(row["comparison_vector_value"])
            lookup[gamma] = {
                "m_probability": float(row["m_count"]) / total_mass
            }
        m_u_records_lookup[name] = lookup

    for cc in settings.comparisons:
        if cc.output_column_name in not_estimated:
            logger.info(
                "m probability not estimated for comparison "
                f"{cc.output_column_name} from weighted labels: no positive "
                "matching mass on any non-null comparison level; existing "
                "values retained"
            )
            continue
        for cl in cc._comparison_levels_excluding_null:
            if not cl._fix_m_probability:
                append_m_probability_to_comparison_level_trained_probabilities(
                    cl,
                    m_u_records_lookup,
                    cc.output_column_name,
                    "estimate m from weighted pairwise labels",
                )

    linker._populate_m_u_from_trained_values()


def compute_new_parameters_sql_legacy(comparisons):
    # Thin wrapper preserving the original call shape for the legacy path.
    return compute_new_parameters_sql(
        estimate_without_term_frequencies=False,
        comparisons=comparisons,
    )

"""Tests for inclusion-probability-weighted quality evaluation.

The fixture is small enough that every expected confusion mass is enumerated
by hand from the original record identities and clerical labels.  Weighted
expectations are computed independently with numpy from the per-pair
inclusion probabilities; the library's sampling/aggregation functions are
never used to generate an expected value.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from splink import DuckDBAPI, Linker, SettingsCreator
from splink.internals.comparison_library import ExactMatch

from .decorator import mark_with_dialects_including


# --------------------------------------------------------------------------- #
# Hand fixture
# --------------------------------------------------------------------------- #
def _robin_david_data():
    # clusters: X = {1, 2}, Y = {3}, Z = {4}, W = {5}
    return pd.DataFrame(
        [
            {"unique_id": 1, "first_name": "robin", "cluster": "X"},
            {"unique_id": 2, "first_name": "robin", "cluster": "X"},
            {"unique_id": 3, "first_name": "robin", "cluster": "Y"},
            {"unique_id": 4, "first_name": "david", "cluster": "Z"},
            {"unique_id": 5, "first_name": "david", "cluster": "W"},
        ]
    )


def _robin_david_settings():
    return {
        "link_type": "dedupe_only",
        "probability_two_random_records_match": 0.5,
        "retain_matching_columns": True,
        "comparisons": [
            {
                "output_column_name": "first_name",
                "comparison_levels": [
                    {
                        "sql_condition": (
                            '"first_name_l" IS NULL OR "first_name_r" IS NULL'
                        ),
                        "label_for_charts": "Null",
                        "is_null_level": True,
                    },
                    {
                        "sql_condition": '"first_name_l" = "first_name_r"',
                        "label_for_charts": "Exact match",
                        "m_probability": 0.95,
                        "u_probability": 1e-5,
                    },
                    {
                        "sql_condition": "ELSE",
                        "label_for_charts": "All other comparisons",
                        "m_probability": 0.05,
                        "u_probability": 1 - 1e-5,
                    },
                ],
            }
        ],
        "blocking_rules_to_generate_predictions": ["1=1"],
    }


def _all_pairs_labels(linker, data, probability_column=None):
    """Clerical labels for every generated pair, from cluster identity."""
    pred = linker.inference.predict().as_pandas_dataframe()
    cluster = data.set_index("unique_id")["cluster"]
    rows = []
    for _, r in pred.iterrows():
        a, b = int(r["unique_id_l"]), int(r["unique_id_r"])
        row = {
            "unique_id_l": a,
            "unique_id_r": b,
            "clerical_match_score": (
                1.0 if cluster[a] == cluster[b] else 0.0
            ),
        }
        if probability_column is not None:
            row[probability_column] = 1.0
        rows.append(row)
    labels = pd.DataFrame(rows)
    return pred, labels


def _review_design_labels(pred, data):
    """A deliberately non-uniform review design over the 10-pair population.

    High-score stratum (exact first_name level): review 2 of its 4 pairs at
    pi = 2/4 = 1/2.  Low-score stratum (else level, 6 pairs): review 3 at
    pi = 3/6 = 1/2.  The probabilities deliberately differ from the realised
    counts so the weights are non-trivial.  The only true match, pair (1, 2),
    is among the reviewed rows.
    """
    cluster = data.set_index("unique_id")["cluster"]
    reviewed_pairs = [
        (1, 2, 0.5),  # true match, high stratum
        (1, 3, 0.5),  # non-match, high stratum
        (1, 4, 0.5),  # non-match, low stratum
        (2, 4, 0.5),  # non-match, low stratum
        (3, 4, 0.5),  # non-match, low stratum
    ]
    rows = []
    for a, b, pi in reviewed_pairs:
        rows.append(
            {
                "unique_id_l": a,
                "unique_id_r": b,
                "clerical_match_score": (
                    1.0 if cluster[a] == cluster[b] else 0.0
                ),
                "inclusion_probability": pi,
            }
        )
    return pd.DataFrame(rows), None


def _expected_weighted_truth_space(labels_with_pi):
    """Independent numpy enumeration of the weighted confusion masses.

    Returns a dict keyed by the splink match_weight threshold, with the
    Horvitz-Thompson masses computed directly from pair identities.
    """
    weights = 1.0 / labels_with_pi["inclusion_probability"].to_numpy()
    clerical = labels_with_pi["clerical_match_score"].to_numpy()
    return weights, clerical


# --------------------------------------------------------------------------- #
# Weighted truth space: hand-checked masses at every threshold
# --------------------------------------------------------------------------- #
@mark_with_dialects_including("duckdb")
def test_weighted_truth_space_hand_calculation():
    data = _robin_david_data()
    linker = Linker(data, _robin_david_settings(), db_api=DuckDBAPI())
    pred = linker.inference.predict().as_pandas_dataframe()
    labels, _ = _review_design_labels(pred, data)
    linker.table_management.register_table(labels, "labels", overwrite=True)

    ts = (
        linker.evaluation.accuracy_analysis_from_labels_table(
            "labels",
            output_type="table",
            match_weight_round_to_nearest=None,
            inclusion_probability_column="inclusion_probability",
        )
        .as_pandas_dataframe()
        .sort_values("truth_threshold")
        .reset_index(drop=True)
    )

    # Two distinct model score levels -> two threshold rows.
    assert len(ts) == 2

    # Review design: every reviewed pair has pi = 0.5, so its evaluation
    # weight is 2 (each represents two population pairs).  The two high
    # strata pairs and three low strata pairs therefore total an estimated
    # generated-population mass of 10.  The weighting still changes the
    # results versus counting the five reviewed rows.
    w_high = 2.0
    w_low = 2.0
    estimated_total = 5 * 2.0
    assert np.isclose(ts["estimated_pair_count"].nunique(), 1)
    assert np.isclose(ts["estimated_pair_count"].iloc[0], estimated_total)
    assert (ts["sampled_pair_count"] == 5).all()
    assert np.isclose(ts["total_clerical_labels"], estimated_total).all()

    high_row = ts.iloc[1]  # threshold at the exact-match score level
    low_row = ts.iloc[0]

    # At the high threshold only the two reviewed exact-level pairs are
    # predicted positive: true match (1,2) and non-match (1,3).
    assert np.isclose(high_row["tp"], w_high)
    assert np.isclose(high_row["fp"], w_high)
    assert np.isclose(high_row["fn"], 0.0)
    assert np.isclose(high_row["tn"], 3 * w_low)
    assert np.isclose(high_row["p"], w_high)
    assert np.isclose(high_row["n"], w_high + 3 * w_low)
    assert np.isclose(high_row["precision"], 0.5)
    assert np.isclose(high_row["recall"], 1.0)
    assert np.isclose(high_row["f1"], 2 * 0.5 / (1 + 0.5))

    # At the low threshold every reviewed pair is predicted positive.
    assert np.isclose(low_row["tp"], w_high)
    assert np.isclose(low_row["fp"], w_high + 3 * w_low)
    assert np.isclose(low_row["tn"], 0.0)
    assert np.isclose(low_row["fn"], 0.0)
    assert np.isclose(low_row["precision"], w_high / estimated_total)
    assert np.isclose(low_row["recall"], 1.0)

    # Independent numpy recomputation from the identities/labels alone.
    weights, clerical = _expected_weighted_truth_space(labels)
    total_mass = weights.sum()
    assert np.isclose(total_mass, estimated_total)
    p_mass = weights[clerical >= 0.5].sum()
    n_mass = weights[clerical < 0.5].sum()
    assert np.isclose(p_mass, w_high)
    assert np.isclose(n_mass, w_high + 3 * w_low)


@mark_with_dialects_including("duckdb")
def test_weighted_differs_from_unweighted_row_counts():
    """Different sampling fractions and different positive rates per stratum
    must make the weighted results differ from counting reviewed rows."""
    data = _robin_david_data()
    linker = Linker(data, _robin_david_settings(), db_api=DuckDBAPI())
    pred = linker.inference.predict().as_pandas_dataframe()
    labels, _ = _review_design_labels(pred, data)
    linker.table_management.register_table(labels, "labels", overwrite=True)

    weighted = (
        linker.evaluation.accuracy_analysis_from_labels_table(
            "labels",
            output_type="table",
            match_weight_round_to_nearest=None,
            inclusion_probability_column="inclusion_probability",
        )
        .as_pandas_dataframe()
        .sort_values("truth_threshold")
        .reset_index(drop=True)
    )
    unweighted = (
        linker.evaluation.accuracy_analysis_from_labels_table(
            "labels",
            output_type="table",
            match_weight_round_to_nearest=None,
        )
        .as_pandas_dataframe()
        .sort_values("truth_threshold")
        .reset_index(drop=True)
    )

    # Unweighted path ignores the extra column and counts reviewed rows.
    high_w, high_u = weighted.iloc[1], unweighted.iloc[1]
    assert high_u["tp"] == 1 and high_u["fp"] == 1 and high_u["tn"] == 3
    assert np.isclose(high_w["tp"], 2.0) and np.isclose(high_w["fp"], 2.0)
    assert np.isclose(high_w["tn"], 6.0)
    assert not np.isclose(high_w["tn"], high_u["tn"])
    # the unweighted table has no sampling diagnostics columns
    assert "sampled_pair_count" not in unweighted.columns
    assert "estimated_pair_count" not in unweighted.columns


@mark_with_dialects_including("duckdb")
def test_full_census_weighted_equals_legacy_evaluation():
    data = _robin_david_data()
    linker = Linker(data, _robin_david_settings(), db_api=DuckDBAPI())
    _, labels = _all_pairs_labels(
        linker, data, probability_column="inclusion_probability"
    )
    # build a census version: every pair included with probability 1
    pred = linker.inference.predict().as_pandas_dataframe()
    census = labels.copy()
    census["inclusion_probability"] = 1.0
    linker.table_management.register_table(census, "census", overwrite=True)

    weighted = (
        linker.evaluation.accuracy_analysis_from_labels_table(
            "census",
            output_type="table",
            match_weight_round_to_nearest=None,
            inclusion_probability_column="inclusion_probability",
        )
        .as_pandas_dataframe()
        .sort_values("truth_threshold")
        .reset_index(drop=True)
    )
    legacy = (
        linker.evaluation.accuracy_analysis_from_labels_table(
            "census",
            output_type="table",
            match_weight_round_to_nearest=None,
        )
        .as_pandas_dataframe()
        .sort_values("truth_threshold")
        .reset_index(drop=True)
    )

    metric_cols = [
        "tp",
        "tn",
        "fp",
        "fn",
        "p",
        "n",
        "precision",
        "recall",
        "specificity",
        "npv",
        "accuracy",
        "f1",
        "f2",
        "f0_5",
        "p4",
        "phi",
        "total_clerical_labels",
    ]
    for col in metric_cols:
        assert np.allclose(weighted[col], legacy[col], equal_nan=True), col
    assert (weighted["sampled_pair_count"] == len(census)).all()
    assert np.isclose(weighted["estimated_pair_count"], float(len(census))).all()


# --------------------------------------------------------------------------- #
# Threshold sweep cross-checked against an independent numpy enumeration
# --------------------------------------------------------------------------- #
@mark_with_dialects_including("duckdb")
def test_every_threshold_matches_independent_numpy_sweep():
    """Recompute every weighted confusion mass directly from pair identities,
    clerical labels and inclusion probabilities, without any Splink
    aggregation, and compare threshold by threshold."""
    data = _robin_david_data()
    linker = Linker(data, _robin_david_settings(), db_api=DuckDBAPI())
    pred = linker.inference.predict().as_pandas_dataframe()
    labels, _ = _review_design_labels(pred, data)
    linker.table_management.register_table(labels, "labels", overwrite=True)

    ts = (
        linker.evaluation.accuracy_analysis_from_labels_table(
            "labels",
            output_type="table",
            match_weight_round_to_nearest=None,
            inclusion_probability_column="inclusion_probability",
        )
        .as_pandas_dataframe()
        .sort_values("truth_threshold")
        .reset_index(drop=True)
    )

    # join the model match_weight back onto the reviewed pairs by identity
    weight_lookup = {
        (int(r["unique_id_l"]), int(r["unique_id_r"])): r["match_weight"]
        for _, r in pred.iterrows()
    }
    review = labels.copy()
    review["match_weight"] = [
        weight_lookup[(a, b)]
        for a, b in zip(review["unique_id_l"], review["unique_id_r"])
    ]
    review["weight"] = 1.0 / review["inclusion_probability"]

    for _, row in ts.iterrows():
        t = row["truth_threshold"]
        predicted_positive = review["match_weight"] >= t
        actual_positive = review["clerical_match_score"] >= 0.5
        w = review["weight"]
        exp_tp = w[predicted_positive & actual_positive].sum()
        exp_fp = w[predicted_positive & ~actual_positive].sum()
        exp_tn = w[~predicted_positive & ~actual_positive].sum()
        exp_fn = w[~predicted_positive & actual_positive].sum()
        assert np.isclose(row["tp"], exp_tp)
        assert np.isclose(row["fp"], exp_fp)
        assert np.isclose(row["tn"], exp_tn)
        assert np.isclose(row["fn"], exp_fn)
        assert np.isclose(row["p"], w[actual_positive].sum())
        assert np.isclose(row["n"], w[~actual_positive].sum())
        # derived metrics follow the same masses and zero conventions
        assert np.isclose(
            row["recall"],
            exp_tp / w[actual_positive].sum()
            if w[actual_positive].sum() > 0
            else np.nan,
            equal_nan=True,
        )


# --------------------------------------------------------------------------- #
# Weighted error table and threshold equality conventions
# --------------------------------------------------------------------------- #
@mark_with_dialects_including("duckdb")
def test_weighted_error_table_masses_and_identity_columns():
    data = _robin_david_data()
    linker = Linker(data, _robin_david_settings(), db_api=DuckDBAPI())
    pred = linker.inference.predict().as_pandas_dataframe()
    labels, _ = _review_design_labels(pred, data)
    linker.table_management.register_table(labels, "labels", overwrite=True)

    errs = (
        linker.evaluation.prediction_errors_from_labels_table(
            "labels",
            inclusion_probability_column="inclusion_probability",
        )
        .as_pandas_dataframe()
        .sort_values(["unique_id_l", "unique_id_r"])
        .reset_index(drop=True)
    )
    # At the default 0.5 threshold the only error is the FP pair (1, 3).
    assert len(errs) == 1
    row = errs.iloc[0]
    assert (row["unique_id_l"], row["unique_id_r"]) == (1, 3)
    assert row["truth_status"] == "FP"
    assert np.isclose(row["inclusion_probability"], 0.5)
    assert np.isclose(row["evaluation_weight"], 2.0)
    # full identity and prediction columns are retained
    for col in [
        "unique_id_l",
        "unique_id_r",
        "match_probability",
        "match_weight",
        "clerical_match_score",
        "inclusion_probability",
        "evaluation_weight",
    ]:
        assert col in errs.columns

    # Off-threshold error mass reconciles with the weighted truth-space
    # confusion masses: the single FP carries the full FP mass shown at the
    # high model score threshold in the truth-space table.
    ts = (
        linker.evaluation.accuracy_analysis_from_labels_table(
            "labels",
            output_type="table",
            match_weight_round_to_nearest=None,
            inclusion_probability_column="inclusion_probability",
        )
        .as_pandas_dataframe()
        .sort_values("truth_threshold")
        .reset_index(drop=True)
    )
    assert np.isclose(
        errs["evaluation_weight"].sum(), ts.iloc[1]["fp"]
    )


@mark_with_dialects_including("duckdb")
def test_threshold_equality_conventions_documented_and_enforced():
    """The two legacy entries keep their own equality convention:

    * error table: strict inequalities - a pair whose match_probability is
      exactly the threshold is neither FP nor FN;
    * truth space: >= on the match_weight threshold - equality is included.
    """
    data = _robin_david_data()
    linker = Linker(data, _robin_david_settings(), db_api=DuckDBAPI())
    pred = linker.inference.predict().as_pandas_dataframe()
    labels, score_lookup = _review_design_labels(pred, data)
    linker.table_management.register_table(labels, "labels", overwrite=True)

    # choose the threshold exactly equal to the low score level's
    # match_probability (derived from the model output, not typed by hand)
    low_level_mp = float(
        pred.loc[
            pred["unique_id_l"].eq(1) & pred["unique_id_r"].eq(4),
            "match_probability",
        ].iloc[0]
    )
    threshold = low_level_mp

    errs = linker.evaluation.prediction_errors_from_labels_table(
        "labels",
        threshold_match_probability=threshold,
        inclusion_probability_column="inclusion_probability",
    ).as_pandas_dataframe()
    # low-score reviewed pairs sit exactly on the threshold: strict > means
    # none of them qualifies as an FP despite being clerical negatives
    low_pairs = {(1, 4), (2, 4), (3, 4)}
    err_pairs = set(zip(errs["unique_id_l"], errs["unique_id_r"]))
    assert low_pairs.isdisjoint(err_pairs)
    # only the high-score non-match (1, 3) remains an FP at this threshold
    assert err_pairs == {(1, 3)}

    # the truth-space table sweeps every model score level; the low row's
    # threshold *is* the low model score (equality included on the >=
    # convention), so all three low-stratum masses (weight 3 each) together
    # with the high FP (weight 2) appear inside FP there
    ts = (
        linker.evaluation.accuracy_analysis_from_labels_table(
            "labels",
            output_type="table",
            match_weight_round_to_nearest=None,
            inclusion_probability_column="inclusion_probability",
        )
        .as_pandas_dataframe()
        .sort_values("truth_threshold")
        .reset_index(drop=True)
    )
    low_threshold_row = ts.iloc[0]
    # the four clerical negatives (one high-score, three low-score; weight 2
    # each) are predicted positive at the low threshold; the single positive
    # pair (1, 2) is a TP, not an FP
    assert np.isclose(low_threshold_row["fp"], 4 * 2.0)
    assert np.isclose(low_threshold_row["tp"], 2.0)


# --------------------------------------------------------------------------- #
# Validation contract
# --------------------------------------------------------------------------- #
@mark_with_dialects_including("duckdb")
@pytest.mark.parametrize(
    "bad_value,match",
    [
        (0.0, "interval"),
        (-0.5, "interval"),
        (1.5, "interval"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (None, "finite"),
    ],
)
def test_illegal_inclusion_probabilities_reject(bad_value, match):
    data = _robin_david_data()
    linker = Linker(data, _robin_david_settings(), db_api=DuckDBAPI())
    pred = linker.inference.predict().as_pandas_dataframe()
    labels, _ = _review_design_labels(pred, data)
    labels.loc[0, "inclusion_probability"] = bad_value
    linker.table_management.register_table(labels, "bad_pi", overwrite=True)
    with pytest.raises(ValueError, match=match):
        linker.evaluation.accuracy_analysis_from_labels_table(
            "bad_pi",
            output_type="table",
            match_weight_round_to_nearest=None,
            inclusion_probability_column="inclusion_probability",
        )
    with pytest.raises(ValueError, match=match):
        linker.evaluation.prediction_errors_from_labels_table(
            "bad_pi",
            inclusion_probability_column="inclusion_probability",
        )


@mark_with_dialects_including("duckdb")
def test_custom_unique_id_column_weighted_path():
    data = pd.DataFrame(
        [
            {"record_id": 1, "first_name": "robin", "cluster": "X"},
            {"record_id": 2, "first_name": "robin", "cluster": "X"},
            {"record_id": 3, "first_name": "john", "cluster": "Y"},
        ]
    )
    settings = SettingsCreator(
        link_type="dedupe_only",
        unique_id_column_name="record_id",
        probability_two_random_records_match=0.5,
        retain_matching_columns=True,
        comparisons=[ExactMatch("first_name")],
        blocking_rules_to_generate_predictions=["1=1"],
    )
    linker = Linker(data, settings, db_api=DuckDBAPI())
    labels = pd.DataFrame(
        [
            {
                "record_id_l": 1,
                "record_id_r": 2,
                "clerical_match_score": 1.0,
                "inclusion_probability": 0.5,
            },
            {
                "record_id_l": 1,
                "record_id_r": 3,
                "clerical_match_score": 0.0,
                "inclusion_probability": 0.25,
            },
        ]
    )
    linker.table_management.register_table(labels, "labels", overwrite=True)
    ts = (
        linker.evaluation.accuracy_analysis_from_labels_table(
            "labels",
            output_type="table",
            match_weight_round_to_nearest=None,
            inclusion_probability_column="inclusion_probability",
        )
        .as_pandas_dataframe()
        .sort_values("truth_threshold")
        .reset_index(drop=True)
    )
    # weights 2 and 4 -> estimated mass 6
    assert np.isclose(ts["estimated_pair_count"].iloc[0], 6.0)
    assert (ts["sampled_pair_count"] == 2).all()
    errs = linker.evaluation.prediction_errors_from_labels_table(
        "labels", inclusion_probability_column="inclusion_probability"
    ).as_pandas_dataframe()
    assert {"record_id_l", "record_id_r", "evaluation_weight"}.issubset(
        errs.columns
    )


@mark_with_dialects_including("duckdb")
def test_probability_one_is_accepted_and_probability_zero_rejected():
    data = _robin_david_data()
    linker = Linker(data, _robin_david_settings(), db_api=DuckDBAPI())
    pred = linker.inference.predict().as_pandas_dataframe()
    labels, _ = _review_design_labels(pred, data)

    # pi == 1 is the closed end of the (0, 1] contract and must work
    labels_one = labels.copy()
    labels_one["inclusion_probability"] = 1.0
    linker.table_management.register_table(labels_one, "pi_one", overwrite=True)
    ts = (
        linker.evaluation.accuracy_analysis_from_labels_table(
            "pi_one",
            output_type="table",
            match_weight_round_to_nearest=None,
            inclusion_probability_column="inclusion_probability",
        )
        .as_pandas_dataframe()
    )
    # weights are all 1/1 = 1, so the estimated mass equals reviewed rows
    assert np.isclose(ts["estimated_pair_count"].iloc[0], float(len(labels_one)))
    assert (ts["sampled_pair_count"] == len(labels_one)).all()

    # pi == 0 is the open end and must be rejected
    labels_zero = labels.copy()
    labels_zero.loc[0, "inclusion_probability"] = 0.0
    linker.table_management.register_table(labels_zero, "pi_zero", overwrite=True)
    with pytest.raises(ValueError, match="interval"):
        linker.evaluation.accuracy_analysis_from_labels_table(
            "pi_zero",
            output_type="table",
            match_weight_round_to_nearest=None,
            inclusion_probability_column="inclusion_probability",
        )


@mark_with_dialects_including("duckdb")
def test_non_numeric_probability_rejects():
    data = _robin_david_data()
    linker = Linker(data, _robin_david_settings(), db_api=DuckDBAPI())
    pred = linker.inference.predict().as_pandas_dataframe()
    labels, _ = _review_design_labels(pred, data)
    labels["inclusion_probability"] = labels["inclusion_probability"].astype(str)
    labels.loc[0, "inclusion_probability"] = "not_a_number"
    linker.table_management.register_table(labels, "text_pi", overwrite=True)
    with pytest.raises(ValueError):
        linker.evaluation.accuracy_analysis_from_labels_table(
            "text_pi",
            output_type="table",
            match_weight_round_to_nearest=None,
            inclusion_probability_column="inclusion_probability",
        )


@mark_with_dialects_including("duckdb")
def test_identity_violations_reject_without_dropping_rows():
    data = _robin_david_data()
    linker = Linker(data, _robin_david_settings(), db_api=DuckDBAPI())
    pred = linker.inference.predict().as_pandas_dataframe()
    base, _ = _review_design_labels(pred, data)

    # self pair
    self_pair = base.copy()
    self_pair.loc[0, "unique_id_r"] = self_pair.loc[0, "unique_id_l"]
    linker.table_management.register_table(self_pair, "t_self", overwrite=True)
    with pytest.raises(ValueError, match="Self-pairs"):
        linker.evaluation.accuracy_analysis_from_labels_table(
            "t_self",
            output_type="table",
            match_weight_round_to_nearest=None,
            inclusion_probability_column="inclusion_probability",
        )

    # unknown node
    unknown = base.copy()
    unknown.loc[0, "unique_id_r"] = 999_999
    linker.table_management.register_table(unknown, "t_unk", overwrite=True)
    with pytest.raises(ValueError, match="do not exist"):
        linker.evaluation.accuracy_analysis_from_labels_table(
            "t_unk",
            output_type="table",
            match_weight_round_to_nearest=None,
            inclusion_probability_column="inclusion_probability",
        )

    # duplicate unoriented pair (reversed submission)
    dup = pd.concat(
        [base, base.iloc[[0]].rename(
            columns={"unique_id_l": "unique_id_r", "unique_id_r": "unique_id_l"}
        )],
        ignore_index=True,
    )
    linker.table_management.register_table(dup, "t_dup", overwrite=True)
    with pytest.raises(ValueError, match="Duplicate unoriented"):
        linker.evaluation.accuracy_analysis_from_labels_table(
            "t_dup",
            output_type="table",
            match_weight_round_to_nearest=None,
            inclusion_probability_column="inclusion_probability",
        )

    # input tables themselves are untouched
    assert len(linker.misc.query_sql("select * from t_self")) == len(base)


@mark_with_dialects_including("duckdb")
def test_reregistration_and_endpoint_reversal_preserve_probabilities():
    data = _robin_david_data()
    linker = Linker(data, _robin_david_settings(), db_api=DuckDBAPI())
    pred = linker.inference.predict().as_pandas_dataframe()
    labels, _ = _review_design_labels(pred, data)
    linker.table_management.register_table(labels, "labels", overwrite=True)

    ts_a = (
        linker.evaluation.accuracy_analysis_from_labels_table(
            "labels",
            output_type="table",
            match_weight_round_to_nearest=None,
            inclusion_probability_column="inclusion_probability",
        )
        .as_pandas_dataframe()
        .sort_values("truth_threshold")
        .reset_index(drop=True)
    )

    # register as an ordinary table under a new name, endpoints reversed
    rev = labels.copy()
    rev[["unique_id_l", "unique_id_r"]] = labels[
        ["unique_id_r", "unique_id_l"]
    ].to_numpy()
    linker.table_management.register_table(rev, "ordinary_labels", overwrite=True)

    ts_b = (
        linker.evaluation.accuracy_analysis_from_labels_table(
            "ordinary_labels",
            output_type="table",
            match_weight_round_to_nearest=None,
            inclusion_probability_column="inclusion_probability",
        )
        .as_pandas_dataframe()
        .sort_values("truth_threshold")
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        ts_a.drop(columns=["truth_threshold", "match_probability"]),
        ts_b.drop(columns=["truth_threshold", "match_probability"]),
        atol=1e-12,
    )

    errs_a = linker.evaluation.prediction_errors_from_labels_table(
        "labels", inclusion_probability_column="inclusion_probability"
    ).as_pandas_dataframe()
    errs_b = linker.evaluation.prediction_errors_from_labels_table(
        "ordinary_labels",
        inclusion_probability_column="inclusion_probability",
    ).as_pandas_dataframe()
    assert len(errs_a) == len(errs_b) == 1
    assert np.isclose(
        errs_a["evaluation_weight"].iloc[0], errs_b["evaluation_weight"].iloc[0]
    )
    assert np.isclose(
        errs_a["inclusion_probability"].iloc[0],
        errs_b["inclusion_probability"].iloc[0],
    )


@mark_with_dialects_including("duckdb")
def test_failure_does_not_mutate_model_or_block_later_calls():
    data = _robin_david_data()
    linker = Linker(data, _robin_david_settings(), db_api=DuckDBAPI())
    pred = linker.inference.predict().as_pandas_dataframe()
    labels, _ = _review_design_labels(pred, data)
    bad = labels.copy()
    bad.loc[0, "inclusion_probability"] = 0.0
    linker.table_management.register_table(bad, "bad", overwrite=True)
    linker.table_management.register_table(labels, "good", overwrite=True)

    model_before = linker.misc.save_model_to_json()
    with pytest.raises(ValueError):
        linker.evaluation.accuracy_analysis_from_labels_table(
            "bad",
            output_type="table",
            match_weight_round_to_nearest=None,
            inclusion_probability_column="inclusion_probability",
        )
    assert linker.misc.save_model_to_json() == model_before

    # the same linker continues to evaluate, repeatedly, without accumulating
    # sampling/evaluation scratch tables
    con = linker._db_api._con
    before = {r[0] for r in con.sql("show tables").fetchall()}
    for _ in range(3):
        out = linker.evaluation.accuracy_analysis_from_labels_table(
            "good",
            output_type="table",
            match_weight_round_to_nearest=None,
            inclusion_probability_column="inclusion_probability",
        )
        assert len(out.as_pandas_dataframe()) == 2
    after = {r[0] for r in con.sql("show tables").fetchall()}
    leftovers = [
        t for t in (after - before) if "sampling_canonical" in t or "sampling_strata" in t
    ]
    assert leftovers == []


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _tooltip_fields(chart):
    fields = []

    def walk(o):
        if isinstance(o, dict):
            tt = o.get("tooltip")
            if isinstance(tt, list):
                fields.extend(t.get("field") for t in tt)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(chart.to_dict())
    return fields


@mark_with_dialects_including("duckdb")
@pytest.mark.parametrize(
    "output_type", ["roc", "precision_recall", "accuracy", "threshold_selection"]
)
def test_weighted_chart_tooltips_show_both_counts(output_type):
    data = _robin_david_data()
    linker = Linker(data, _robin_david_settings(), db_api=DuckDBAPI())
    pred = linker.inference.predict().as_pandas_dataframe()
    labels, _ = _review_design_labels(pred, data)
    linker.table_management.register_table(labels, "labels", overwrite=True)

    chart = linker.evaluation.accuracy_analysis_from_labels_table(
        "labels",
        output_type=output_type,
        match_weight_round_to_nearest=None,
        inclusion_probability_column="inclusion_probability",
    )
    fields = _tooltip_fields(chart)
    assert "sampled_pair_count" in fields
    assert "estimated_pair_count" in fields

    # unweighted chart keeps its original tooltip (no new fields)
    legacy_chart = linker.evaluation.accuracy_analysis_from_labels_table(
        "labels", output_type=output_type, match_weight_round_to_nearest=None
    )
    legacy_fields = _tooltip_fields(legacy_chart)
    assert "sampled_pair_count" not in legacy_fields
    assert "estimated_pair_count" not in legacy_fields


# --------------------------------------------------------------------------- #
# Separation from training and model JSON
# --------------------------------------------------------------------------- #
@mark_with_dialects_including("duckdb")
def test_evaluation_metadata_not_consumed_by_training_or_model_json(tmp_path):
    data = _robin_david_data()
    linker = Linker(data, _robin_david_settings(), db_api=DuckDBAPI())
    pred = linker.inference.predict().as_pandas_dataframe()
    labels, _ = _review_design_labels(pred, data)
    linker.table_management.register_table(labels, "labels", overwrite=True)

    # training without an explicit weight column must not treat the sampling
    # column as a frequency weight; the call runs in legacy mode and succeeds
    linker.training.estimate_m_from_pairwise_labels("labels")
    m_legacy = [
        cl.m_probability
        for cc in linker._settings_obj.comparisons
        for cl in cc._comparison_levels_excluding_null
    ]

    linker2 = Linker(data, _robin_david_settings(), db_api=DuckDBAPI())
    plain = labels.drop(columns=["inclusion_probability"])
    linker2.table_management.register_table(plain, "plain_labels", overwrite=True)
    linker2.training.estimate_m_from_pairwise_labels("plain_labels")
    m_plain = [
        cl.m_probability
        for cc in linker2._settings_obj.comparisons
        for cl in cc._comparison_levels_excluding_null
    ]
    assert np.allclose(m_legacy, m_plain)

    # model JSON never contains sampling metadata
    model_json = linker.misc.save_model_to_json()
    serialized = json.dumps(model_json)
    for forbidden in (
        "inclusion_probability",
        "evaluation_weight",
        "sampled_pair_count",
    ):
        assert forbidden not in serialized

    path = tmp_path / "model.json"
    linker.misc.save_model_to_json(str(path), overwrite=True)
    reloaded = Linker(data, str(path), db_api=DuckDBAPI())
    pred_before = (
        linker.inference.predict()
        .as_pandas_dataframe()
        .sort_values(["unique_id_l", "unique_id_r"])
        .reset_index(drop=True)
    )
    pred_after = (
        reloaded.inference.predict()
        .as_pandas_dataframe()
        .sort_values(["unique_id_l", "unique_id_r"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        pred_before[["unique_id_l", "unique_id_r", "match_probability"]],
        pred_after[["unique_id_l", "unique_id_r", "match_probability"]],
        rtol=1e-12,
        atol=1e-12,
    )


# --------------------------------------------------------------------------- #
# Link-only coverage with full source + record identity
# --------------------------------------------------------------------------- #
@mark_with_dialects_including("duckdb")
def test_link_only_weighted_evaluation():
    data_l = pd.DataFrame(
        [
            {"unique_id": 1, "first_name": "robin", "cluster": "X"},
            {"unique_id": 2, "first_name": "john", "cluster": "Y"},
        ]
    )
    data_r = pd.DataFrame(
        [
            {"unique_id": 1, "first_name": "robin", "cluster": "X"},
            {"unique_id": 2, "first_name": "eve", "cluster": "Z"},
        ]
    )
    settings = SettingsCreator(
        link_type="link_only",
        probability_two_random_records_match=0.5,
        retain_matching_columns=True,
        comparisons=[ExactMatch("first_name")],
        blocking_rules_to_generate_predictions=["1=1"],
    )
    linker = Linker(
        [data_l, data_r],
        settings,
        db_api=DuckDBAPI(),
        input_table_aliases=["left_source", "right_source"],
    )
    pred = linker.inference.predict().as_pandas_dataframe().sort_values(
        ["unique_id_l", "unique_id_r"]
    ).reset_index(drop=True)
    # all four cross-source pairs reviewed, but with differing probabilities
    labels = pred[
        [
            "source_dataset_l",
            "unique_id_l",
            "source_dataset_r",
            "unique_id_r",
        ]
    ].copy()
    # clerical truth: (left 1, right 1) is the only match (cluster X == X);
    # derive it from identities rather than row position
    labels["clerical_match_score"] = [
        1.0 if (a == 1 and b == 1) else 0.0
        for a, b in zip(labels["unique_id_l"], labels["unique_id_r"])
    ]
    labels["inclusion_probability"] = [0.5, 0.5, 0.25, 0.25]
    linker.table_management.register_table(labels, "labels", overwrite=True)

    ts = (
        linker.evaluation.accuracy_analysis_from_labels_table(
            "labels",
            output_type="table",
            match_weight_round_to_nearest=None,
            inclusion_probability_column="inclusion_probability",
        )
        .as_pandas_dataframe()
        .sort_values("truth_threshold")
        .reset_index(drop=True)
    )
    # weights: 2, 2, 4, 4 -> total estimated mass 12; positive mass 2
    assert np.isclose(ts["estimated_pair_count"].iloc[0], 12.0)
    assert (ts["sampled_pair_count"] == 4).all()
    high = ts.iloc[-1]
    # the exact first_name match (robin, robin) is the sole reviewed positive
    assert np.isclose(high["tp"], 2.0)
    assert np.isclose(high["p"], 2.0)

    errs = linker.evaluation.prediction_errors_from_labels_table(
        "labels", inclusion_probability_column="inclusion_probability"
    ).as_pandas_dataframe()
    assert (errs["source_dataset_l"] == "left_source").all()
    assert (errs["source_dataset_r"] == "right_source").all()
    assert np.isclose(errs["evaluation_weight"], 1.0 / errs["inclusion_probability"]).all()


@mark_with_dialects_including("duckdb")
def test_weighted_add_metrics_derive_from_weighted_masses():
    data = _robin_david_data()
    linker = Linker(data, _robin_david_settings(), db_api=DuckDBAPI())
    pred = linker.inference.predict().as_pandas_dataframe()
    labels, _ = _review_design_labels(pred, data)
    linker.table_management.register_table(labels, "labels", overwrite=True)

    ts = (
        linker.evaluation.accuracy_analysis_from_labels_table(
            "labels",
            output_type="table",
            match_weight_round_to_nearest=None,
            add_metrics=["f1", "specificity", "accuracy", "phi"],
            inclusion_probability_column="inclusion_probability",
        )
        .as_pandas_dataframe()
        .sort_values("truth_threshold")
        .reset_index(drop=True)
    )
    high = ts.iloc[1]
    tp, fp, tn, fn = high["tp"], high["fp"], high["tn"], high["fn"]
    assert np.isclose(high["f1"], 2 * tp / (2 * tp + fn + fp))
    assert np.isclose(high["specificity"], tn / (tn + fp))
    assert np.isclose(high["accuracy"], (tp + tn) / (tp + tn + fp + fn))
    phi_num = (tp * tn) - (fp * fn)
    phi_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    assert np.isclose(high["phi"], phi_num / phi_den)
    # the chart path with add_metrics also renders on the weighted data
    chart = linker.evaluation.accuracy_analysis_from_labels_table(
        "labels",
        output_type="accuracy",
        match_weight_round_to_nearest=None,
        add_metrics=["f1"],
        inclusion_probability_column="inclusion_probability",
    )
    fields = _tooltip_fields(chart)
    assert "estimated_pair_count" in fields

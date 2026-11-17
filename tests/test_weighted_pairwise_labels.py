"""Tests for weighted clerical pairwise-label training.

Covers the shared evidence contract used by both the direct m estimator and
the partially-labelled expectation-maximisation path:

* hand-checkable direct m estimation and one EM update;
* same/reversed and repeated submissions, identical vector/different labels;
* 0 / 1 / fractional clerical scores and non-integer weights;
* cross-source id collisions, custom id/source-dataset columns;
* null comparison levels, fixed parameters and zero effective mass;
* no-label regression, term-frequency behaviour and JSON/predict consistency.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
import pytest

import splink.internals.comparison_library as cm
from splink import DuckDBAPI, Linker, SettingsCreator, block_on


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _dedupe_linker(data, **kwargs):
    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=[
            cm.ExactMatch("first_name"),
            cm.LevenshteinAtThresholds("surname", 1),
        ],
        **kwargs,
    )
    return Linker(data, settings, db_api=DuckDBAPI())


def _small_data():
    return pd.DataFrame(
        [
            {"unique_id": 1, "first_name": "A", "surname": "Smith"},
            {"unique_id": 2, "first_name": "A", "surname": "Smith"},
            {"unique_id": 3, "first_name": "A", "surname": "Smyth"},
            {"unique_id": 4, "first_name": "A", "surname": "Jones"},
            {"unique_id": 5, "first_name": "B", "surname": "Brown"},
            {"unique_id": 6, "first_name": "B", "surname": "Browne"},
        ]
    )


def _labels(rows):
    return pd.DataFrame(rows)


def _surname_m(linker):
    surname = linker._settings_obj.comparisons[1]
    return {
        cl.comparison_vector_value: cl.m_probability
        for cl in surname._comparison_levels_excluding_null
    }


# --------------------------------------------------------------------------- #
# Direct, weighted m estimation (hand calculation)
# --------------------------------------------------------------------------- #
def test_direct_weighted_m_hand_calculation():
    df = _small_data()
    # Blocking on first_name: surname gammas are
    #   (1,2) exact (2), (1,3) lev (1), (1,4) else (0)
    labels = _labels(
        [
            # reverse duplicate of a certain match -> merges to weight 2
            {"unique_id_l": 1, "unique_id_r": 2, "clerical_match_score": 1.0},
            {"unique_id_l": 2, "unique_id_r": 1, "clerical_match_score": 1.0},
            # uncertain pair, weight 2
            {"unique_id_l": 1, "unique_id_r": 3, "clerical_match_score": 0.5},
            # confirmed non-match, weight 3 (no positive mass)
            {"unique_id_l": 1, "unique_id_r": 4, "clerical_match_score": 0.0},
        ]
    )
    labels["w"] = [1, 1, 2, 3]
    linker = _dedupe_linker(df)
    linker.table_management.register_table(labels, "lab", overwrite=True)
    summary = linker.training.estimate_m_from_pairwise_labels(
        "lab", use_label_scores=True, label_weight_column="w"
    )

    # weighted positive mass: level2 = 2*1 = 2; level1 = 2*0.5 = 1;
    # level0 = 3*0 = 0  -> total 3
    m = _surname_m(linker)
    assert m[2] == pytest.approx(2 / 3)
    assert m[1] == pytest.approx(1 / 3)
    assert m[0] == pytest.approx(0.0)

    s = summary.as_dict()
    assert s["labelled_pair_count"] == 3  # the reversed duplicate was merged
    assert s["label_weight_sum"] == pytest.approx(7.0)
    assert s["positive_mass_sum"] == pytest.approx(3.0)
    assert s["negative_mass_sum"] == pytest.approx(4.0)


def test_integer_weight_equals_repeated_evidence():
    df = pd.DataFrame(
        [
            {"unique_id": 1, "first_name": "A", "surname": "Smith"},
            {"unique_id": 2, "first_name": "A", "surname": "Smyth"},
            {"unique_id": 3, "first_name": "A", "surname": "Jones"},
        ]
    )
    settings = SettingsCreator(
        link_type="dedupe_only", comparisons=[cm.ExactMatch("surname")]
    )
    weighted = pd.DataFrame(
        [
            {"unique_id_l": 1, "unique_id_r": 2, "clerical_match_score": 1.0, "w": 3},
            {"unique_id_l": 1, "unique_id_r": 3, "clerical_match_score": 1.0, "w": 1},
        ]
    )
    repeated = pd.DataFrame(
        [{"unique_id_l": 1, "unique_id_r": 2, "clerical_match_score": 1.0}] * 3
        + [{"unique_id_l": 1, "unique_id_r": 3, "clerical_match_score": 1.0}]
    )
    lw = Linker(df, settings, db_api=DuckDBAPI())
    lw.table_management.register_table(weighted, "w", overwrite=True)
    lw.training.estimate_m_from_pairwise_labels(
        "w", use_label_scores=True, label_weight_column="w"
    )
    lr = Linker(df, settings, db_api=DuckDBAPI())
    lr.table_management.register_table(repeated, "r", overwrite=True)
    lr.training.estimate_m_from_pairwise_labels("r", use_label_scores=True)
    for cc_w, cc_r in zip(
        lw._settings_obj.comparisons, lr._settings_obj.comparisons
    ):
        for cl_w, cl_r in zip(
            cc_w._comparison_levels_excluding_null,
            cc_r._comparison_levels_excluding_null,
        ):
            assert cl_w.m_probability == pytest.approx(cl_r.m_probability)


def test_null_level_excluded_from_normalisation():
    df = pd.DataFrame(
        [
            {"unique_id": 1, "first_name": "A", "surname": "Smith"},
            {"unique_id": 2, "first_name": "A", "surname": None},
            {"unique_id": 3, "first_name": "A", "surname": "Jones"},
        ]
    )
    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=[cm.ExactMatch("first_name"), cm.ExactMatch("surname")],
    )
    labels = _labels(
        [
            {"unique_id_l": 1, "unique_id_r": 2, "clerical_match_score": 1.0},
            {"unique_id_l": 1, "unique_id_r": 3, "clerical_match_score": 1.0},
        ]
    )
    linker = Linker(df, settings, db_api=DuckDBAPI())
    linker.table_management.register_table(labels, "lab", overwrite=True)
    linker.training.estimate_m_from_pairwise_labels(
        "lab", use_label_scores=True
    )
    surname = linker._settings_obj.comparisons[1]
    m = {
        cl.comparison_vector_value: cl.m_probability
        for cl in surname._comparison_levels_excluding_null
    }
    # only the non-match level has positive mass among NON-null levels
    assert m[0] == pytest.approx(1.0)


def test_zero_positive_mass_raises():
    df = _small_data()
    labels = _labels(
        [{"unique_id_l": 1, "unique_id_r": 2, "clerical_match_score": 0.0}]
    )
    linker = _dedupe_linker(df)
    linker.table_management.register_table(labels, "lab", overwrite=True)
    with pytest.raises(ValueError, match="positive"):
        linker.training.estimate_m_from_pairwise_labels(
            "lab", use_label_scores=True
        )


def test_legacy_mode_ignores_scores():
    data = pd.DataFrame(
        [
            {"unique_id": 1, "name": "Robin", "cluster": 1},
            {"unique_id": 2, "name": "Robyn", "cluster": 1},
            {"unique_id": 3, "name": "Robin", "cluster": 1},
            {"unique_id": 4, "name": "James", "cluster": 2},
            {"unique_id": 5, "name": "David", "cluster": 2},
        ]
    )
    settings = {
        "link_type": "dedupe_only",
        "comparisons": [cm.LevenshteinAtThresholds("name", 2)],
    }
    df_l = data[["unique_id", "cluster"]]
    df_labels = df_l.merge(
        df_l, on="cluster", suffixes=("_l", "_r")
    )
    df_labels = df_labels[df_labels["unique_id_l"] < df_labels["unique_id_r"]]
    df_labels["clerical_match_score"] = 0.0  # must be ignored
    linker = Linker(data, settings, db_api=DuckDBAPI())
    linker.table_management.register_table(df_labels, "labels", overwrite=True)
    linker.training.estimate_m_from_pairwise_labels("labels")
    cc = linker._settings_obj.comparisons[0]
    assert (
        cc._get_comparison_level_by_comparison_vector_value(2).m_probability
        == 1 / 4
    )
    assert (
        cc._get_comparison_level_by_comparison_vector_value(1).m_probability
        == 2 / 4
    )


# --------------------------------------------------------------------------- #
# Validation contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad_score", [-0.1, 1.1, float("nan"), float("inf")]
)
def test_illegal_scores_raise(bad_score):
    linker = _dedupe_linker(_small_data())
    labels = _labels(
        [{"unique_id_l": 1, "unique_id_r": 2, "clerical_match_score": bad_score}]
    )
    linker.table_management.register_table(labels, "lab", overwrite=True)
    with pytest.raises(ValueError, match="clerical_match_score"):
        linker.training.estimate_m_from_pairwise_labels(
            "lab", use_label_scores=True
        )


@pytest.mark.parametrize(
    "bad_weight", [0.0, -1.0, float("nan"), float("inf")]
)
def test_illegal_weights_raise(bad_weight):
    linker = _dedupe_linker(_small_data())
    labels = _labels(
        [{"unique_id_l": 1, "unique_id_r": 2, "clerical_match_score": 1.0}]
    )
    labels["w"] = [bad_weight]
    linker.table_management.register_table(labels, "lab", overwrite=True)
    with pytest.raises(ValueError, match="weight"):
        linker.training.estimate_m_from_pairwise_labels(
            "lab", use_label_scores=True, label_weight_column="w"
        )


def test_weight_column_requires_score_mode():
    linker = _dedupe_linker(_small_data())
    labels = _labels(
        [{"unique_id_l": 1, "unique_id_r": 2, "clerical_match_score": 1.0}]
    )
    labels["w"] = [1.0]
    linker.table_management.register_table(labels, "lab", overwrite=True)
    with pytest.raises(ValueError, match="use_label_scores"):
        linker.training.estimate_m_from_pairwise_labels(
            "lab", label_weight_column="w"
        )
    with pytest.raises(ValueError, match="label_weight_column"):
        linker.training.estimate_parameters_using_expectation_maximisation(
            block_on("first_name"), label_weight_column="w"
        )


def test_missing_columns_self_pair_unknown_record_conflict():
    linker = _dedupe_linker(_small_data())

    with pytest.raises(ValueError, match="missing required column"):
        linker.table_management.register_table(
            pd.DataFrame([{"unique_id_l": 1, "unique_id_r": 2}]),
            "no_score",
            overwrite=True,
        )
        linker.training.estimate_m_from_pairwise_labels(
            "no_score", use_label_scores=True
        )

    self_pair = _labels(
        [{"unique_id_l": 1, "unique_id_r": 1, "clerical_match_score": 1.0}]
    )
    linker.table_management.register_table(self_pair, "selfp", overwrite=True)
    with pytest.raises(ValueError, match="Self-pairs"):
        linker.training.estimate_m_from_pairwise_labels(
            "selfp", use_label_scores=True
        )

    unknown = _labels(
        [{"unique_id_l": 1, "unique_id_r": 99, "clerical_match_score": 1.0}]
    )
    linker.table_management.register_table(unknown, "unk", overwrite=True)
    with pytest.raises(ValueError, match="do not exist"):
        linker.training.estimate_m_from_pairwise_labels(
            "unk", use_label_scores=True
        )

    conflict = _labels(
        [
            {"unique_id_l": 1, "unique_id_r": 2, "clerical_match_score": 1.0},
            {"unique_id_l": 2, "unique_id_r": 1, "clerical_match_score": 0.2},
        ]
    )
    linker.table_management.register_table(conflict, "conf", overwrite=True)
    with pytest.raises(ValueError, match="Conflicting"):
        linker.training.estimate_m_from_pairwise_labels(
            "conf", use_label_scores=True
        )


def test_failure_leaves_model_untouched():
    linker = _dedupe_linker(_small_data())
    linker.training.estimate_u_using_random_sampling(max_pairs=1e3)
    before = linker.misc.save_model_to_json()
    bad = _labels(
        [{"unique_id_l": 1, "unique_id_r": 99, "clerical_match_score": 1.0}]
    )
    linker.table_management.register_table(bad, "lab", overwrite=True)
    with pytest.raises(ValueError):
        linker.training.estimate_m_from_pairwise_labels(
            "lab", use_label_scores=True
        )
    assert linker.misc.save_model_to_json() == before
    n_sessions = len(linker._em_training_sessions)
    with pytest.raises(ValueError):
        linker.training.estimate_parameters_using_expectation_maximisation(
            block_on("first_name"), pairwise_labels="lab"
        )
    assert len(linker._em_training_sessions) == n_sessions


def test_link_only_cross_source_and_custom_id():
    data_a = pd.DataFrame(
        [
            {"unique_id": 1, "first_name": "A", "surname": "X"},
            {"unique_id": 2, "first_name": "A", "surname": "Y"},
        ]
    )
    data_b = pd.DataFrame(
        [
            {"unique_id": 1, "first_name": "A", "surname": "X"},
            {"unique_id": 2, "first_name": "A", "surname": "Z"},
        ]
    )
    settings = SettingsCreator(
        link_type="link_only",
        comparisons=[cm.ExactMatch("first_name"), cm.ExactMatch("surname")],
    )
    linker = Linker(
        [data_a, data_b],
        settings,
        db_api=DuckDBAPI(),
        input_table_aliases=["da", "db"],
    )
    # equal numeric ids in different datasets are distinct records
    labels = pd.DataFrame(
        [
            {
                "source_dataset_l": "da",
                "unique_id_l": 1,
                "source_dataset_r": "db",
                "unique_id_r": 1,
                "clerical_match_score": 1.0,
            }
        ]
    )
    linker.table_management.register_table(labels, "lab", overwrite=True)
    summary = linker.training.estimate_m_from_pairwise_labels(
        "lab", use_label_scores=True
    )
    assert summary.labelled_pair_count == 1

    same_source = pd.DataFrame(
        [
            {
                "source_dataset_l": "da",
                "unique_id_l": 1,
                "source_dataset_r": "da",
                "unique_id_r": 2,
                "clerical_match_score": 1.0,
            }
        ]
    )
    linker.table_management.register_table(
        same_source, "bad", overwrite=True
    )
    with pytest.raises(ValueError, match="different source datasets"):
        linker.training.estimate_m_from_pairwise_labels(
            "bad", use_label_scores=True
        )

    # custom unique id column name
    data = pd.DataFrame(
        [
            {"record_id": 1, "first_name": "A", "surname": "X"},
            {"record_id": 2, "first_name": "A", "surname": "Y"},
        ]
    )
    custom = SettingsCreator(
        link_type="dedupe_only",
        unique_id_column_name="record_id",
        comparisons=[cm.ExactMatch("first_name"), cm.ExactMatch("surname")],
    )
    linker2 = Linker(data, custom, db_api=DuckDBAPI())
    linker2.table_management.register_table(
        pd.DataFrame(
            [{"record_id_l": 1, "record_id_r": 2, "clerical_match_score": 1.0}]
        ),
        "lab2",
        overwrite=True,
    )
    summary2 = linker2.training.estimate_m_from_pairwise_labels(
        "lab2", use_label_scores=True
    )
    assert summary2.labelled_pair_count == 1


# --------------------------------------------------------------------------- #
# Labelled EM
# --------------------------------------------------------------------------- #
def _em_settings(max_iterations=3):
    return SettingsCreator(
        link_type="dedupe_only",
        comparisons=[
            cm.ExactMatch("first_name"),
            cm.LevenshteinAtThresholds("surname", 1),
        ],
        blocking_rules_to_generate_predictions=[block_on("first_name")],
        max_iterations=max_iterations,
    )


def test_labelled_em_summary_and_split():
    df = _small_data()
    labels = _labels(
        [
            {"unique_id_l": 1, "unique_id_r": 2, "clerical_match_score": 1.0},
            {"unique_id_l": 1, "unique_id_r": 3, "clerical_match_score": 0.5},
            {"unique_id_l": 1, "unique_id_r": 4, "clerical_match_score": 0.0},
            {"unique_id_l": 5, "unique_id_r": 6, "clerical_match_score": 0.8},
        ]
    )
    for tf in (True, False):
        linker = Linker(df, _em_settings(), db_api=DuckDBAPI())
        linker.table_management.register_table(labels, "lab", overwrite=True)
        session = linker.training.estimate_parameters_using_expectation_maximisation(
            block_on("first_name"),
            estimate_without_term_frequencies=tf,
            fix_u_probabilities=False,
            pairwise_labels="lab",
        )
        s = session.label_training_summary.as_dict()
        assert s["labelled_pair_count"] == 4
        assert s["unlabelled_pair_count"] == 3
        assert s["label_weight_sum"] == pytest.approx(4.0)
        assert s["positive_mass_sum"] == pytest.approx(2.3)
        assert s["negative_mass_sum"] == pytest.approx(1.7)
        assert s["stop_reason"] in ("converged", "max_iterations_reached")
        # no NaNs leak into trained parameters
        for cc in linker._settings_obj.comparisons:
            for cl in cc._comparison_levels_excluding_null:
                assert not math.isnan(cl.m_probability)
                assert not math.isnan(cl.u_probability)


def test_labelled_em_first_update_matches_direct_m():
    """All-pairs-labelled first EM update must reproduce the direct weighted m
    estimate on trainable comparisons, within tight numerical tolerances."""
    df = _small_data()
    all_labels = _labels(
        [
            {"unique_id_l": 1, "unique_id_r": 2, "clerical_match_score": 1.0},
            {"unique_id_l": 1, "unique_id_r": 3, "clerical_match_score": 0.5},
            {"unique_id_l": 1, "unique_id_r": 4, "clerical_match_score": 0.0},
            {"unique_id_l": 2, "unique_id_r": 3, "clerical_match_score": 0.5},
            {"unique_id_l": 2, "unique_id_r": 4, "clerical_match_score": 0.0},
            {"unique_id_l": 3, "unique_id_r": 4, "clerical_match_score": 0.0},
            {"unique_id_l": 5, "unique_id_r": 6, "clerical_match_score": 0.8},
        ]
    )
    direct_linker = Linker(df, _em_settings(), db_api=DuckDBAPI())
    direct_linker.table_management.register_table(
        all_labels, "all_labels", overwrite=True
    )
    direct_linker.training.estimate_m_from_pairwise_labels(
        "all_labels", use_label_scores=True
    )
    direct_m = {}
    for cc in direct_linker._settings_obj.comparisons:
        for cl in cc._comparison_levels_excluding_null:
            direct_m[(cc.output_column_name, cl.comparison_vector_value)] = (
                cl.m_probability
            )

    em_settings = _em_settings(max_iterations=1)
    em_linker = Linker(df, em_settings, db_api=DuckDBAPI())
    em_linker.table_management.register_table(
        all_labels, "all_labels", overwrite=True
    )
    session = em_linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("first_name"),
        estimate_without_term_frequencies=False,
        fix_u_probabilities=True,
        fix_probability_two_random_records_match=True,
        pairwise_labels="all_labels",
    )
    trainable = {
        cc.output_column_name
        for cc in session.core_model_settings.comparisons
    }
    for cc in em_linker._settings_obj.comparisons:
        if cc.output_column_name not in trainable:
            continue
        for cl in cc._comparison_levels_excluding_null:
            assert np.isclose(
                cl.m_probability,
                direct_m[(cc.output_column_name, cl.comparison_vector_value)],
                rtol=1e-9,
                atol=1e-12,
            )


def test_label_outside_block_raises():
    df = _small_data()
    labels = _labels(
        [{"unique_id_l": 1, "unique_id_r": 5, "clerical_match_score": 1.0}]
    )
    linker = Linker(df, _em_settings(), db_api=DuckDBAPI())
    linker.table_management.register_table(labels, "lab", overwrite=True)
    with pytest.raises(ValueError, match="not generated by the training blocking"):
        linker.training.estimate_parameters_using_expectation_maximisation(
            block_on("first_name"), pairwise_labels="lab"
        )


def test_empty_label_table_matches_unlabelled_em():
    data = pd.read_csv("./tests/datasets/fake_1000_from_splink_demos.csv")
    settings = dict(
        link_type="dedupe_only",
        comparisons=[
            cm.ExactMatch("first_name"),
            cm.ExactMatch("surname"),
            cm.ExactMatch("email"),
        ],
    )
    empty = pd.DataFrame(
        {
            "unique_id_l": pd.Series(dtype="int64"),
            "unique_id_r": pd.Series(dtype="int64"),
            "clerical_match_score": pd.Series(dtype="float64"),
        }
    )
    plain = Linker(data, settings, db_api=DuckDBAPI())
    labelled = Linker(data, settings, db_api=DuckDBAPI())
    labelled.table_management.register_table(empty, "empty", overwrite=True)
    s0 = plain.training.estimate_parameters_using_expectation_maximisation(
        "l.email = r.email"
    )
    s1 = labelled.training.estimate_parameters_using_expectation_maximisation(
        "l.email = r.email", pairwise_labels="empty"
    )
    assert s1.label_training_summary is None
    pd.testing.assert_frame_equal(
        pd.DataFrame(s0._lambda_history_records),
        pd.DataFrame(s1._lambda_history_records),
    )


def test_zero_mass_retains_previous_parameters():
    # every blocked pair is a certain match -> no negative mass -> u retained
    data = pd.DataFrame(
        [
            {"unique_id": 1, "first_name": "A", "surname": "X"},
            {"unique_id": 2, "first_name": "A", "surname": "X"},
            {"unique_id": 3, "first_name": "A", "surname": "Y"},
        ]
    )
    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=[cm.ExactMatch("surname")],
        max_iterations=2,
    )
    labels = _labels(
        [
            {"unique_id_l": 1, "unique_id_r": 2, "clerical_match_score": 1.0},
            {"unique_id_l": 1, "unique_id_r": 3, "clerical_match_score": 1.0},
            {"unique_id_l": 2, "unique_id_r": 3, "clerical_match_score": 1.0},
        ]
    )
    linker = Linker(data, settings, db_api=DuckDBAPI())
    surname = linker._settings_obj.comparisons[0]
    prev_u = {
        cl.comparison_vector_value: cl.u_probability
        for cl in surname._comparison_levels_excluding_null
    }
    linker.table_management.register_table(labels, "lab", overwrite=True)
    session = linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("first_name"),
        estimate_without_term_frequencies=True,
        fix_u_probabilities=False,
        pairwise_labels="lab",
    )
    ne = {
        (d["output_column_name"], d["parameter"])
        for d in session.label_training_summary.not_estimated
    }
    assert ("surname", "u") in ne
    for cl in surname._comparison_levels_excluding_null:
        assert not math.isnan(cl.u_probability)
        assert cl.u_probability == prev_u[cl.comparison_vector_value]


def test_same_vector_different_scores_not_merged_in_compressed_path():
    data = pd.DataFrame(
        [
            {"unique_id": 1, "first_name": "A", "surname": "X"},
            {"unique_id": 2, "first_name": "A", "surname": "X"},
            {"unique_id": 3, "first_name": "A", "surname": "X"},
            {"unique_id": 4, "first_name": "A", "surname": "X"},
        ]
    )
    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=[cm.ExactMatch("surname")],
        max_iterations=1,
    )
    labels = _labels(
        [
            {"unique_id_l": 1, "unique_id_r": 2, "clerical_match_score": 1.0},
            {"unique_id_l": 1, "unique_id_r": 3, "clerical_match_score": 0.0},
            {"unique_id_l": 1, "unique_id_r": 4, "clerical_match_score": 0.5},
        ]
    )
    for tf in (True, False):
        linker = Linker(data, settings, db_api=DuckDBAPI())
        linker.table_management.register_table(labels, "lab", overwrite=True)
        session = linker.training.estimate_parameters_using_expectation_maximisation(
            block_on("first_name"),
            estimate_without_term_frequencies=tf,
            fix_u_probabilities=False,
            pairwise_labels="lab",
        )
        assert session.label_training_summary.positive_mass_sum == pytest.approx(1.5)


def test_no_label_em_regression_and_tf_equivalence_of_evidence():
    data = pd.read_csv("./tests/datasets/fake_1000_from_splink_demos.csv")
    settings = dict(
        link_type="dedupe_only",
        comparisons=[
            cm.ExactMatch("first_name"),
            cm.ExactMatch("surname"),
            cm.ExactMatch("email"),
        ],
        max_iterations=3,
    )
    # no labels -> labelled flag stays off and results are reproducible
    for tf in (True, False):
        a = Linker(data, settings, db_api=DuckDBAPI())
        b = Linker(data, settings, db_api=DuckDBAPI())
        sa = a.training.estimate_parameters_using_expectation_maximisation(
            "l.email = r.email", estimate_without_term_frequencies=tf
        )
        sb = b.training.estimate_parameters_using_expectation_maximisation(
            "l.email = r.email", estimate_without_term_frequencies=tf
        )
        assert sb.label_training_summary is None
        pd.testing.assert_frame_equal(
            pd.DataFrame(sa._iteration_history_records),
            pd.DataFrame(sb._iteration_history_records),
        )


def test_json_roundtrip_and_predict_omit_labels(tmp_path):
    data = pd.read_csv("./tests/datasets/fake_1000_from_splink_demos.csv")
    pairs = []
    for _, a in data.head(50).iterrows():
        for _, b in data.head(50).iterrows():
            if a["unique_id"] < b["unique_id"] and a["first_name"] == b["first_name"]:
                pairs.append(
                    {
                        "unique_id_l": a["unique_id"],
                        "unique_id_r": b["unique_id"],
                        "clerical_match_score": (
                            1.0 if a["surname"] == b["surname"] else 0.3
                        ),
                    }
                )
    labels = pd.DataFrame(pairs[:60])
    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=[
            cm.ExactMatch("first_name"),
            cm.ExactMatch("surname"),
            cm.ExactMatch("email"),
        ],
        blocking_rules_to_generate_predictions=[block_on("first_name")],
        max_iterations=3,
    )
    linker = Linker(data, settings, db_api=DuckDBAPI())
    linker.table_management.register_table(labels, "lab", overwrite=True)
    linker.training.estimate_u_using_random_sampling(max_pairs=1e4)
    linker.training.estimate_parameters_using_expectation_maximisation(
        "l.first_name = r.first_name",
        fix_u_probabilities=False,
        pairwise_labels="lab",
    )
    linker.training.estimate_m_from_pairwise_labels(
        "lab", use_label_scores=True
    )
    pred_before = (
        linker.inference.predict()
        .as_pandas_dataframe()
        .sort_values(["unique_id_l", "unique_id_r"])
        .reset_index(drop=True)
    )
    assert "clerical_match_score" not in pred_before.columns

    model_json = linker.misc.save_model_to_json()
    serialized = json.dumps(model_json)
    for forbidden in (
        "clerical_match_score",
        "__splink_label_weight",
        "label_weight",
    ):
        assert forbidden not in serialized

    path = os.path.join(tmp_path, "model.json")
    linker.misc.save_model_to_json(path, overwrite=True)
    reloaded = Linker(data, path, db_api=DuckDBAPI())
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

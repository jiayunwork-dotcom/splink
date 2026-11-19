"""Tests for stratified sampling of pairwise predictions.

All expected values are hand-derived from the original record identities and
the declared sampling design (stratum counts, largest-remainder allocation,
inclusion probability n_h/N_h).  The tests never use the sampling or
aggregation functions under test to generate their own expectations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from splink import DuckDBAPI, Linker, SettingsCreator
from splink.internals.comparison_library import ExactMatch
from splink.internals.pairwise_sampling import (
    INCLUSION_PROBABILITY_COLUMN,
    SAMPLE_STRATUM_COLUMN,
    STRATUM_POPULATION_COLUMN,
    STRATUM_SAMPLE_SIZE_COLUMN,
    largest_remainder_allocation,
    sample_pairwise_predictions,
)

from .decorator import mark_with_dialects_including


def _linker_and_predictions(
    n_records=20, *, link_only=False, retain=True, seed=11
):
    rng = np.random.default_rng(seed)
    if link_only:
        df_l = pd.DataFrame(
            {
                "unique_id": range(n_records),
                "first_name": rng.choice(list("abcde"), n_records),
                "source_dataset": "left_source",
            }
        )
        df_r = pd.DataFrame(
            {
                "unique_id": range(n_records),
                "first_name": rng.choice(list("abcde"), n_records),
                "source_dataset": "right_source",
            }
        )
        settings = SettingsCreator(
            link_type="link_only",
            comparisons=[ExactMatch("first_name")],
            blocking_rules_to_generate_predictions=["1=1"],
            retain_matching_columns=retain,
        )
        linker = Linker(
            [df_l, df_r],
            settings,
            db_api=DuckDBAPI(),
            input_table_aliases=["left_source", "right_source"],
        )
    else:
        df = pd.DataFrame(
            {
                "unique_id": range(n_records),
                "first_name": rng.choice(list("abcde"), n_records),
            }
        )
        settings = SettingsCreator(
            link_type="dedupe_only",
            comparisons=[ExactMatch("first_name")],
            blocking_rules_to_generate_predictions=["1=1"],
            retain_matching_columns=retain,
        )
        linker = Linker(df, settings, db_api=DuckDBAPI())
    return linker, linker.inference.predict()


def _canonical_pairs(df_sample):
    """Unoriented (l, r) identity tuples with l < r."""
    pairs = set()
    for row in df_sample.to_dict("records"):
        l = (row.get("source_dataset_l"), row["unique_id_l"])
        r = (row.get("source_dataset_r"), row["unique_id_r"])
        pairs.add(tuple(sorted((l, r))))
    return pairs


# --------------------------------------------------------------------------- #
# Argument validation
# --------------------------------------------------------------------------- #
@mark_with_dialects_including("duckdb")
def test_invalid_score_bins_rejected():
    linker, pred = _linker_and_predictions()
    bad_bins = [
        [0, 1],  # not wrong itself; keep a valid baseline below
    ]
    # valid baseline works
    linker.evaluation.sample_pairwise_predictions(
        pred, budget=2, score_bins=[0.0, 1.0]
    )

    invalid = [
        [0.1, 1.0],  # does not start at 0
        [0.0, 0.9],  # does not end at 1
        [0.0, 0.5, 0.5, 1.0],  # not strictly increasing
        [0.0, float("nan"), 1.0],
        [0.0, float("inf"), 1.0],
        [True, 0.5, 1.0],  # bool is not a real number
        [0, 0.5, False],
        [],
        [0, 1, 2],
    ]
    for bins in invalid:
        with pytest.raises(ValueError):
            linker.evaluation.sample_pairwise_predictions(
                pred, budget=2, score_bins=bins
            )


@mark_with_dialects_including("duckdb")
def test_invalid_budget_and_seed_rejected():
    linker, pred = _linker_and_predictions()
    for budget in [0, -1, True, 1.5, 2.0, "3"]:
        with pytest.raises(ValueError):
            linker.evaluation.sample_pairwise_predictions(
                pred, budget=budget, score_bins=[0.0, 1.0]
            )
    for seed in [-1, 2**32, True, 1.5, "0"]:
        with pytest.raises(ValueError):
            linker.evaluation.sample_pairwise_predictions(
                pred, budget=2, score_bins=[0.0, 1.0], seed=seed
            )


# --------------------------------------------------------------------------- #
# Largest-remainder allocation (hand enumerated)
# --------------------------------------------------------------------------- #
def test_largest_remainder_allocation_hand_calculation():
    # populations 5, 3, 2 across 3 strata; budget 6
    # floor 1 each -> remaining 3; capacity 4,2,1, total 7
    # raw shares of 3: 12/7=1.714, 6/7=0.857, 3/7=0.429
    # floors 1,0,0 (handed 1), leftover 2 -> remainders .714, .857, .429
    # extras to stratum 1 then stratum 0 -> allocations 3,2,1
    alloc = largest_remainder_allocation([5, 3, 2], 6)
    assert alloc == [3, 2, 1]
    assert sum(alloc) == 6
    assert all(a <= n for a, n in zip(alloc, [5, 3, 2]))
    assert all(a >= 1 for a in alloc)

    # budget >= population: everything taken
    assert largest_remainder_allocation([5, 3, 2], 10) == [5, 3, 2]
    assert largest_remainder_allocation([5, 3, 2], 100) == [5, 3, 2]

    # every non-empty stratum gets at least one
    alloc = largest_remainder_allocation([1, 1, 1, 100], 4)
    assert alloc == [1, 1, 1, 1]

    # saturated small strata redistribute: strata of size 1 each cannot take
    # more than one; a larger stratum absorbs the rest.
    alloc = largest_remainder_allocation([1, 1, 8], 5)
    assert alloc[0] == 1 and alloc[1] == 1 and alloc[2] == 3
    assert sum(alloc) == 5


def test_budget_below_non_empty_strata_rejected():
    # Four strata, but make sure all are non-empty by controlling scores.
    linker, pred = _linker_and_predictions(n_records=40)
    df = pred.as_pandas_dataframe()
    # Force distinct probabilities into four bins by editing the table.
    edges = [0.0, 0.25, 0.5, 0.75, 1.0]
    probs = []
    for i in range(len(df)):
        probs.append(edges[i % 4] + 0.01)
    df["match_probability"] = probs
    name = "forced_scores"
    linker.table_management.register_table(df, name, overwrite=True)
    with pytest.raises(ValueError, match="non-empty strata"):
        linker.evaluation.sample_pairwise_predictions(
            name, budget=3, score_bins=edges
        )


# --------------------------------------------------------------------------- #
# Sampling invariants on a hand-built population
# --------------------------------------------------------------------------- #
def _hand_population_linker():
    # 6 records -> 15 dedupe pairs; assign known probabilities deterministically
    n = 6
    df = pd.DataFrame(
        {"unique_id": range(n), "first_name": list("abcdef")}
    )
    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=[ExactMatch("first_name")],
        blocking_rules_to_generate_predictions=["1=1"],
        retain_matching_columns=True,
    )
    linker = Linker(df, settings, db_api=DuckDBAPI())
    pred = linker.inference.predict().as_pandas_dataframe()
    # bins [0, .33, .66, 1]: place 8 pairs in stratum 0, 4 in stratum 1,
    # 3 in stratum 2 (total 15), with one pair exactly on a bin edge and one
    # pair exactly at 1.
    scores = (
        [0.05] * 7
        + [0.33]  # exactly on edge -> belongs to stratum 1 [0.33, 0.66)
        + [0.40] * 3
        + [0.66]  # exactly on edge -> belongs to stratum 2 [0.66, 1]
        + [0.9] * 2
        + [1.0]  # final stratum contains 1
    )
    assert len(scores) == len(pred) == 15
    pred = pred.sort_values(["unique_id_l", "unique_id_r"]).reset_index(drop=True)
    pred["match_probability"] = scores
    # keep match_weight consistent-ish (not used by sampling)
    linker.table_management.register_table(pred, "hand_pop", overwrite=True)
    return linker, "hand_pop", scores


def test_hand_population_strata_sizes_and_metadata():
    linker, table, scores = _hand_population_linker()
    # expected stratum membership from the score list alone:
    # stratum 0: seven 0.05 scores; stratum 1: the 0.33 edge pair plus three
    # 0.40 scores (4); stratum 2: the 0.66 edge pair, two 0.9 and the 1.0
    # pair (4).
    expected_pop = {0: 7, 1: 4, 2: 4}
    sample = linker.evaluation.sample_pairwise_predictions(
        table, budget=9, score_bins=[0.0, 0.33, 0.66, 1.0], seed=5
    )
    out = sample.as_pandas_dataframe()

    # sample size and per-stratum sizes match an allocation computed by hand:
    # floor 1 per stratum, 6 remaining over capacity (6, 3, 3): proportional
    # shares 36/12=3.0, 18/12=1.5, 18/12=1.5 -> floors 3,1,1 (5 handed),
    # one leftover to one of the tied .5 strata (lowest index wins) ->
    # allocations (4, 3, 2)
    assert len(out) == 9
    got_sizes = out.groupby(SAMPLE_STRATUM_COLUMN).size().to_dict()
    alloc = largest_remainder_allocation([7, 4, 4], 9)
    assert alloc == [4, 3, 2]
    for h, n_h in enumerate(alloc):
        assert got_sizes[h] == n_h
        rows = out[out[SAMPLE_STRATUM_COLUMN] == h]
        # inclusion probability equals n_h / N_h for every row
        assert np.allclose(
            rows[INCLUSION_PROBABILITY_COLUMN],
            n_h / expected_pop[h],
        )
        assert (rows[STRATUM_POPULATION_COLUMN] == expected_pop[h]).all()
        assert (rows[STRATUM_SAMPLE_SIZE_COLUMN] == n_h).all()

    # no duplicate pairs in the sample
    pairs = _canonical_pairs(out)
    assert len(pairs) == len(out)

    # mass conservation: sum of 1/pi over the sample estimates the population
    weights = 1.0 / out[INCLUSION_PROBABILITY_COLUMN]
    assert np.isclose(weights.sum(), 15.0)


def test_full_census_degenerates_to_probability_one():
    linker, table, _ = _hand_population_linker()
    sample = linker.evaluation.sample_pairwise_predictions(
        table, budget=1000, score_bins=[0.0, 0.33, 0.66, 1.0], seed=5
    )
    out = sample.as_pandas_dataframe()
    assert len(out) == 15
    assert (out[INCLUSION_PROBABILITY_COLUMN] == 1.0).all()
    assert (out[STRATUM_SAMPLE_SIZE_COLUMN] == out[STRATUM_POPULATION_COLUMN]).all()


def test_reproducibility_seed_row_order_reversal_reregistration():
    linker, table, _ = _hand_population_linker()
    bins = [0.0, 0.33, 0.66, 1.0]

    def draw(seed):
        return linker.evaluation.sample_pairwise_predictions(
            table, budget=9, score_bins=bins, seed=seed
        ).as_pandas_dataframe()

    base = draw(7)
    again = draw(7)
    pd.testing.assert_frame_equal(
        base.sort_values(["unique_id_l", "unique_id_r"]).reset_index(drop=True),
        again.sort_values(["unique_id_l", "unique_id_r"]).reset_index(drop=True),
    )

    # different seed -> a valid (generally different) sample, never a fixed
    # head-of-table selection
    other = draw(8)
    assert not _canonical_pairs(base) == _canonical_pairs(other)
    assert len(_canonical_pairs(other)) == 9

    # shuffle input rows
    raw = linker.misc.query_sql(f"select * from {table}")
    shuffled = raw.sample(frac=1.0, random_state=0).reset_index(drop=True)
    linker.table_management.register_table(shuffled, "shuffled", overwrite=True)
    shuffled_sample = linker.evaluation.sample_pairwise_predictions(
        "shuffled", budget=9, score_bins=bins, seed=7
    ).as_pandas_dataframe()
    assert _canonical_pairs(shuffled_sample) == _canonical_pairs(base)

    # reverse every pair's endpoints, swapping every _l/_r column pair
    rev = raw.copy()
    for c in [c for c in raw.columns if c.endswith("_l")]:
        base_name = c[:-2]
        r_col = base_name + "_r"
        if r_col in rev.columns:
            rev[c], rev[r_col] = raw[r_col].copy(), raw[c].copy()
    linker.table_management.register_table(rev, "reversed", overwrite=True)
    rev_sample = linker.evaluation.sample_pairwise_predictions(
        "reversed", budget=9, score_bins=bins, seed=7
    ).as_pandas_dataframe()
    assert _canonical_pairs(rev_sample) == _canonical_pairs(base)

    # re-register under a different physical name
    linker.table_management.register_table(raw, "brand_new_name", overwrite=True)
    renamed_sample = linker.evaluation.sample_pairwise_predictions(
        "brand_new_name", budget=9, score_bins=bins, seed=7
    ).as_pandas_dataframe()
    assert _canonical_pairs(renamed_sample) == _canonical_pairs(base)
    # metadata identical for the same pairs
    meta_cols = [
        SAMPLE_STRATUM_COLUMN,
        STRATUM_POPULATION_COLUMN,
        STRATUM_SAMPLE_SIZE_COLUMN,
        INCLUSION_PROBABILITY_COLUMN,
    ]
    b = (
        base.sort_values(["unique_id_l", "unique_id_r"])[meta_cols]
        .reset_index(drop=True)
    )
    r = (
        renamed_sample.sort_values(["unique_id_l", "unique_id_r"])[meta_cols]
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(b, r)


def test_empty_population_returns_empty_typed_result():
    linker, pred = _linker_and_predictions()
    empty = pred.as_pandas_dataframe().head(0)
    linker.table_management.register_table(empty, "empty_pred", overwrite=True)
    out = linker.evaluation.sample_pairwise_predictions(
        "empty_pred", budget=5, score_bins=[0.0, 0.5, 1.0]
    ).as_pandas_dataframe()
    assert len(out) == 0
    for col in [
        SAMPLE_STRATUM_COLUMN,
        STRATUM_POPULATION_COLUMN,
        STRATUM_SAMPLE_SIZE_COLUMN,
        INCLUSION_PROBABILITY_COLUMN,
        "match_probability",
        "match_weight",
        "unique_id_l",
        "unique_id_r",
    ]:
        assert col in out.columns


def test_invalid_predictions_content_rejected():
    linker, pred = _linker_and_predictions()
    df = pred.as_pandas_dataframe()

    # non-finite / out-of-range scores
    for bad in [float("nan"), float("inf"), -0.1, 1.1]:
        d = df.copy()
        d.loc[0, "match_probability"] = bad
        linker.table_management.register_table(d, "bad_score", overwrite=True)
        with pytest.raises(ValueError):
            linker.evaluation.sample_pairwise_predictions(
                "bad_score", budget=2, score_bins=[0.0, 0.5, 1.0]
            )

    # self pair
    d = df.copy()
    d.loc[0, "unique_id_r"] = d.loc[0, "unique_id_l"]
    linker.table_management.register_table(d, "self_pair", overwrite=True)
    with pytest.raises(ValueError, match="Self-pairs"):
        linker.evaluation.sample_pairwise_predictions(
            "self_pair", budget=2, score_bins=[0.0, 0.5, 1.0]
        )

    # unknown node
    d = df.copy()
    d.loc[0, "unique_id_r"] = 999_999
    linker.table_management.register_table(d, "unknown", overwrite=True)
    with pytest.raises(ValueError, match="do not exist"):
        linker.evaluation.sample_pairwise_predictions(
            "unknown", budget=2, score_bins=[0.0, 0.5, 1.0]
        )

    # reversed duplicate
    d = df.copy()
    dup = d.iloc[[0]].copy()
    (
        dup["unique_id_l"],
        dup["unique_id_r"],
    ) = (
        d.loc[0, "unique_id_r"],
        d.loc[0, "unique_id_l"],
    )
    if "first_name_l" in dup.columns:
        dup["first_name_l"], dup["first_name_r"] = (
            d.loc[0, "first_name_r"],
            d.loc[0, "first_name_l"],
        )
    d = pd.concat([d, dup], ignore_index=True)
    linker.table_management.register_table(d, "dup", overwrite=True)
    with pytest.raises(ValueError, match="Duplicate unoriented pairs"):
        linker.evaluation.sample_pairwise_predictions(
            "dup", budget=2, score_bins=[0.0, 0.5, 1.0]
        )


def test_failure_leaves_no_temp_tables_and_linker_reusable():
    linker, pred = _linker_and_predictions()
    df = pred.as_pandas_dataframe()
    d = df.copy()
    d.loc[0, "unique_id_r"] = d.loc[0, "unique_id_l"]
    linker.table_management.register_table(d, "bad_self", overwrite=True)
    con = linker._db_api._con
    before = {r[0] for r in con.sql("show tables").fetchall()}
    with pytest.raises(ValueError):
        linker.evaluation.sample_pairwise_predictions(
            "bad_self", budget=2, score_bins=[0.0, 0.5, 1.0]
        )
    after = {r[0] for r in con.sql("show tables").fetchall()}
    leftovers = [t for t in after - before if t.startswith("__splink__sampling")]
    assert leftovers == []
    # linker still usable, and repeat successful calls do not accumulate tables
    n_tables = len(after)
    for _ in range(3):
        linker.evaluation.sample_pairwise_predictions(
            pred, budget=4, score_bins=[0.0, 0.5, 1.0], seed=2
        )
    final = {r[0] for r in con.sql("show tables").fetchall()}
    sampling_left = [t for t in final if t.startswith("__splink__sampling_")]
    # canonical/stratum scratch tables are dropped; only outputs remain and
    # each call uses a fresh unique suffix, so no scratch objects accumulate
    assert all("canonical" not in t and "strata" not in t for t in sampling_left)
    assert len(final) <= n_tables + 3


# --------------------------------------------------------------------------- #
# Multi-seed equal-probability check
# --------------------------------------------------------------------------- #
def test_multi_seed_within_stratum_frequencies():
    """Across many seeds, every member of a stratum is selected at a rate
    consistent with the claimed equal-probability SRSWOR design."""
    linker, table, _ = _hand_population_linker()
    bins = [0.0, 0.33, 0.66, 1.0]
    # fix a design: stratum 0 has N_0 = 7; force n_0 by choosing budget 3
    # (one per stratum).  Then each member of stratum 0 has pi = 1/7.
    pair_counts: dict = {}
    n_trials = 210
    for seed in range(n_trials):
        out = linker.evaluation.sample_pairwise_predictions(
            table, budget=3, score_bins=bins, seed=seed
        ).as_pandas_dataframe()
        stratum0 = out[out[SAMPLE_STRATUM_COLUMN] == 0]
        assert len(stratum0) == 1
        for pair in _canonical_pairs(stratum0):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    n0 = 7
    expected_rate = 1 / n0
    # chi-square goodness of fit against a uniform 1/7 selection rate
    counts = np.array([pair_counts.get(p, 0) for p in sorted(pair_counts)])
    assert len(counts) == n0
    expected = n_trials * expected_rate
    chi2 = np.sum((counts - expected) ** 2 / expected)
    # df = 6; 99th percentile of chi2(6) ~ 16.81.  A correct uniform design
    # will fall below this almost always; a fixed/biased draw would not.
    assert chi2 < 16.81, chi2
    # tighter practical check: observed proportions within +/- 9 percentage
    # points of 1/7 (~0.143) at 210 trials (standard error ~0.024).
    rates = counts / n_trials
    assert np.all(np.abs(rates - expected_rate) < 0.09), rates


def test_link_only_uses_full_source_plus_record_identity():
    # equal unique ids across the two sources must not collide
    linker, pred = _linker_and_predictions(n_records=8, link_only=True)
    out = linker.evaluation.sample_pairwise_predictions(
        pred, budget=10, score_bins=[0.0, 0.5, 1.0], seed=4
    ).as_pandas_dataframe()
    assert (out["source_dataset_l"] == "left_source").all()
    assert (out["source_dataset_r"] == "right_source").all()
    # self pairs across sources with the same id are legitimate (different
    # full identities) and must survive; duplicates must not appear
    pairs = list(
        zip(
            out["source_dataset_l"],
            out["unique_id_l"],
            out["source_dataset_r"],
            out["unique_id_r"],
        )
    )
    assert len(pairs) == len(set(pairs))


def test_singleton_source_dedupe_runs():
    # a single-record source yields an empty prediction table gracefully
    df = pd.DataFrame([{"unique_id": 0, "first_name": "a"}])
    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=[ExactMatch("first_name")],
        blocking_rules_to_generate_predictions=["1=1"],
    )
    linker = Linker(df, settings, db_api=DuckDBAPI())
    pred = linker.inference.predict()
    out = linker.evaluation.sample_pairwise_predictions(
        pred, budget=2, score_bins=[0.0, 1.0]
    ).as_pandas_dataframe()
    assert len(out) == 0


def test_empty_strata_get_no_sample_and_budget_caps_at_population():
    linker, table, _ = _hand_population_linker()
    # five edges -> four strata, but no score falls in [0.75, 1.0):
    # the hand fixture max below 1 is 0.9, so add an empty middle stratum
    # using edges that no score lands in
    bins = [0.0, 0.33, 0.60, 0.66, 1.0]
    out = linker.evaluation.sample_pairwise_predictions(
        table, budget=20, score_bins=bins, seed=2
    ).as_pandas_dataframe()
    present_strata = set(out[SAMPLE_STRATUM_COLUMN].unique())
    # stratum 2 ([0.60, 0.66)) contains no population rows and must not
    # appear in the output; every populated stratum is still represented
    assert 2 not in present_strata
    assert present_strata == {0, 1, 3}
    # budget larger than the population caps at the population size, and
    # every inclusion probability becomes exactly 1 (full census)
    assert len(out) == 15
    assert (out[INCLUSION_PROBABILITY_COLUMN] == 1.0).all()
    # populated stratum metadata remains internally consistent
    for h in present_strata:
        rows = out[out[SAMPLE_STRATUM_COLUMN] == h]
        assert (
            rows[STRATUM_SAMPLE_SIZE_COLUMN]
            == rows[STRATUM_POPULATION_COLUMN]
        ).all()


def test_custom_unique_id_column_contract():
    df = pd.DataFrame(
        [
            {"record_id": 1, "first_name": "a"},
            {"record_id": 2, "first_name": "a"},
            {"record_id": 3, "first_name": "b"},
        ]
    )
    settings = SettingsCreator(
        link_type="dedupe_only",
        unique_id_column_name="record_id",
        comparisons=[ExactMatch("first_name")],
        blocking_rules_to_generate_predictions=["1=1"],
        retain_matching_columns=True,
    )
    linker = Linker(df, settings, db_api=DuckDBAPI())
    pred = linker.inference.predict()
    out = linker.evaluation.sample_pairwise_predictions(
        pred, budget=3, score_bins=[0.0, 1.0], seed=1
    ).as_pandas_dataframe()
    assert {"record_id_l", "record_id_r"}.issubset(out.columns)
    assert len(out) == 3
    # self pair expressed with the custom id column is still rejected
    bad = pred.as_pandas_dataframe()
    bad.loc[0, "record_id_r"] = bad.loc[0, "record_id_l"]
    linker.table_management.register_table(bad, "custom_self", overwrite=True)
    with pytest.raises(ValueError, match="Self-pairs"):
        linker.evaluation.sample_pairwise_predictions(
            "custom_self", budget=2, score_bins=[0.0, 1.0]
        )


def test_duckdb_only_enforcement(sqlite_helper=None):
    # The feature is explicitly DuckDB-only in the first version.
    linker, pred = _linker_and_predictions()
    assert isinstance(linker._db_api, DuckDBAPI)
    # monkey-patch the type check to prove the guard exists and is meaningful
    from splink.internals import pairwise_sampling as mod

    class NotDuckDB:
        pass

    original = mod.DuckDBAPI
    try:
        mod.DuckDBAPI = NotDuckDB
        with pytest.raises(ValueError, match="DuckDB"):
            mod.sample_pairwise_predictions(
                linker, pred, budget=2, score_bins=[0.0, 1.0]
            )
    finally:
        mod.DuckDBAPI = original
        assert original is mod.DuckDBAPI


def test_stratum_boundary_scores_assigned_to_higher_bin():
    # score exactly on an internal edge goes to the upper, half-open stratum
    linker, table, _ = _hand_population_linker()
    out = linker.evaluation.sample_pairwise_predictions(
        table, budget=15, score_bins=[0.0, 0.33, 0.66, 1.0], seed=1
    ).as_pandas_dataframe()
    edge_33 = out[np.isclose(out["match_probability"], 0.33)]
    edge_66 = out[np.isclose(out["match_probability"], 0.66)]
    edge_one = out[np.isclose(out["match_probability"], 1.0)]
    edge_zero = out[np.isclose(out["match_probability"], 0.0)]
    assert (edge_33[SAMPLE_STRATUM_COLUMN] == 1).all()
    assert (edge_66[SAMPLE_STRATUM_COLUMN] == 2).all()
    assert (edge_one[SAMPLE_STRATUM_COLUMN] == 2).all()
    # no exact-zero rows in this fixture; guard logic instead via direct case
    assert (edge_zero[SAMPLE_STRATUM_COLUMN] == 0).all() if len(edge_zero) else True

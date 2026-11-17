"""Runnable end-to-end example of weighted clerical-label training.

Demonstrates, on a small hand-checkable dataset:

1. direct weighted m estimation from clerical scores and weights;
2. partially-labelled EM on both the term-frequency and compressed paths;
3. session.label_training_summary diagnostics;
4. JSON save/load with identical predictions and no label leakage.

Run with the project virtualenv, e.g.

    .venv/bin/python -I examples/weighted_labels/weighted_labels_example.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from splink import DuckDBAPI, Linker, SettingsCreator, block_on
import splink.internals.comparison_library as cm


def build_settings() -> SettingsCreator:
    return SettingsCreator(
        link_type="dedupe_only",
        comparisons=[
            cm.ExactMatch("first_name"),
            cm.LevenshteinAtThresholds("surname", 1),
        ],
        blocking_rules_to_generate_predictions=[block_on("first_name")],
        max_iterations=3,
    )


def main() -> None:
    records = pd.DataFrame(
        [
            {"unique_id": 1, "first_name": "A", "surname": "Smith"},
            {"unique_id": 2, "first_name": "A", "surname": "Smith"},
            {"unique_id": 3, "first_name": "A", "surname": "Smyth"},
            {"unique_id": 4, "first_name": "A", "surname": "Jones"},
            {"unique_id": 5, "first_name": "B", "surname": "Brown"},
            {"unique_id": 6, "first_name": "B", "surname": "Browne"},
        ]
    )

    # Clerical evidence: certain match, uncertain, certain non-match and a
    # cross-block uncertain pair.  The weight column is optional.
    labels = pd.DataFrame(
        [
            {
                "unique_id_l": 1, "unique_id_r": 2,
                "clerical_match_score": 1.0, "review_weight": 2.0,
            },
            {
                "unique_id_l": 1, "unique_id_r": 3,
                "clerical_match_score": 0.5, "review_weight": 2.0,
            },
            {
                "unique_id_l": 1, "unique_id_r": 4,
                "clerical_match_score": 0.0, "review_weight": 3.0,
            },
            {
                "unique_id_l": 5, "unique_id_r": 6,
                "clerical_match_score": 0.8, "review_weight": 1.0,
            },
        ]
    )

    # 1. Direct weighted m estimation ---------------------------------------- #
    direct = Linker(records, build_settings(), db_api=DuckDBAPI())
    direct.table_management.register_table(labels, "labels", overwrite=True)
    summary = direct.training.estimate_m_from_pairwise_labels(
        "labels",
        use_label_scores=True,
        label_weight_column="review_weight",
    )
    print("Direct m estimation summary:", summary.as_dict())
    for cc in direct._settings_obj.comparisons:
        for cl in cc._comparison_levels_excluding_null:
            print(f"  m[{cc.output_column_name}][{cl.comparison_vector_value}]"
                  f" = {cl.m_probability:.6f}")

    # Hand check for surname positive mass (weight * score):
    #   exact level (pair 1,2): 2 * 1.0 = 2.0
    #   lev   level (pairs 1,3 and 5,6): 2 * 0.5 + 1 * 0.8 = 1.8
    #   else  level (pair 1,4): 3 * 0.0 = 0.0
    # total positive mass = 3.8 -> m(exact) = 2/3.8, m(lev) = 1.8/3.8,
    # m(else) = 0
    surname = direct._settings_obj.comparisons[1]
    m = {cl.comparison_vector_value: cl.m_probability
         for cl in surname._comparison_levels_excluding_null}
    assert np.isclose(m[2], 2.0 / 3.8)
    assert np.isclose(m[1], 1.8 / 3.8)
    assert np.isclose(m[0], 0.0)

    # 2. Partially-labelled EM on both evidence paths ------------------------ #
    for compressed in (True, False):
        linker = Linker(records, build_settings(), db_api=DuckDBAPI())
        linker.table_management.register_table(labels, "labels", overwrite=True)
        session = linker.training.estimate_parameters_using_expectation_maximisation(
            block_on("first_name"),
            estimate_without_term_frequencies=compressed,
            fix_u_probabilities=False,
            pairwise_labels="labels",
            label_weight_column="review_weight",
        )
        print(
            f"\nEM path estimate_without_term_frequencies={compressed}:",
            session.label_training_summary.as_dict(),
        )

    # 3. JSON round trip: no labels in the JSON, identical predictions ------- #
    # Use a larger, realistic dataset for the round-trip check so that trained
    # u probabilities stay strictly positive. (On a tiny dataset a u value can
    # be driven to exactly 0, which the pre-existing JSON writer stores as
    # null; that is unrelated to the label feature.)
    import json
    import tempfile
    import os

    realistic = pd.read_csv("./tests/datasets/fake_1000_from_splink_demos.csv")
    r_pairs = []
    for _, a in realistic.head(50).iterrows():
        for _, b in realistic.head(50).iterrows():
            if (
                a["unique_id"] < b["unique_id"]
                and a["first_name"] == b["first_name"]
            ):
                r_pairs.append(
                    {
                        "unique_id_l": a["unique_id"],
                        "unique_id_r": b["unique_id"],
                        "clerical_match_score": (
                            1.0 if a["surname"] == b["surname"] else 0.3
                        ),
                    }
                )
    r_labels = pd.DataFrame(r_pairs[:60])
    r_settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=[
            cm.ExactMatch("first_name"),
            cm.ExactMatch("surname"),
            cm.ExactMatch("email"),
        ],
        blocking_rules_to_generate_predictions=[block_on("first_name")],
        max_iterations=3,
    )
    linker = Linker(realistic, r_settings, db_api=DuckDBAPI())
    linker.table_management.register_table(r_labels, "labels", overwrite=True)
    linker.training.estimate_u_using_random_sampling(max_pairs=1e4)
    linker.training.estimate_parameters_using_expectation_maximisation(
        "l.first_name = r.first_name",
        fix_u_probabilities=False,
        pairwise_labels="labels",
    )
    linker.training.estimate_m_from_pairwise_labels(
        "labels", use_label_scores=True
    )

    model_json = linker.misc.save_model_to_json()
    blob = json.dumps(model_json)
    assert "clerical_match_score" not in blob
    assert "review_weight" not in blob

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.json")
        linker.misc.save_model_to_json(path, overwrite=True)
        reloaded = Linker(realistic, path, db_api=DuckDBAPI())
        before = (
            linker.inference.predict()
            .as_pandas_dataframe()
            .sort_values(["unique_id_l", "unique_id_r"])
            .reset_index(drop=True)
        )
        after = (
            reloaded.inference.predict()
            .as_pandas_dataframe()
            .sort_values(["unique_id_l", "unique_id_r"])
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            before[["unique_id_l", "unique_id_r", "match_probability"]],
            after[["unique_id_l", "unique_id_r", "match_probability"]],
            rtol=1e-12,
            atol=1e-12,
        )
    print("\nAll example checks passed.")


if __name__ == "__main__":
    main()

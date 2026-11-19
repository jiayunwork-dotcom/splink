"""End-to-end example: stratified review sampling and weighted evaluation.

The clerical review budget is limited, so reviewing a uniform random subset
is often impractical: the interesting (high-score) part of the prediction
table would contain too few pairs to characterise precisely, while the
low-score tail would consume most of the budget.  Splink therefore supports a

1. stratified, reproducible sample from the full generated prediction table
   with a known per-pair inclusion probability,
2. clerical labelling of just that sample,
3. weighted evaluation in which every reviewed pair represents
   ``1 / inclusion_probability`` pairs of the generated population.

The estimated quantities describe the *generated* prediction population only.
Splink never generates pairs that fail its blocking rules, so no claim is
made about the ungenerated all-pairs universe, and no confidence intervals
are produced.

Run with the DuckDB backend (the only backend supported by the sampling
entry point in this version):

    python weighted_sampling_evaluation_example.py
"""
import pandas as pd

from splink import DuckDBAPI, Linker, SettingsCreator, block_on
from splink.internals.comparison_library import ExactMatch

# --------------------------------------------------------------------------- #
# 1. A small input dataset and a normal predict() call, with no score cut.
# --------------------------------------------------------------------------- #
data = pd.DataFrame(
    [
        {"unique_id": 1, "first_name": "robin", "cluster": "X"},
        {"unique_id": 2, "first_name": "robin", "cluster": "X"},
        {"unique_id": 3, "first_name": "robin", "cluster": "Y"},
        {"unique_id": 4, "first_name": "david", "cluster": "Z"},
        {"unique_id": 5, "first_name": "david", "cluster": "W"},
    ]
)

settings = SettingsCreator(
    link_type="dedupe_only",
    probability_two_random_records_match=0.5,
    retain_matching_columns=True,
    comparisons=[ExactMatch("first_name")],
    blocking_rules_to_generate_predictions=[block_on("first_name"), "1=1"],
)
linker = Linker(data, settings, db_api=DuckDBAPI())

# The full, untruncated prediction table is the sampling population.
predictions = linker.inference.predict()

# --------------------------------------------------------------------------- #
# 2. Stratified sampling under a fixed review budget.
# --------------------------------------------------------------------------- #
# Four score strata; every non-empty stratum is guaranteed at least one
# reviewed pair, and the remaining budget is apportioned proportionally to
# stratum size.  The draw is label-agnostic, reproducible for this exact
# (data, bins, budget, seed) combination and invariant to row order, table
# name and the left/right orientation of a pair.
sampled = linker.evaluation.sample_pairwise_predictions(
    predictions,
    budget=6,
    score_bins=[0.0, 0.25, 0.5, 0.75, 1.0],
    seed=8,
)
sampled_df = sampled.as_pandas_dataframe()
print("Sampled pairs and their design metadata:")
print(
    sampled_df[
        [
            "unique_id_l",
            "unique_id_r",
            "match_probability",
            "sample_stratum",
            "stratum_population",
            "stratum_sample_size",
            "inclusion_probability",
        ]
    ].to_string(index=False)
)

# --------------------------------------------------------------------------- #
# 3. Clerical review happens outside Splink (here simulated from the known
#    clusters).  The inclusion_probability column must be carried along with
#    the clerical_match_score; it is the only link back to the design.
# --------------------------------------------------------------------------- #
cluster = data.set_index("unique_id")["cluster"]
reviewed = sampled_df[
    ["unique_id_l", "unique_id_r", "inclusion_probability"]
].copy()
reviewed["clerical_match_score"] = [
    1.0 if cluster[a] == cluster[b] else 0.0
    for a, b in zip(reviewed["unique_id_l"], reviewed["unique_id_r"])
]
labels_table = linker.table_management.register_labels_table(reviewed)

# --------------------------------------------------------------------------- #
# 4. Weighted threshold table, charts and error list.
# --------------------------------------------------------------------------- #
truth_space = linker.evaluation.accuracy_analysis_from_labels_table(
    labels_table,
    output_type="table",
    match_weight_round_to_nearest=None,
    inclusion_probability_column="inclusion_probability",
).as_pandas_dataframe()

print("\nWeighted truth space (masses, not row counts):")
print(
    truth_space[
        [
            "truth_threshold",
            "tp",
            "fp",
            "tn",
            "fn",
            "precision",
            "recall",
            "sampled_pair_count",
            "estimated_pair_count",
        ]
    ].to_string(index=False)
)

# All four chart types consume the weighted results; on the weighted path
# their tooltips show both the actual reviewed count and the estimated
# generated-population mass.
for output_type in ("roc", "precision_recall", "accuracy", "threshold_selection"):
    chart = linker.evaluation.accuracy_analysis_from_labels_table(
        labels_table,
        output_type=output_type,
        match_weight_round_to_nearest=None,
        inclusion_probability_column="inclusion_probability",
    )
    assert "sampled_pair_count" in str(chart.to_dict())

errors = linker.evaluation.prediction_errors_from_labels_table(
    labels_table,
    inclusion_probability_column="inclusion_probability",
).as_pandas_dataframe()
print("\nWeighted error pairs (each carries the mass it represents):")
if len(errors):
    print(
        errors[
            [
                "unique_id_l",
                "unique_id_r",
                "truth_status",
                "inclusion_probability",
                "evaluation_weight",
            ]
        ].to_string(index=False)
    )
else:
    print("(no errors at the default 0.5 threshold for this sample)")

# --------------------------------------------------------------------------- #
# 5. The same labels without the probability column retain the legacy
#    unweighted behaviour and chart appearance.
# --------------------------------------------------------------------------- #
legacy_labels = linker.table_management.register_labels_table(
    reviewed.drop(columns=["inclusion_probability"]),
    overwrite=True,
)
legacy = linker.evaluation.accuracy_analysis_from_labels_table(
    legacy_labels,
    output_type="table",
    match_weight_round_to_nearest=None,
).as_pandas_dataframe()
assert "sampled_pair_count" not in legacy.columns
print("\nLegacy unweighted evaluation remains available and unchanged.")

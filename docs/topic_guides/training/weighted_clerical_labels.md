# Weighted clerical labels in training

When human reviewers audit candidate record pairs, the resulting evidence is
not always a binary "match / no match". A reviewer may be uncertain, and the
same pair may have been reviewed more than once. Splink 4.0.8's
`estimate_m_from_pairwise_labels` treated *every* supplied pair as a certain
match and the EM trainer could not consume pairwise labels at all.

The weighted-label training described here gives one consistent meaning to a
piece of clerical evidence across both training entry points:

* `linker.training.estimate_m_from_pairwise_labels(...)` — direct m
  estimation, and
* `linker.training.estimate_parameters_using_expectation_maximisation(...)` —
  the EM trainer.

## The evidence contract

Each normalised label is an **unoriented pair of fully identified records**
(source dataset + unique id together identify each side, so equal ids in
different datasets never collide) with:

* a `clerical_match_score` in the closed interval `[0, 1]`:
  * `1` — confirmed match,
  * `0` — confirmed non-match,
  * a fractional value — retained uncertainty, interpreted as the matching
    quality share of the pair;
* a strictly positive, finite `label_weight` (default `1`). Integer weights
  behave exactly like repeating the same independent pair evidence that many
  times.

Repeated or reversed submissions of the same pair with the **same** score and
weight are merged into one piece of evidence whose weight is the sum. The same
pair supplied with *conflicting* scores or weights is a hard `ValueError` —
conflicts are never silently averaged or dropped.

All normalisation and validation happens inside the SQL backend; only
parameter-level aggregates (never the full pair table) cross into Python.

## Direct, weighted m estimation

```python
linker.table_management.register_table(
    pairwise_labels, "labels", overwrite=True
)

summary = linker.training.estimate_m_from_pairwise_labels(
    "labels",
    use_label_scores=True,
    label_weight_column="review_weight",  # optional; default weight is 1
)
```

For each comparison, the m probability at a non-null level `k` is the weighted
positive matching mass observed at that level divided by the weighted positive
mass over all non-null levels:

```
m_k = Σ(weight · score · 1[γ = k]) / Σ(weight · score · 1[γ ≠ null])
```

The u probabilities and the global prior are left unchanged. Each comparison's
own null level never enters its normalisation. If the whole evidence set has
zero positive mass (e.g. every score is `0`) a `ValueError` is raised and the
existing model is untouched.

The default `use_label_scores=False` preserves the historical behaviour: every
row is treated as a certain match and any `clerical_match_score` column is
ignored. `label_weight_column` may only be supplied in score mode.

## Labels inside expectation maximisation

```python
session = linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("first_name"),
    estimate_without_term_frequencies=False,  # or True
    fix_u_probabilities=False,
    pairwise_labels="labels",
    label_weight_column="review_weight",  # optional
)
```

* **Labelled pairs** hold their clerical score fixed in every iteration. Their
  contribution to an m count is `Σ(weight · score)` and to a u count
  `Σ(weight · (1 − score))`; the same weight scales their contribution to the
  training-block prior.
* **Unlabelled pairs** continue to be scored by the current model with
  implicit weight 1.
* The training block is still defined solely by `blocking_rule`. A label
  referencing a pair the blocking rule does not generate raises a
  `ValueError`; labels can never enlarge a block. Comparisons "used up" by the
  blocking rule remain untrained as before.
* Labels affect training only. They are never attached to
  `linker.inference.predict()` output and are never written into the saved
  model JSON.

Both EM paths honour the same contract. With
`estimate_without_term_frequencies=True` the unlabelled rows are compressed to
agreement-pattern counts, but labelled rows are deliberately **not** folded
into that compression, so identical comparison vectors carrying different
clerical scores remain distinct evidence. With `False`, term-frequency
adjustments still apply to the unlabelled rows. The two paths may therefore
produce different parameters — that is expected.

### When effective mass is zero

If a comparison's effective m mass (or u mass) is zero for an iteration, its
previous m (or u) parameters are retained rather than overwritten with `NaN`.
The session diagnostics list these under `not_estimated`, and no convergence is
claimed on a parameter that did not move because it had no evidence.

## Session diagnostics

`session.label_training_summary` exposes only aggregate, parameter-level
information:

```python
session.label_training_summary.as_dict()
# {
#   "labelled_pair_count": 42,
#   "unlabelled_pair_count": 2211,
#   "label_weight_sum": 42.0,
#   "positive_mass_sum": 34.3,
#   "negative_mass_sum": 7.7,
#   "stop_reason": "converged" | "max_iterations_reached",
#   "not_estimated": [{"output_column_name": "surname",
#                      "parameter": "u", "reason": "zero_mass"}, ...],
# }
```

## Validation rules

The following all raise `ValueError` *before* any model state changes, so a
failed call leaves parameters, training history and predictions exactly as
they were:

* missing required id / `clerical_match_score` columns, or a missing weight
  column;
* non-finite or out-of-range scores (outside `[0, 1]`, `NaN`, infinities) or
  non-finite / non-positive weights;
* self-pairs; pairs violating `link_type` (e.g. same-dataset pairs for
  `link_only`); pairs referencing records that do not exist;
* conflicting evidence for the same canonical pair;
* labels outside the EM training block;
* a weight column supplied without score mode, or without labels.

An empty but schema-valid label table passed to EM is equivalent to passing no
labels at all.

## Saving and reloading

Models keep using the existing JSON save/load. The JSON contains trained
parameters only — never the raw label table, weights or per-pair clerical
scores — so predictions from a reloaded model are identical to the pre-save
model.

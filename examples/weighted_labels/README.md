# Weighted clerical pairwise-label training

This feature adds optional **weighted, probabilistic clerical labels** to both
Splink training entry points, while keeping the historical behaviour as the
default.

## Public API

### 1. Direct m estimation

```python
linker.training.estimate_m_from_pairwise_labels(
    labels,                       # registered table name or SplinkDataFrame
    use_label_scores=False,       # default: legacy certain-match behaviour
    label_weight_column=None,     # optional positive weight column
)
```

### 2. Expectation maximisation

```python
session = linker.training.estimate_parameters_using_expectation_maximisation(
    blocking_rule,
    estimate_without_term_frequencies=False,   # or True
    pairwise_labels=None,          # register table name or SplinkDataFrame
    label_weight_column=None,      # optional positive weight column
)
session.label_training_summary     # aggregate diagnostics only
```

Defaults are unchanged: callers that pass neither new argument get exactly the
previous behaviour, including identical numerical results.

## The evidence contract

A normalised label is an **unoriented pair of fully identified records** with a
finite `clerical_match_score` in `[0, 1]` and a finite, strictly positive
`label_weight` (default `1`):

* `score = 1` confirmed match, `score = 0` confirmed non-match, fractional =
  retained uncertainty;
* source dataset + unique id identify each side, so equal ids across datasets
  never collide;
* left/right swaps and exact repeats merge into one piece of evidence, summing
  the weight (an integer weight is equivalent to that many repeated
  observations);
* conflicting scores/weights for the same pair raise `ValueError`;
* self pairs, unknown records, `link_type` violations, missing columns,
  non-finite or out-of-range scores/weights all raise `ValueError` before any
  model state changes;
* EM labels must lie inside the training block (they cannot enlarge it); an
  empty but valid label table is equivalent to no labels;
* each comparison's own null level is excluded from its m/u normalisation;
* labels influence training only — never `predict()`, never the saved JSON.

## How the two estimators consume the evidence

### Direct weighted m

For comparison level `k`:

```
m_k = Σ(weight · score · 1[γ=k]) / Σ(weight · score · 1[γ≠null])
```

u and the global prior are untouched. Zero total positive mass raises
`ValueError`. A comparison with no positive mass on any non-null level keeps
its current m values.

### Partially-labelled EM

Each iteration separates the blocked comparison vectors into two tables:

* **labelled rows** hold the clerical score fixed: m mass
  `Σ weight·score`, u mass `Σ weight·(1−score)`, with the same weight also
  scaling their contribution to the block prior;
* **unlabelled rows** are scored by the current model with implicit weight 1.

The two sets of per-level counts are combined on parameter-level data and
normalised. Where a comparison's effective m or u mass is zero, the previous
parameter is retained and the comparison/parameter is reported under
`not_estimated` in `session.label_training_summary` — no `NaN` is ever written
and no convergence is claimed on an unmoved parameter.

## Design: what is compressed, what stays pair-level

The requirement is to let a large unlabelled majority and a small set of
weighted labels iterate together without pulling the full pair table into
Python.

* **Pair and probability work stays in DuckDB.** Blocking, label
  normalisation/validation, the labelled/unlabelled split, expectation scoring
  and all count aggregation run as SQL. Only parameter-level count tables
  (one row per comparison level plus a prior row) cross into Python.
* **Compressed path (`estimate_without_term_frequencies=True`).** The
  *unlabelled* rows are aggregated to agreement-pattern counts (small: at most
  one row per realised vector). The *labelled* rows are deliberately **not**
  folded into that compression, because two rows with identical comparison
  vectors but different clerical scores are distinct evidence. Their fixed
  mass counts are computed once and reused every iteration.
* **Term-frequency path (`False`).** Unlabelled rows retain their per-pair
  term-frequency adjustments exactly as in standard Splink EM; labelled rows
  still use their fixed scores. The two paths share the same evidence contract
  but may legitimately produce different parameters.

### Costs

* **SQL workload:** one extra pair-level join to split labelled from
  unlabelled rows (once per session), one fixed labelled aggregation (once),
  then per iteration the same scoring/aggregation as standard EM over the
  unlabelled rows plus a cheap parameter-level combine. The labelled table is
  typically ~1% of the block and is never re-scored by the model.
* **Temporary table footprint:** in addition to the usual blocked/CVV tables,
  two pair-level tables (labelled and unlabelled CVV) and one small labelled
  count table are materialised. The compressed path adds the small
  agreement-pattern table. No whole-pair table is held in Python.
* **Numerical reliability:** all weighted sums are computed in `float8` inside
  DuckDB; division happens only after checking denominators, with zero-mass
  classes retaining prior parameters. Direct m and the first all-labelled EM
  update agree to `rtol=1e-9, atol=1e-12` on trainable comparisons.

## Measured results

Four three-level comparisons, dominant-block training, ~1% labelled pairs,
three EM iterations, DuckDB 1.1.3. `python_peak_mib` is the extra Python
allocation measured with `tracemalloc` starting after input registration and
shared term-frequency preparation; DuckDB memory and temp-disk usage are
reported separately.

| block pairs | EM path        | wall s (3 iter) | Python peak MiB | DuckDB mem MiB | DuckDB temp MiB |
|-------------|----------------|-----------------|-----------------|----------------|-----------------|
| 199,396     | no-TF (fast)   | 0.239           | 0.377           | 14.5           | 0.0             |
| 199,396     | with-TF        | 0.288           | 0.280           | 14.5           | 0.0             |
| 998,991     | no-TF (fast)   | 0.397           | 0.281           | 9.0            | 0.0             |
| 998,991     | with-TF        | 0.539           | 0.279           | 9.0            | 0.0             |

The Python extra peak stays far below the 128 MiB budget (sub-MiB), because no
pairwise data is materialised in Python. In this in-memory run DuckDB did not
spill to temp disk; with a lower `memory_limit` or larger data it may, and that
footprint is captured separately by the experiment.

## Reproducing

From the repository root, with Python 3.11 and the pinned stack
(duckdb 1.1.3, pandas 2.2.3, numpy 1.26.4, sqlglot 25.34.1):

```bash
# create an isolated environment (uv shown; venv/pip work too)
uv venv --python 3.11 .venv
uv pip install -p .venv/bin/python \
  "duckdb==1.1.3" "pandas==2.2.3" "numpy==1.26.4" "sqlglot==25.34.1" \
  jsonschema Jinja2 altair pytest pyarrow==17.0.0
uv pip install -p .venv/bin/python -e .

# focused feature tests (hand-checkable direct + EM, validation, JSON)
.venv/bin/python -m pytest tests/test_weighted_pairwise_labels.py -q

# broader regression (DuckDB)
.venv/bin/python -m pytest tests/ -q \
  -k "not spark and not postgres and not athena and not sqlite"

# runnable end-to-end example
.venv/bin/python -I examples/weighted_labels/weighted_labels_example.py

# scale experiment (writes its own temp DuckDB directory per run)
.venv/bin/python -I examples/weighted_labels/scale_experiment.py \
  --targets 200000 1000000
```

## Files

* `splink/internals/labelled_pairwise.py` — normalisation, validation,
  labelled/unlabelled split and fixed mass-count SQL;
* `splink/internals/m_from_labels.py` — legacy and weighted direct m paths;
* `splink/internals/expectation_maximisation.py` — partially-labelled EM,
  zero-mass retention, parameter-level combine;
* `splink/internals/em_training_session.py` — session wiring, block-membership
  check, `label_training_summary`;
* `splink/internals/linker_components/training.py` — public keyword arguments;
* `tests/test_weighted_pairwise_labels.py` — hand-checkable and contract tests;
* `docs/topic_guides/training/weighted_clerical_labels.md` — user guide.

# Stratified review sampling and weighted quality evaluation

A separate, evaluation-only layer sits on top of the same pairwise identity
contract. It addresses the case where clerical review has a fixed budget and
is deliberately non-uniform across the score range, so counting reviewed rows
no longer represents the original prediction table.

## Public API

```python
# 1. reproducible stratified sample from the full generated predictions
sampled = linker.evaluation.sample_pairwise_predictions(
    predictions,                 # untruncated predict() table or table name
    budget=1000,                 # positive Python int
    score_bins=[0.0, 0.5, 0.9, 1.0],
    seed=42,                     # int in [0, 2**32)
)

# 2. after clerical review, the same table carries clerical_match_score and
#    the inclusion_probability produced above
linker.evaluation.accuracy_analysis_from_labels_table(
    labels_table,
    inclusion_probability_column="inclusion_probability",  # None = legacy
)
linker.evaluation.prediction_errors_from_labels_table(
    labels_table,
    inclusion_probability_column="inclusion_probability",
)
```

Each reviewed pair represents `1 / inclusion_probability` pairs of the
*generated* prediction population (its `evaluation_weight`). The truth-space
`p/n/tp/tn/fp/fn` and every derived metric become weighted masses, the table
adds `sampled_pair_count` and `estimated_pair_count`, and the four chart
tooltips show both counts. The population is exactly the generated prediction
table: no claim is made about pairs blocking never generated, and no
confidence intervals are produced.

## Design and measured results

See `docs/topic_guides/evaluation/weighted_sampling_design.md` for the
allocation rule (one pair per non-empty stratum plus largest-remainder
proportional allocation), the within-stratum canonical-hash SRSWOR draw, the
rejected alternatives and the reproducibility/memory trade-offs.

Measured at `budget = 1000` over four strata (Python 3.11, DuckDB 1.1.3):

| prediction pairs | SQL wall time | Python extra peak | DuckDB memory | mass-conservation error |
|-----------------:|--------------:|------------------:|--------------:|------------------------:|
| 100,000 | 0.17 s | 0.08 MiB | 3.4 MiB | 0.0 |
| 1,000,000 | 0.42 s | 0.08 MiB | 30.9 MiB | 0.0 |

## Files

* `splink/internals/pairwise_sampling.py` — argument validation, canonical
  identities, largest-remainder allocation, hash-ranked SRSWOR;
* `splink/internals/weighted_evaluation.py` — probability-column validation
  and the weighted truth-space SQL;
* `splink/internals/accuracy.py`, `block_from_labels.py`,
  `comparison_vector_values.py`, `predict.py`, `charts.py` and
  `linker_components/evaluation.py` — threading the probability through the
  existing blocking → comparison vectors → prediction → aggregation → chart
  chain, with `None` preserving the legacy behaviour;
* `tests/test_pairwise_sampling.py`,
  `tests/test_weighted_evaluation.py` — hand-enumerated and contract tests;
* `examples/weighted_labels/weighted_sampling_evaluation_example.py` —
  end-to-end worked example;
* `examples/weighted_labels/sampling_scale_experiment.py` — scale/resource
  experiment;
* `docs/topic_guides/evaluation/weighted_sampling_design.md` — design and
  population-boundary documentation.

## Reproducing

```bash
.venv/bin/python -m pytest \
  tests/test_pairwise_sampling.py tests/test_weighted_evaluation.py -q

.venv/bin/python -I \
  examples/weighted_labels/weighted_sampling_evaluation_example.py

.venv/bin/python -I \
  examples/weighted_labels/sampling_scale_experiment.py \
  --sizes 100000 1000000 --budget 1000
```

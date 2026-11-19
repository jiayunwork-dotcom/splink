# Stratified quality evaluation of predicted pairwise matches

## Problem and population boundary

Clerical review budget is limited and the review sample is deliberately not
uniform across the score range: few pairs are checked in the low-score strata
and many in the high-score strata.  After review, the raw number of correct
rows therefore no longer represents the original prediction table.

The population for every estimate produced here is **exactly the table of
generated pairwise predictions passed to
`linker.evaluation.sample_pairwise_predictions(...)`**.  Splink only ever
generates pairs that pass its blocking rules, so the machinery in this design:

* estimates match quality (precision-like quantities) over the *generated*
  prediction table;
* does **not** say anything about pairs that blocking never generated, and in
  particular does not claim to estimate the recall of the full all-pairs
  universe;
* does **not** provide confidence intervals.

Each reviewed pair represents `1 / inclusion_probability` population pairs
(its Horvitz–Thompson mass, called `evaluation_weight` in the error table).
The sampling inclusion probabilities are decided **before any clerical label
is looked at**: the allocation only depends on stratum populations, the budget
and the seed — never on labels, input row order or the physical table name.

## The sampling contract

`linker.evaluation.sample_pairwise_predictions(predictions, *, budget, score_bins, seed=0)`

* `predictions`: a `SplinkDataFrame` (e.g. the result of
  `linker.inference.predict()` with no score threshold) or a registered table
  name.  One row is one **unoriented** pair of fully identified records.
* `score_bins`: strictly increasing finite Python `float`/`int` list, starting
  at `0` and ending at `1` (`bool` rejected).  Adjacent edges define
  half-open strata `[b_h, b_{h+1})`; the final stratum contains `1`.
* `budget`: positive non-`bool` Python `int`.
* `seed`: non-`bool` Python `int` in `[0, 2**32)`.

Validation (all raise `ValueError`, nothing is mutated or dropped):

* non-finite or out-of-range `match_probability`;
* self-pairs (the two full identities are equal);
* unknown nodes (identities absent from the linker's concat table);
* repeated unoriented pairs (canonical identity seen twice, including reversed
  submissions);
* `budget < number of non-empty strata`;
* missing required identity / score columns.

An empty population returns an empty table with the full output column list.

Output columns: every input column (the pair identities,
`match_probability`, `match_weight`, ...) plus

| column | meaning |
|---|---|
| `sample_stratum` | zero-based stratum index |
| `stratum_population` | number of population pairs in that stratum |
| `stratum_sample_size` | number actually drawn from that stratum |
| `inclusion_probability` | probability that this pair was drawn |

Every non-empty stratum gets at least one pair; total sample size is
`min(budget, N)` where `N` is the population size; within a stratum sampling
is simple random sampling **without replacement** with equal probabilities, so
`inclusion_probability = n_h / N_h` for every drawn row of stratum `h`.

### Allocation of the budget across strata

Let `N_h` be the population of stratum `h` and `H'` the number of non-empty
strata.

1. Each non-empty stratum receives a floor allocation of one pair.
2. The remaining `budget - H'` pairs are apportioned proportionally to `N_h`
   using the **largest-remainder (Hamilton) method**: each stratum gets the
   integer part of `(N_h / N) * (budget - H')`, then leftover slots go one by
   one to the strata with the largest fractional remainders, ties broken by
   ascending stratum index.  Every allocation is capped at `N_h` (a stratum
   cannot give more pairs than it contains); any unspent slot caused by a cap
   is redistributed to the remaining non-saturated strata by the same
   largest-remainder rule.
3. If `budget >= N` the whole population is taken (`n_h = N_h`, probabilities
   are all `1`).

The allocation is computed in Python from the stratum counts only — its
memory grows with the number of strata, never with `N`.

**Why this allocation.**  Proportional allocation makes the overall sampling
fraction as even as the one-pair floor allows, which minimises the variance of
population-level Horvitz–Thompson totals (and therefore of precision/recall
aggregates) when within-stratum variances are comparable.  What is sacrificed
is *per-stratum* precision: a tiny stratum is oversampled relative to its
population share (its `n_h/N_h` is inflated), so its individual rate estimates
are noisier per reviewed pair than under equal allocation, and large strata
receive fewer reviewed pairs than an equal-allocation design would give them.
The floor guarantees that no non-empty stratum is completely invisible in the
review.  An alternative (equal allocation `n_h ≈ budget / H'`) would equalise
per-stratum precision but inflate the variance of the global estimates; it is
discussed in the alternatives section below.

### Within-stratum draw — chosen design (Option A): canonical hash ranking

All work stays in DuckDB; Python never materialises the pair identities:

1. A **canonical identity** is built per side as
   `cast(source_dataset as varchar) || '-__-' || cast(unique_id as varchar)`
   (or just `cast(unique_id as varchar)` for `dedupe_only`), and each row is
   oriented with `least(id_l, id_r)` on the left and `greatest(...)` on the
   right.  Reversal of the input endpoints therefore changes nothing, and the
   composite source+id key means equal ids in different datasets never
   collide.
2. A deterministic pseudo-random 64-bit key is computed in SQL as
   `hash(cast(seed as varchar) || chr(1) || canonical_key_l || chr(1) ||
   canonical_key_r)` (DuckDB `hash`, stable inside a DuckDB release, salted by
   the seed).
3. `row_number() over (partition by sample_stratum order by <hash key>,
   canonical_key_l, canonical_key_r)` ranks rows inside each stratum.  The
   canonical-key tie breakers make the ordering total, so the result cannot
   depend on physical row order or table name.
4. Rows with rank `<= n_h` are the sample; metadata columns are joined back
   from the small per-stratum allocation table and the inclusion probability is
   `n_h / N_h`.

Properties: exact SRSWOR with equal probabilities *as a design* (every
`n_h`-subset of a stratum is equally likely across seeds), fully reproducible
for fixed inputs/edges/budget/seed, invariant to row order, endpoint reversal
and re-registration under a new physical name.  Cost: one scan for
validation/counts, one scan plus a window sort for the draw (DuckDB can spill
the sort to its controlled temp directory), temp space of order the sample
plus sorting buffers; only `H'` count rows and the final sample (≤ budget
rows) ever cross into Python.

### Rejected / alternative designs

* **Option B — `random()` ordering per stratum.**  Rank by
  `row_number() over (partition by stratum order by random())` after a
  `setseed(seed)` on the connection.  Same scan/sort cost and temp profile as
  Option A and equally valid SRSWOR, but DuckDB's `setseed` is connection
  global (it perturbs every concurrent query on the connection), its stream is
  consumed by any extra expression evaluated on the rows, and reproducibility
  then depends on query pipeline details rather than only on the declared
  inputs.  Option A has none of these hazards and needs no connection-level
  state, so it is the implemented design.
* **Bernoulli / Poisson per-row acceptance.**  Accept each row independently
  with probability `π_h`; no sort is needed (single scan), but sample sizes are
  random, the "every non-empty stratum contributes at least one" and
  "total size equals `min(budget, N)`" contracts cannot be guaranteed, and a
  second repair stage would be required — rejected.
* **Python reservoir sampling.**  Would require streaming every pair identity
  through Python, violating the memory contract (no full population or full
  identity set in Python) — rejected.
* **Cross-stratum systematic sampling on sorted scores.**  Single cheap scan,
  but cannot guarantee an integer SRSWOR size per stratum and produces
  systematic rather than equal-probability random designs — rejected.

## Threading inclusion probabilities through the existing evaluation chain

`accuracy_analysis_from_labels_table` and `prediction_errors_from_labels_table`
gain a keyword-only `inclusion_probability_column=None`.

* `None`: byte-for-byte legacy behaviour, unchanged SQL and charts.
* Column name given: the labels table must contain that column with finite,
  strictly positive values `<= 1` (null/non-numeric/`0`/negative/`>1`/NaN/inf
  all raise `ValueError`).  Existing checks still apply: unknown nodes,
  self-pairs and duplicated unoriented labels raise rather than dropping rows.
  Validation runs before any model state is touched and never rewrites the
  input table.

The probability column is carried through the *same* SQL chain as
`clerical_match_score`: label normalisation/join
(`block_from_labels`), comparison vectors, prediction, and the truth-space
aggregations.  No pandas recomputation, no parallel evaluation path.

### Weighted truth space

Every reviewed pair carries mass `w = 1 / inclusion_probability`.  The
existing cumulative threshold SQL is reused with weighted sums:

* `count(*)` becomes `sum(evaluation_weight)`;
* `sum(clerical_positive)` becomes
  `sum(evaluation_weight * clerical_positive)` and likewise for negatives;
* the threshold rounding, blocking-rule (`found_by_blocking_rules`) logic and
  all zero-denominator conventions are untouched;
* `precision`, `recall` and every `add_metrics` expression keep their existing
  formulas but operate on the weighted `tp/tn/fp/fn`, so they derive from the
  same weighted statistics.

New columns on the weighted truth-space table:

* `sampled_pair_count` — actual number of clerically reviewed pairs (an
  integer count, identical on every threshold row);
* `estimated_pair_count` — estimated total population mass
  `sum(1/π)` (may be fractional).

The existing `total_clerical_labels`, `p`, `n`, `tp`, `tn`, `fp`, `fn` become
**weighted masses** on this path — they are never described as numbers of
reviewed rows.  When the entire population was sampled (`π = 1` everywhere)
all weights equal one and the weighted results reproduce the legacy
evaluation exactly.

### Weighted error table

`prediction_errors_from_labels_table(..., inclusion_probability_column=...)`
keeps every existing identity/prediction column, retains the probability
column and adds `evaluation_weight = 1 / π`.  The legacy equality conventions
at the threshold are deliberately preserved and separately tested:

* errors use strict inequalities (`match_probability > t` for FP,
  `< t` for FN) — a pair exactly at the threshold is neither FP nor FN in this
  table;
* the truth-space table classifies on `match_weight >= truth_threshold`
  (equality included).

For pairs not sitting exactly on a threshold, weighted error mass reconciles
with the weighted truth-space confusion masses.

### Charts

The four chart entry points (threshold selection, ROC, precision–recall,
accuracy) consume the weighted truth-space records, so their plotted curves
use the weighted statistics.  On the weighted path each chart's tooltip gains
two fields on top of the existing ones:

* `Sampled pairs (reviewed): sampled_pair_count`;
* `Estimated population pairs: estimated_pair_count` (formatted to allow
  decimals).

Unweighted charts keep their current tooltips and formats exactly.

## Separation from training weights

The parent product's `label_weight` / `__splink_label_weight` represents
evidence frequency for m/EM training.  The sampling inclusion probability is
a property of the *review design*, not evidence:

* evaluation operations never call training code and never modify model
  parameters;
* training defaults never read the sampling metadata —
  `estimate_m_from_pairwise_labels` and EM only consume a weight column when
  the caller explicitly names `label_weight_column`;
* model JSON serialisation is unchanged and contains no sampling metadata.

## Scale and resource behaviour

* Pair processing stays entirely in DuckDB 1.1.3, which may spill its sort to
  the configured temp directory.
* Python memory scales with the number of strata (allocation bookkeeping) and
  the final sample size only — never with `N`, and no all-pairs re-enumeration
  is performed.
* Only temp objects created by this feature are dropped; caller-registered
  tables are preserved; failed calls leave no temp tables behind and the same
  linker remains usable, with repeat calls not accumulating objects.

### Measured scale run (Python 3.11, DuckDB 1.1.3, `budget = 1000`, four strata)

Reproduce with
`python examples/weighted_labels/sampling_scale_experiment.py
--sizes 100000 1000000 --budget 1000` (synthetic prediction tables built
inside DuckDB; no pair identities cross into Python).

| population pairs | sample size | SQL wall time | Python extra peak | DuckDB memory | DuckDB temp disk | mass-conservation error |
|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 1,000 | 0.17 s | 0.08 MiB | 3.4 MiB | 0 MiB | 0.0 |
| 1,000,000 | 1,000 | 0.42 s | 0.08 MiB | 30.9 MiB | 0 MiB | 0.0 |

`sum(1 / inclusion_probability)` over the sample equals the population size
exactly within NumPy 1.26.4 float64 tolerance on every run.  The Python peak
is independent of the population size because only the per-stratum counts and
the 1,000-row sample are materialised in Python.

## What this design does not do

* It does not estimate recall over pairs that blocking never generated — the
  population is the generated prediction table and nothing more.
* It provides no confidence intervals or variance estimators.
* It does not change model training: evaluation never calls m/EM training,
  and the training entry points only consume a frequency weight when the
  caller explicitly names `label_weight_column`.  Sampling metadata never
  enters the model JSON.

An end-to-end worked example lives in
`examples/weighted_labels/weighted_sampling_evaluation_example.py`.

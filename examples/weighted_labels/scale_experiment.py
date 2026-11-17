"""Scale experiment: four three-level comparisons, 200k and 1M block pairs.

Measures (for both EM paths, 3 iterations):
  * wall-clock time per iteration,
  * Python extra peak via tracemalloc (measured from after input registration
    and shared term-frequency prep),
  * DuckDB temp disk usage and its own memory.

Pairwise data stays in DuckDB; only parameter-level aggregates come back.
"""
import argparse
import gc
import os
import shutil
import tempfile
import time
import tracemalloc

import duckdb
import numpy as np
import pandas as pd

from splink import DuckDBAPI, Linker, SettingsCreator, block_on
import splink.internals.comparison_library as cm


def _parse_size_to_bytes(text: str) -> int:
    units = {
        "bytes": 1,
        "KiB": 1024,
        "MiB": 1024 ** 2,
        "GiB": 1024 ** 3,
        "TiB": 1024 ** 4,
    }
    parts = str(text).strip().split()
    if len(parts) == 1:
        return int(float(parts[0]))
    value, unit = parts[0], parts[1]
    return int(float(value) * units.get(unit, 1))


def build_synthetic(n_records, seed=7, dominant_share=0.9, n_buckets=20):
    rng = np.random.default_rng(seed)
    # Skewed blocking column: a dominant bucket holding `dominant_share` of
    # records gives a single large, controllable training block while the
    # remaining buckets stay small (and keep term-frequency work realistic).
    n_dominant = int(round(n_records * dominant_share))
    first_name = np.concatenate(
        [
            np.zeros(n_dominant, dtype=np.int64),
            rng.integers(1, n_buckets, n_records - n_dominant),
        ]
    )
    rng.shuffle(first_name)
    surname = rng.integers(0, 200, n_records)
    city = rng.integers(0, 100, n_records)
    email = rng.integers(0, 400, n_records)
    phone = rng.integers(0, 300, n_records)
    df = pd.DataFrame(
        {
            "unique_id": np.arange(n_records),
            "first_name": first_name.astype(str),
            "surname": surname.astype(str),
            "city": city.astype(str),
            "email": email.astype(str),
            "phone": phone.astype(str),
        }
    )
    return df


def target_block_size(df, block_col, target_pairs):
    """Pick a threshold value whose self-join block is closest to target."""
    counts = df[block_col].value_counts()
    pairs = (counts * (counts - 1) // 2).sort_values(ascending=False)
    return pairs.index[0]


def run(target_pairs, tf_path, temp_dir):
    # A dominant first_name bucket holding 90% of records yields a self-join
    # block of roughly (0.9*n)^2/2 pairs.  Size n to land near the target.
    dominant_share = 0.9
    n_records = int(np.sqrt(2 * target_pairs) / dominant_share)
    n_records = min(max(n_records, 500), 120_000)
    df = build_synthetic(n_records, dominant_share=dominant_share)

    # find a first_name value giving a block near the target
    counts = df["first_name"].value_counts()
    pair_sizes = (counts * (counts - 1) // 2).sort_values(ascending=False)
    block_value = str(pair_sizes.index[0])
    actual_pairs = int(pair_sizes.iloc[0])

    con = duckdb.connect(database=os.path.join(temp_dir, "duckdb.duckdb"))
    con.execute(f"SET temp_directory='{temp_dir}/ddbtmp'")
    con.execute("SET memory_limit='2GB'")
    db_api = DuckDBAPI(connection=con)

    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=[
            # four three-level comparisons (null + 2 non-null levels each)
            cm.LevenshteinAtThresholds("surname", 1),
            cm.LevenshteinAtThresholds("city", 1),
            cm.LevenshteinAtThresholds("email", 1),
            cm.LevenshteinAtThresholds("phone", 1),
        ],
        blocking_rules_to_generate_predictions=[block_on("first_name")],
        max_iterations=3,
    )
    linker = Linker(df, settings, db_api=db_api)

    # build labels for 1% of block pairs
    block_df = df[df["first_name"] == block_value].reset_index(drop=True)
    ids = block_df["unique_id"].to_numpy()
    n_block = len(ids)
    all_pairs = [
        (int(ids[i]), int(ids[j]))
        for i in range(n_block)
        for j in range(i + 1, n_block)
    ]
    rng = np.random.default_rng(11)
    n_lab = max(1, int(round(actual_pairs * 0.01)))
    chosen = rng.choice(len(all_pairs), size=n_lab, replace=False)
    label_rows = []
    for k in chosen:
        a, b = all_pairs[k]
        score = float(rng.choice([0.0, 0.5, 1.0], p=[0.4, 0.2, 0.4]))
        label_rows.append(
            {
                "unique_id_l": a,
                "unique_id_r": b,
                "clerical_match_score": score,
            }
        )
    labels = pd.DataFrame(label_rows)
    linker.table_management.register_table(labels, "lab", overwrite=True)

    # shared prep: concat + term frequencies, exactly once
    from splink.internals.pipeline import CTEPipeline
    from splink.internals.vertically_concatenate import compute_df_concat_with_tf

    compute_df_concat_with_tf(linker, CTEPipeline())
    gc.collect()

    # measurement start point
    tracemalloc.start()
    t0 = time.perf_counter()
    session = linker.training.estimate_parameters_using_expectation_maximisation(
        f"l.first_name = r.first_name and l.first_name = '{block_value}'",
        estimate_without_term_frequencies=tf_path,
        fix_u_probabilities=False,
        pairwise_labels="lab",
    )
    wall = time.perf_counter() - t0
    _, py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # DuckDB's own memory and temp footprint.  PRAGMA database_size reports
    # memory_usage as a human-readable string ("12.4 MiB"); parse it to bytes.
    mem_text = con.execute("PRAGMA database_size").fetchone()[7]
    ddb_mem = _parse_size_to_bytes(mem_text)
    tmp_path = os.path.join(temp_dir, "ddbtmp")
    tmp_bytes = 0
    if os.path.isdir(tmp_path):
        for root, _, files in os.walk(tmp_path):
            for f in files:
                try:
                    tmp_bytes += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass

    summary = session.label_training_summary.as_dict()
    con.close()
    return {
        "target_pairs": target_pairs,
        "actual_block_pairs": actual_pairs,
        "tf_path": "no_tf" if tf_path else "with_tf",
        "wall_seconds_3_iter": round(wall, 3),
        "python_peak_mib": round(py_peak / (1024 ** 2), 3),
        "duckdb_memory_mib": round(ddb_mem / (1024 ** 2), 3),
        "duckdb_temp_mib": round(tmp_bytes / (1024 ** 2), 3),
        "summary": summary,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[200_000, 1_000_000])
    args = ap.parse_args()
    results = []
    for target in args.targets:
        for tf_path in (True, False):
            d = tempfile.mkdtemp(prefix=f"splinkscale_{target}_{tf_path}_")
            try:
                r = run(target, tf_path, d)
                results.append(r)
                print(r, flush=True)
            finally:
                shutil.rmtree(d, ignore_errors=True)
    out = pd.DataFrame(
        [
            {k: v for k, v in r.items() if k != "summary"}
            for r in results
        ]
    )
    print("\n=== SCALE RESULTS ===")
    print(out.to_string(index=False))
    # explicit memory-budget assertion for the Python side
    assert out["python_peak_mib"].max() < 128.0, out
    print("\nPython extra peak under 128 MiB in every run: OK")

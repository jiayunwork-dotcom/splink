"""Scale experiment for stratified pairwise sampling.

Generates 100k and 1M already-generated prediction pairs entirely inside
DuckDB (no all-pairs re-enumeration in Python), runs
``sample_pairwise_predictions`` with a fixed budget of 1000 and a fixed set
of score strata, and records:

  * SQL wall-clock time of the sampling call,
  * Python extra peak memory via tracemalloc,
  * DuckDB's own memory footprint and controlled temp-disk usage,
  * sample size, per-stratum inclusion probabilities and mass conservation.

Python memory is only allowed to grow with the number of strata and the
final sample: the full prediction population and its pair identities never
cross into Python.
"""
from __future__ import annotations

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

from splink import DuckDBAPI, Linker, SettingsCreator
from splink.internals.comparison_library import ExactMatch


def _parse_size_to_bytes(text: str) -> int:
    units = {
        "bytes": 1,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
        "TiB": 1024**4,
    }
    parts = str(text).strip().split()
    if len(parts) == 1:
        return int(float(parts[0]))
    value, unit = parts[0], parts[1]
    return int(float(value) * units.get(unit, 1))


def build_prediction_table(con, n_pairs: int, table_name: str, seed: int = 7):
    """Materialise n_pairs synthetic prediction rows inside DuckDB.

    Every row is an unoriented pair of distinct synthetic identities
    (``unique_id_r`` is always larger than ``unique_id_l``), so no self-pair
    or duplicated unoriented pair is present.  Scores are spread
    deterministically across four strata with at least one pair on every
    internal edge and at the closed endpoints 0 and 1.
    """
    rng = np.random.default_rng(seed)
    con.execute(
        f"""
        create table {table_name} as
        with r as (
            select
                i as unique_id_l,
                i + {n_pairs} as unique_id_r,
                case
                    when i % 10 = 0 then 0.25
                    when i % 10 = 1 then 0.5
                    when i % 10 = 2 then 0.75
                    when i % 10 = 3 then 0.0
                    when i % 10 = 4 then 1.0
                    else (i % 1000) / 1000.0
                end as match_probability,
                cast(0.0 as double) as match_weight
            from range({n_pairs}) t(i)
        )
        select
            unique_id_l, unique_id_r, match_probability,
            case when match_probability = 0.0 then cast(-1e9 as double)
                 when match_probability = 1.0 then cast(1e9 as double)
                 else log2(match_probability / (1 - match_probability))
            end as match_weight,
            cast(unique_id_l as varchar) as first_name_l,
            cast(unique_id_r as varchar) as first_name_r,
            0 as gamma_first_name
        from r
        """
    )
    # touch the rng to keep the explicit seed contract obvious
    _ = rng.integers(0, 10, size=1)
    return [0.0, 0.25, 0.5, 0.75, 1.0]


def build_linker(con, n_records: int):
    """A minimal dedupe linker whose concat table covers the synthetic ids.

    The sampling unknown-node check consults this concat table, so it must
    contain every identity referenced by the synthetic prediction rows.
    """
    ids = pd.DataFrame({"unique_id": range(2 * n_records)})
    ids["first_name"] = "a"
    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=[ExactMatch("first_name")],
        blocking_rules_to_generate_predictions=["1=1"],
        retain_matching_columns=True,
    )
    return Linker(ids, settings, db_api=DuckDBAPI(connection=con))


def run(n_pairs: int, temp_dir: str, budget: int = 1000):
    con = duckdb.connect(database=os.path.join(temp_dir, "duckdb.duckdb"))
    con.execute(f"SET temp_directory='{temp_dir}/ddbtmp'")
    con.execute("SET memory_limit='2GB'")
    db_api = DuckDBAPI(connection=con)

    table_name = "synthetic_predictions"
    bins = build_prediction_table(con, n_pairs, table_name)
    actual_n = con.sql(f"select count(*) from {table_name}").fetchone()[0]

    # the concat table only needs to contain the referenced identities; the
    # synthetic right-hand ids run from n_pairs to 2*n_pairs - 1
    linker = build_linker(con, n_records=n_pairs)

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    sample = linker.evaluation.sample_pairwise_predictions(
        table_name, budget=budget, score_bins=bins, seed=42
    )
    wall = time.perf_counter() - t0
    _, py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    out = sample.as_pandas_dataframe()
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

    # --- invariants ------------------------------------------------------- #
    assert len(out) == min(budget, actual_n)
    assert out["inclusion_probability"].notna().all()
    assert np.all((out["inclusion_probability"] > 0.0) &
                  (out["inclusion_probability"] <= 1.0))
    # inclusion probability matches n_h / N_h for every sampled row
    meta = (
        out.groupby("sample_stratum")
        .agg(
            n=("sample_stratum", "size"),
            pop=("stratum_population", "first"),
            size=("stratum_sample_size", "first"),
            pi=("inclusion_probability", "first"),
        )
        .reset_index()
    )
    assert np.allclose(meta["pi"], meta["size"] / meta["pop"])
    assert (meta["n"] == meta["size"]).all()
    # mass conservation (float64 tolerance): sum of 1/pi over the sample
    mass = (1.0 / out["inclusion_probability"].to_numpy(dtype=np.float64)).sum()
    assert np.isclose(mass, float(actual_n), rtol=1e-12, atol=1e-6), mass
    # no duplicate identities
    assert not out.duplicated(["unique_id_l", "unique_id_r"]).any()

    result = {
        "n_pairs": n_pairs,
        "budget": budget,
        "sample_size": len(out),
        "n_strata": int(meta.shape[0]),
        "wall_seconds": round(wall, 4),
        "python_peak_mib": round(py_peak / 1024**2, 4),
        "duckdb_memory_mib": round(ddb_mem / 1024**2, 4),
        "duckdb_temp_mib": round(tmp_bytes / 1024**2, 4),
        "estimated_mass": float(mass),
        "mass_error": float(abs(mass - actual_n)),
    }
    con.close()
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[100_000, 1_000_000])
    ap.add_argument("--budget", type=int, default=1000)
    args = ap.parse_args()

    rows = []
    for size in args.sizes:
        d = tempfile.mkdtemp(prefix=f"splinksample_{size}_")
        try:
            r = run(size, d, budget=args.budget)
            rows.append(r)
            print(r, flush=True)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    df = pd.DataFrame(rows)
    print("\n=== SAMPLING SCALE RESULTS ===")
    print(df.to_string(index=False))
    # the Python side must stay bounded by strata + sample bookkeeping
    assert df["python_peak_mib"].max() < 64.0, df
    print("\nPython extra peak under 64 MiB in every run: OK")
    print("All mass-conservation and probability invariants: OK")

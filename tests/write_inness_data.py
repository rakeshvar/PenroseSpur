"""Write per-mask inness histograms to SQLite databases.

For each symmetry and requested tile count N, all 1,400 masks are sampled in
one batch with rotation and translation disabled. Histogram values are raw
integer counts rounded to 0.1 inness bins, so every row sums exactly to N.

Usage: python tests/write_inness_data.py
"""

import sqlite3
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sampler import SpurSampler


NS = tuple(
    int(multiplier * 2**power)
    for power in (5, 6, 7)
    for multiplier in (1, 1.5)
)
SEED = 0
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "inness_data"
CONFIGS = {
    6: ("hex.sqlite3", 7.0),
    5: ("pen.sqlite3", 5.0),
}


def histogram_columns(max_inness):
    """Return SQLite-safe names for bins 0.0, 0.1, ..., max_inness."""
    return [f"bin_{index // 10}_{index % 10}" for index in range(round(max_inness * 10) + 1)]


def inness_histograms(inness, max_inness):
    """Count rounded 0.1 inness bins for each sample without normalization."""
    max_bin = round(max_inness * 10)
    bins = torch.round(inness * 10).long().clamp_(0, max_bin)
    counts = torch.zeros(
        bins.shape[0],
        max_bin + 1,
        dtype=torch.int64,
        device=bins.device,
    )
    counts.scatter_add_(1, bins, torch.ones_like(bins))
    return counts


def replace_table(connection, table_name, columns, rows, num_tiles):
    """Atomically replace one N table and validate its raw counts."""
    quoted_columns = ", ".join(f'"{column}" INTEGER NOT NULL' for column in columns)
    column_names = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join("?" for _ in range(2 + len(columns)))
    count_sum = " + ".join(f'"{column}"' for column in columns)

    with connection:
        connection.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        connection.execute(
            f'CREATE TABLE "{table_name}" ('
            "classid INTEGER NOT NULL, "
            "inclassid INTEGER NOT NULL, "
            f"{quoted_columns}, "
            "PRIMARY KEY (classid, inclassid), "
            f"CHECK ({count_sum} = {num_tiles})"
            ")"
        )
        connection.executemany(
            f'INSERT INTO "{table_name}" '
            f"(classid, inclassid, {column_names}) VALUES ({placeholders})",
            rows,
        )

        row_count = connection.execute(
            f'SELECT COUNT(*) FROM "{table_name}"'
        ).fetchone()[0]
        invalid_count = connection.execute(
            f'SELECT COUNT(*) FROM "{table_name}" WHERE {count_sum} != ?',
            (num_tiles,),
        ).fetchone()[0]
        if row_count != 1400:
            raise RuntimeError(
                f"{table_name}: expected 1400 rows, found {row_count}"
            )
        if invalid_count:
            raise RuntimeError(
                f"{table_name}: {invalid_count} histograms do not sum to {num_tiles}"
            )


def write_database(symmetry, database_name, max_inness):
    """Generate all requested N tables for one tiling symmetry."""
    database_path = OUTPUT_DIR / database_name
    columns = histogram_columns(max_inness)

    with sqlite3.connect(database_path) as connection:
        for num_tiles in NS:
            started = time.time()
            print(
                f"[symmetry {symmetry}] N={num_tiles}: loading sampler ...",
                flush=True,
            )
            sampler = SpurSampler(
                symmetry=symmetry,
                num_tiles=num_tiles,
                translation=0.0,
                rotation=0.0,
                seed=SEED,
            )
            mask_idx = torch.arange(len(sampler), device=sampler.device)
            generator = torch.Generator(device=sampler.device).manual_seed(SEED)

            print(
                f"[symmetry {symmetry}] N={num_tiles}: "
                f"sampling all {len(sampler)} masks at once ...",
                flush=True,
            )
            batch = sampler.sample_batch(
                len(sampler),
                mask_idx=mask_idx,
                generator=generator,
            )
            counts = inness_histograms(batch["inness"], max_inness).cpu()
            totals = counts.sum(dim=1)
            if not torch.all(totals == num_tiles):
                raise RuntimeError(
                    f"N={num_tiles}: histogram totals range from "
                    f"{totals.min().item()} to {totals.max().item()}"
                )

            labels = sampler.labels.cpu().tolist()
            inclass_ids = sampler.inclass_ids.cpu().tolist()
            rows = [
                (classid, inclassid, *histogram)
                for classid, inclassid, histogram in zip(
                    labels,
                    inclass_ids,
                    counts.tolist(),
                )
            ]
            replace_table(
                connection,
                f"n_{num_tiles}",
                columns,
                rows,
                num_tiles,
            )
            print(
                f"[symmetry {symmetry}] N={num_tiles}: wrote 1400 rows "
                f"in {time.time() - started:.1f}s",
                flush=True,
            )

    print(f"Saved {database_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for symmetry, (database_name, max_inness) in CONFIGS.items():
        write_database(symmetry, database_name, max_inness)


if __name__ == "__main__":
    main()

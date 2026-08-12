"""
Statistics-only check over a sweep of num_tiles (N).

For each symmetry and each N, generates 70 classes x 20 samples x 10 copies
= 14000 on-the-fly samples (every mask exactly 10 times) and reports coordinate,
inness and color statistics. Results are saved to tests/check_stats.md.

Usage: python tests/check_stats.py [copies]
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sampler import SpurSampler

NS = [64, 96, 128] #, 192, 256, 384, 512]
COPIES = 10
POINT_BUDGET = 6_000_000     # batch_size * M * (V+1) target, keeps memory bounded
FULL_TOLERANCE = 0.01
REPORT_PATH = Path(__file__).with_suffix(".md")


def run_config(symmetry, num_tiles, copies):
    sampler = SpurSampler(symmetry, num_tiles, 2.0, seed=0)
    K = len(sampler)
    V1 = sampler.V1
    total = K * copies

    order = torch.arange(K).repeat(copies)
    batch_size = max(1, min(256, POINT_BUDGET // (sampler.M * V1)))

    # Accumulators (float64 for numerical safety)
    sum_xy = torch.zeros(2, dtype=torch.float64, device=sampler.device)
    sum_xy2 = torch.zeros(2, dtype=torch.float64, device=sampler.device)
    inness_sum = 0.0
    min_inness_sum = 0.0
    full = 0
    color1 = 0
    n_tiles = 0

    t0 = time.time()
    for start in range(0, total, batch_size):
        idx = order[start:start + batch_size]
        b = sampler.sample_batch(len(idx), mask_idx=idx)
        xy = b["xya"][..., :2].double()
        sum_xy += xy.sum(dim=(0, 1))
        sum_xy2 += (xy ** 2).sum(dim=(0, 1))
        inness = b["inness"]
        inness_sum += inness.sum().item()
        min_inness_sum += inness.min(dim=1).values.sum().item()
        full += (inness >= V1 - FULL_TOLERANCE).sum().item()
        color1 += (b["colors"] == 1).sum().item()
        n_tiles += b["colors"].numel()
    elapsed = time.time() - t0

    mean = sum_xy / n_tiles
    std = (sum_xy2 / n_tiles - mean ** 2).sqrt()

    return {
        "N": num_tiles,
        "M": sampler.M,
        "side": sampler.side,
        "x_mean": mean[0].item(), "y_mean": mean[1].item(),
        "x_std": std[0].item(), "y_std": std[1].item(),
        "inness_mean": inness_sum / n_tiles,
        "pct_full": full / n_tiles * 100,
        "min_inness_mean": min_inness_sum / total,
        "pct_color1": color1 / n_tiles * 100,
        "sps": total / elapsed,
        "total": total,
        "elapsed": elapsed,
    }


def print_report(symmetry, rows):
    V1 = 7 if symmetry == 6 else 5
    name = "Hexagons" if symmetry == 6 else "Penrose rhombuses"
    color = "dark" if symmetry == 6 else "thin"
    print(f"\n{'=' * 96}")
    print(f"Symmetry {symmetry} ({name})   inness range 0..{V1}   "
          f"{rows[0]['total']} samples per N   color-1 = {color}")
    print(f"{'=' * 96}")
    header = (f"{'N':>4} {'M':>6} {'side':>6} {'x_mean':>8} {'y_mean':>8} "
              f"{'x_std':>7} {'y_std':>7} {'inness':>7} {'%full':>6} "
              f"{'min_in':>7} {'%col1':>6} {'samp/s':>8} {'time':>7}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['N']:>4} {r['M']:>6} {r['side']:>6.2f} "
              f"{r['x_mean']:>+8.3f} {r['y_mean']:>+8.3f} "
              f"{r['x_std']:>7.3f} {r['y_std']:>7.3f} "
              f"{r['inness_mean']:>7.2f} {r['pct_full']:>6.1f} "
              f"{r['min_inness_mean']:>7.2f} {r['pct_color1']:>6.1f} "
              f"{r['sps']:>8.0f} {r['elapsed']:>6.0f}s")
    print("-" * len(header))
    print(f"inness = mean tile inness; %full = within {FULL_TOLERANCE:g} "
          "of maximum inness; "
          "min_in = mean per-sample minimum inness")


def save_report(results, copies):
    lines = [
        "# Sampler statistics",
        "",
        f"Generated with `{copies}` copies of each of the 1,400 masks "
        f"({1400 * copies:,} samples per configuration).",
        "",
        f"`% full` means inness is within {FULL_TOLERANCE:g} of its maximum. "
        "`% color 1` is dark hexagons for symmetry 6 and thin rhombuses "
        "for symmetry 5.",
    ]
    for symmetry, rows in results.items():
        name = "Hexagons" if symmetry == 6 else "Penrose rhombuses"
        lines.extend([
            "",
            f"## Symmetry {symmetry}: {name}",
            "",
            "| N | M | side | x mean | y mean | x std | y std | "
            "inness | % full | min inness | % color 1 | samples/s | time (s) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            lines.append(
                f"| {row['N']} | {row['M']} | {row['side']:.4f} | "
                f"{row['x_mean']:+.4f} | {row['y_mean']:+.4f} | "
                f"{row['x_std']:.4f} | {row['y_std']:.4f} | "
                f"{row['inness_mean']:.4f} | {row['pct_full']:.2f} | "
                f"{row['min_inness_mean']:.4f} | {row['pct_color1']:.2f} | "
                f"{row['sps']:.1f} | {row['elapsed']:.1f} |"
            )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved {REPORT_PATH}")


def main(copies=COPIES):
    results = {}
    for symmetry in (6, 5):
        rows = []
        for n in NS:
            print(f"[symmetry {symmetry}] N={n} ...", flush=True)
            rows.append(run_config(symmetry, n, copies))
        print_report(symmetry, rows)
        results[symmetry] = rows
    save_report(results, copies)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else COPIES)

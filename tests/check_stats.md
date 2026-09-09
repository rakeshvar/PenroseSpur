# Sampler statistics

Run date: `2026-09-08`

Command: `~/.aivenv/bin/python tests/check_stats.py 10`

Generated with `10` copies of each of the 1,400 masks (14,000 samples per configuration), translation `2.0` polygon sides, canvas seed `0`, and `VAR_PER_AREA = 0.1463`. Penrose sampling generated a fresh De Bruijn grid for every `sample_batch` call.

Column key: `N` is returned tiles, `M` is canvas tiles in the final sampled batch, `side` is polygon side length in canvas units, `inness` is mean normalized soft-mask inness, `% full` means inness is within 0.01 of its maximum, `min inness` is the mean per-sample minimum, and `% color 1` is dark hexagons for symmetry 6 or thin rhombuses for symmetry 5.

## Symmetry 6: Hexagons

| N | M | side | x mean | y mean | x std | y std | inness | % full | min inness | % color 1 | samples/s | time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 847 | 0.2028 | +0.0000 | +0.0000 | 0.9639 | 1.0256 | 0.8511 | 51.44 | 0.4796 | 33.34 | 1975.9 | 7.1 |
| 96 | 1213 | 0.1655 | +0.0000 | +0.0000 | 0.9694 | 1.0336 | 0.8713 | 57.37 | 0.4814 | 33.32 | 1565.1 | 8.9 |
| 128 | 1573 | 0.1434 | -0.0000 | +0.0000 | 0.9764 | 1.0345 | 0.8839 | 61.24 | 0.4819 | 33.34 | 1245.1 | 11.2 |

## Symmetry 5: Penrose rhombuses

| N | M | side | x mean | y mean | x std | y std | inness | % full | min inness | % color 1 | samples/s | time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 977 | 0.3626 | -0.0000 | -0.0000 | 0.9603 | 1.0249 | 0.8398 | 47.22 | 0.4747 | 38.17 | 1994.0 | 7.0 |
| 96 | 1360 | 0.2961 | +0.0000 | -0.0000 | 0.9669 | 1.0327 | 0.8616 | 53.46 | 0.4778 | 38.18 | 1595.1 | 8.8 |
| 128 | 1730 | 0.2564 | +0.0000 | +0.0000 | 0.9734 | 1.0334 | 0.8748 | 57.49 | 0.4795 | 38.17 | 1539.4 | 9.1 |

## Findings

1. Per-sample centering remains effective: every aggregate `x mean` and `y mean` rounds to `0.0000`, with only signed floating-point residuals.
2. The `x std + y std` sums are 1.9895, 2.0030, and 2.0109 for hexagons and 1.9852, 1.9996, and 2.0068 for Penrose rhombuses at N=64, 96, and 128 respectively. The full range is 1.9852–2.0109, or -0.74% to +0.55% relative to 2.00, so all configurations satisfy the approximately 1% calibration target.
3. Calibration retains a small upward trend with N for both symmetries, crossing 2.00 near N=96. The drift remains within the target band, so no rescaling is indicated.
4. Color fractions meet their targets: hexagons are 33.32–33.34% dark (target 33.33%), and Penrose rhombuses are 38.17–38.18% thin (target 38.20%).
5. Mean per-sample minimum normalized inness remains well above zero (0.4796–0.4819 for hexagons and 0.4747–0.4795 for Penrose rhombuses), with no evidence of garbage tiles. `% full` rises monotonically with N for both symmetries.
6. No throughput regression is visible relative to the preceding report. Rates improved to 1,245–1,976 samples/s for hexagons and 1,539–1,994 samples/s for Penrose rhombuses, despite rebuilding a fresh Penrose canvas per batch. As these are single CPU runs, the magnitude of the improvement may include host/runtime variability.

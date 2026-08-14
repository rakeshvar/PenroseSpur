# Sampler statistics

Run date: `2026-08-14`

Command: `~/.aivenv/bin/python tests/check_stats.py 10`

Generated with `10` copies of each of the 1,400 masks (14,000 samples per configuration), translation `2.0` polygon sides, canvas seed `0`, and `VAR_PER_AREA = 0.1463`.

Column key: `N` is returned tiles, `M` is canvas tiles, `side` is polygon side length in canvas units, `inness` is mean normalized soft-mask inness, `% full` means inness is within 0.01 of its maximum, `min inness` is the mean per-sample minimum, and `% color 1` is dark hexagons for symmetry 6 or thin rhombuses for symmetry 5.

## Symmetry 6: Hexagons

| N | M | side | x mean | y mean | x std | y std | inness | % full | min inness | % color 1 | samples/s | time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 847 | 0.2028 | -0.0000 | +0.0000 | 0.9638 | 1.0256 | 0.8511 | 51.45 | 0.4804 | 33.31 | 1849.5 | 7.6 |
| 96 | 1213 | 0.1655 | +0.0000 | -0.0000 | 0.9710 | 1.0317 | 0.8713 | 57.41 | 0.4814 | 33.32 | 1506.2 | 9.3 |
| 128 | 1573 | 0.1434 | -0.0000 | -0.0000 | 0.9748 | 1.0357 | 0.8839 | 61.22 | 0.4819 | 33.32 | 1281.3 | 10.9 |

## Symmetry 5: Penrose rhombuses

| N | M | side | x mean | y mean | x std | y std | inness | % full | min inness | % color 1 | samples/s | time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 970 | 0.3626 | -0.0000 | -0.0000 | 0.9621 | 1.0282 | 0.8393 | 47.11 | 0.4724 | 37.87 | 2075.1 | 6.7 |
| 96 | 1354 | 0.2961 | +0.0000 | -0.0000 | 0.9718 | 1.0305 | 0.8612 | 53.36 | 0.4777 | 38.13 | 1688.2 | 8.3 |
| 128 | 1727 | 0.2564 | -0.0000 | +0.0000 | 0.9724 | 1.0359 | 0.8751 | 57.50 | 0.4830 | 38.37 | 1551.7 | 9.0 |

## Findings

1. Per-sample centering remains effective: every aggregate `x mean` and `y mean` rounds to `0.0000`, with only signed floating-point residuals.
2. The `x std + y std` sums are 1.9894, 2.0027, and 2.0105 for hexagons and 1.9903, 2.0023, and 2.0083 for Penrose rhombuses at N=64, 96, and 128 respectively. The full range is 1.9894–2.0105, or -0.53% to +0.53% relative to 2.00, so all configurations satisfy the approximately 1% calibration target.
3. Calibration retains a small upward trend with N for both symmetries, crossing 2.00 near N=96, but the full drift remains within the target band. No rescaling is indicated.
4. Color fractions meet their targets: hexagons are 33.31–33.32% dark (target 33.33%), and Penrose rhombuses are 37.87–38.37% thin (target 38.20%).
5. Mean per-sample minimum normalized inness remains well above zero (0.4804–0.4819 for hexagons and 0.4724–0.4830 for Penrose rhombuses), with no evidence of garbage tiles. `% full` rises monotonically with N for both symmetries.
6. Throughput declines with canvas size as expected, with no anomalous regression. Rates are 1,281–1,850 samples/s for hexagons and 1,552–2,075 samples/s for Penrose rhombuses.

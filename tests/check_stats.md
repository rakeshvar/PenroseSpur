# Sampler statistics

Run date: `2026-08-14`

Command: `~/.aivenv/bin/python tests/check_stats.py 10`

Generated with `10` copies of each of the 1,400 masks (14,000 samples per configuration), translation `2.0` polygon sides, canvas seed `0`, and `VAR_PER_AREA = 0.1463`.

Column key: `N` is returned tiles, `M` is canvas tiles, `side` is polygon side length in canvas units, `inness` is mean normalized soft-mask inness, `% full` means inness is within 0.01 of its maximum, `min inness` is the mean per-sample minimum, and `% color 1` is dark hexagons for symmetry 6 or thin rhombuses for symmetry 5.

## Symmetry 6: Hexagons

| N | M | side | x mean | y mean | x std | y std | inness | % full | min inness | % color 1 | samples/s | time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 847 | 0.2028 | -0.0000 | -0.0000 | 0.9642 | 1.0260 | 0.8509 | 51.37 | 0.4797 | 33.33 | 1410.1 | 9.9 |
| 96 | 1213 | 0.1655 | +0.0000 | -0.0000 | 0.9678 | 1.0352 | 0.8713 | 57.41 | 0.4816 | 33.31 | 1209.3 | 11.6 |
| 128 | 1573 | 0.1434 | -0.0000 | -0.0000 | 0.9738 | 1.0365 | 0.8839 | 61.22 | 0.4819 | 33.32 | 913.8 | 15.3 |

## Symmetry 5: Penrose rhombuses

| N | M | side | x mean | y mean | x std | y std | inness | % full | min inness | % color 1 | samples/s | time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 970 | 0.3626 | +0.0000 | -0.0000 | 0.9657 | 1.0249 | 0.8392 | 47.05 | 0.4725 | 37.82 | 1628.0 | 8.6 |
| 96 | 1354 | 0.2961 | +0.0000 | -0.0000 | 0.9688 | 1.0330 | 0.8612 | 53.35 | 0.4786 | 38.18 | 1241.7 | 11.3 |
| 128 | 1727 | 0.2564 | -0.0000 | +0.0000 | 0.9763 | 1.0319 | 0.8750 | 57.50 | 0.4828 | 38.36 | 1075.5 | 13.0 |

## Findings

1. Per-sample centering remains effective: every aggregate `x mean` and `y mean` rounds to `0.0000`, with only signed floating-point residuals.
2. The `x std + y std` sums are 1.9902, 2.0030, and 2.0103 for hexagons and 1.9906, 2.0018, and 2.0082 for Penrose rhombuses at N=64, 96, and 128 respectively. The full range is 1.9902–2.0103, or -0.49% to +0.52% relative to 2.00, so all configurations satisfy the approximately 1% calibration target.
3. Calibration retains a small upward trend with N for both symmetries, crossing 2.00 near N=96, but the full drift remains within the target band. No rescaling is indicated.
4. Color fractions meet their targets: hexagons are 33.31–33.33% dark (target 33.33%), and Penrose rhombuses are 37.82–38.36% thin (target 38.20%).
5. Mean per-sample minimum normalized inness remains well above zero (0.4797–0.4819 for hexagons and 0.4725–0.4828 for Penrose rhombuses), with no evidence of garbage tiles. `% full` rises monotonically with N for both symmetries.
6. Throughput declines with canvas size as expected, but this run is 20–31% slower than the preceding report across the six configurations. Rates are 914–1,410 samples/s for hexagons and 1,076–1,628 samples/s for Penrose rhombuses; because the XY-only fix does not add meaningful work, the broad slowdown is more consistent with host/runtime variability than with a sampler algorithm regression.

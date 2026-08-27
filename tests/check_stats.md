# Sampler statistics

Run date: `2026-08-27`

Command: `~/.aivenv/bin/python tests/check_stats.py`

Generated with `10` copies of each of the 1,400 masks (14,000 samples per configuration), translation `2.0` polygon sides, canvas seed `0`, and `VAR_PER_AREA = 0.1463`.

Column key: `N` is returned tiles, `M` is canvas tiles, `side` is polygon side length in canvas units, `inness` is mean normalized soft-mask inness, `% full` means inness is within 0.01 of its maximum, `min inness` is the mean per-sample minimum, and `% color 1` is dark hexagons for symmetry 6 or thin rhombuses for symmetry 5.

## Symmetry 6: Hexagons

| N | M | side | x mean | y mean | x std | y std | inness | % full | min inness | % color 1 | samples/s | time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 847 | 0.2028 | -0.0000 | -0.0000 | 0.9647 | 1.0247 | 0.8512 | 51.48 | 0.4795 | 33.34 | 759.3 | 18.4 |
| 96 | 1213 | 0.1655 | -0.0000 | +0.0000 | 0.9691 | 1.0337 | 0.8713 | 57.39 | 0.4821 | 33.35 | 1193.7 | 11.7 |
| 128 | 1573 | 0.1434 | +0.0000 | +0.0000 | 0.9722 | 1.0383 | 0.8839 | 61.25 | 0.4819 | 33.32 | 855.0 | 16.4 |

## Symmetry 5: Penrose rhombuses

| N | M | side | x mean | y mean | x std | y std | inness | % full | min inness | % color 1 | samples/s | time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 970 | 0.3626 | +0.0000 | -0.0000 | 0.9663 | 1.0251 | 0.8388 | 46.93 | 0.4716 | 37.76 | 370.5 | 37.8 |
| 96 | 1354 | 0.2961 | -0.0000 | -0.0000 | 0.9701 | 1.0318 | 0.8610 | 53.30 | 0.4785 | 38.17 | 579.6 | 24.2 |
| 128 | 1727 | 0.2564 | +0.0000 | +0.0000 | 0.9734 | 1.0349 | 0.8751 | 57.51 | 0.4825 | 38.36 | 479.6 | 29.2 |

## Findings

1. Per-sample centering remains effective: every aggregate `x mean` and `y mean` rounds to `0.0000`, with only signed floating-point residuals.
2. The `x std + y std` sums are 1.9894, 2.0028, and 2.0105 for hexagons and 1.9914, 2.0019, and 2.0083 for Penrose rhombuses at N=64, 96, and 128 respectively. The full range is 1.9894–2.0105, or -0.53% to +0.53% relative to 2.00, so all configurations satisfy the approximately 1% calibration target.
3. Calibration retains a small upward trend with N for both symmetries, crossing 2.00 near N=96, but the full drift remains within the target band. No rescaling is indicated.
4. Color fractions meet their targets: hexagons are 33.32–33.35% dark (target 33.33%), and Penrose rhombuses are 37.76–38.36% thin (target 38.20%).
5. Mean per-sample minimum normalized inness remains well above zero (0.4795–0.4821 for hexagons and 0.4716–0.4825 for Penrose rhombuses), with no evidence of garbage tiles. `% full` rises monotonically with N for both symmetries.
6. Throughput is substantially lower than the preceding report in five configurations and roughly unchanged for hexagons at N=96. Rates are 759–1,194 samples/s for hexagons and 371–580 samples/s for Penrose rhombuses; Penrose throughput is 53–77% lower than the preceding run, so the regression is anomalous, though a single CPU run cannot distinguish host/runtime variability from a code regression.

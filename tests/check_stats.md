# Sampler statistics

Run date: `2026-08-12`

Command: `~/.aivenv/bin/python tests/check_stats.py 10`

Generated with `10` copies of each of the 1,400 masks (14,000 samples per configuration), translation `2.0` polygon sides, canvas seed `0`, and `VAR_PER_AREA = 0.1463`.

Column key: `N` is returned tiles, `M` is canvas tiles, `side` is polygon side length in canvas units, `inness` is mean tile inness, `% full` means inness is within 0.01 of its maximum, `min inness` is the mean per-sample minimum, and `% color 1` is dark hexagons for symmetry 6 or thin rhombuses for symmetry 5.

## Symmetry 6: Hexagons

| N | M | side | x mean | y mean | x std | y std | inness | % full | min inness | % color 1 | samples/s | time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 847 | 0.2028 | -0.0000 | +0.0000 | 0.9617 | 1.0267 | 5.9573 | 50.83 | 3.3567 | 33.34 | 1554.8 | 9.0 |
| 96 | 1213 | 0.1655 | -0.0000 | +0.0000 | 0.9729 | 1.0298 | 6.0989 | 56.74 | 3.3702 | 33.32 | 1396.0 | 10.0 |
| 128 | 1573 | 0.1434 | +0.0000 | +0.0000 | 0.9745 | 1.0357 | 6.1878 | 60.55 | 3.3725 | 33.31 | 1109.0 | 12.6 |

## Symmetry 5: Penrose rhombuses

| N | M | side | x mean | y mean | x std | y std | inness | % full | min inness | % color 1 | samples/s | time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 970 | 0.3626 | -0.0000 | -0.0000 | 0.9609 | 1.0298 | 4.1944 | 46.59 | 2.3578 | 37.83 | 1785.3 | 7.8 |
| 96 | 1354 | 0.2961 | +0.0000 | -0.0000 | 0.9684 | 1.0333 | 4.3054 | 52.89 | 2.3887 | 38.18 | 1522.2 | 9.2 |
| 128 | 1727 | 0.2564 | -0.0000 | +0.0000 | 0.9736 | 1.0345 | 4.3754 | 57.00 | 2.4123 | 38.36 | 927.8 | 15.1 |

## Findings

1. Per-sample centering remains effective: every aggregate `x mean` and `y mean` rounds to `0.0000`, with only signed floating-point residuals.
2. The `x std + y std` sums are 1.9884, 2.0027, and 2.0102 for hexagons and 1.9907, 2.0017, and 2.0081 for Penrose rhombuses at N=64, 96, and 128 respectively. The full range is 1.9884–2.0102, or -0.58% to +0.51% relative to 2.00, so all configurations satisfy the approximately 1% calibration target.
3. Calibration has a small upward trend with N for both symmetries, crossing 2.00 near N=96, but the full drift remains within the target band. No further rescaling is indicated.
4. Color fractions meet their targets: hexagons are 33.31–33.34% dark (target 33.33%), and Penrose rhombuses are 37.83–38.36% thin (target 38.20%).
5. Mean per-sample minimum inness remains well above zero (3.36–3.37 for hexagons and 2.36–2.41 for Penrose rhombuses), with no evidence of garbage tiles. `% full` rises monotonically with N for both symmetries.
6. Throughput generally declines with canvas size as expected, but the Penrose N=128 rate of 927.8 samples/s is unusually low versus 1,422.0 samples/s in the preceding run. This isolated CPU timing does not affect the statistical calibration result.

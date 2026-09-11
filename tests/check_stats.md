# Sampler statistics

Run date: `2026-09-10`

Command: `~/.aivenv/bin/python tests/check_stats.py 10`

Generated with `10` copies of each of the 1,400 masks (14,000 samples per configuration), translation `2.0` polygon sides, canvas seed `0`, and `VAR_PER_AREA = 0.1463`. Penrose sampling generated a fresh De Bruijn grid for every `sample_batch` call.

Column key: `N` is returned tiles, `M` is canvas tiles in the final sampled batch, `side` is polygon side length in canvas units, `inness` is mean normalized soft-mask inness, `% full` means inness is within 0.01 of its maximum, `min inness` is the mean per-sample minimum, and `% color 1` is dark hexagons for symmetry 6 or thin rhombuses for symmetry 5.

## Symmetry 6: Hexagons

| N | M | side | x mean | y mean | x std | y std | inness | % full | min inness | % color 1 | samples/s | time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 847 | 0.2028 | -0.0000 | +0.0000 | 0.9649 | 1.0240 | 0.8511 | 51.46 | 0.4793 | 33.33 | 1225.1 | 11.4 |
| 96 | 1213 | 0.1655 | +0.0000 | +0.0000 | 0.9709 | 1.0318 | 0.8712 | 57.36 | 0.4815 | 33.34 | 1242.3 | 11.3 |
| 128 | 1573 | 0.1434 | +0.0000 | -0.0000 | 0.9746 | 1.0366 | 0.8840 | 61.23 | 0.4823 | 33.32 | 1055.9 | 13.3 |

## Symmetry 5: Penrose rhombuses

| N | M | side | x mean | y mean | x std | y std | inness | % full | min inness | % color 1 | samples/s | time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 977 | 0.3626 | +0.0000 | +0.0000 | 0.9591 | 1.0258 | 0.8403 | 47.25 | 0.4763 | 38.23 | 1650.4 | 8.5 |
| 96 | 1360 | 0.2961 | +0.0000 | -0.0000 | 0.9685 | 1.0303 | 0.8614 | 53.45 | 0.4780 | 38.17 | 1323.4 | 10.6 |
| 128 | 1730 | 0.2564 | -0.0000 | -0.0000 | 0.9753 | 1.0320 | 0.8746 | 57.48 | 0.4785 | 38.10 | 1171.4 | 12.0 |

## Findings

1. Per-sample centering remains effective: every aggregate `x mean` and `y mean` rounds to `0.0000`, with only signed floating-point residuals.
2. The `x std + y std` sums are 1.9889, 2.0027, and 2.0112 for hexagons and 1.9849, 1.9988, and 2.0073 for Penrose rhombuses at N=64, 96, and 128 respectively. The full range is 1.9849–2.0112, or -0.76% to +0.56% relative to 2.00, so all configurations satisfy the approximately 1% calibration target.
3. Calibration retains a small upward trend with N for both symmetries, crossing 2.00 near N=96. The drift remains within the target band, so no rescaling is indicated.
4. Color fractions meet their targets: hexagons are 33.32–33.34% dark (target 33.33%), and Penrose rhombuses are 38.10–38.23% thin (target 38.20%).
5. Mean per-sample minimum normalized inness remains well above zero (0.4793–0.4823 for hexagons and 0.4763–0.4785 for Penrose rhombuses), with no evidence of garbage tiles. `% full` rises monotonically with N for both symmetries.
6. Throughput was 1,056–1,242 samples/s for hexagons and 1,171–1,650 samples/s for Penrose rhombuses. Most configurations improved over the immediately preceding report; hexagon N=64 was about 5% slower, consistent with normal single-run CPU variability rather than a concerning regression.

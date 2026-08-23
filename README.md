# PenroseSpur

On-the-fly GPU tile sampling from MPEG7 masks. Instead of pre-generating
datasets (as `PenroseDiffusion/scripts/create_dataset.py` does), PenroseSpur
loads two precomputed tensors — the normalized masks and a mother canvas — and
generates every training batch on the device.

## How it works

1. **Masks** (`masks.py`, built once): all 1400 MPEG7 masks are binarized,
   cropped, rescaled so every mask has the same number of ON pixels
   (density-normalized, target = median ON count), and center-padded to common
   max dimensions. Result: a single `(1400, 613, 846)` uint8 tensor.
2. **Mother canvas** (`canvas.py`, built once per symmetry/N/T): a patch of M
   tiles covering a disk of radius `hypot(H/2, W/2)·scaling + T·unit_side` —
   large enough for any mask under any rotation plus a translation jitter of
   T polygon sides.
   - Symmetry 6: hexagonal grid (formulas from `code/polygons/hex/qrs.py`;
     color 1 = "dark", exactly 1/3 of tiles).
   - Symmetry 5: Penrose P3 rhombuses via the **de Bruijn pentagrid**
     construction (ported from `Notes/Sklar/elephant/generate_random_elephant.py`)
     with random grid offsets `gamma` — every build (seed) is a distinct
     Penrose patch. Angle/color conventions match
     `PenroseDiffusion/code/polygons/pen/xya.py` (color 1 = thin, fraction
     psi^2 = 0.382).
3. **Sampler** (`sampler.py`, every batch, fully vectorized on the device):
   rotate + translate all M tiles (`SpurSampler.transform_and_inness`), calculate
   each tile's soft inness 0..1 from its center + V vertices, record whether
   each probe exceeds 0.5, break ties with
   U(0,1) noise, and keep the best `num_tiles` via
   `torch.topk`.

## Calibration

`unit_side = sqrt(C / num_tiles)` with C = 2.37 (hex) / 7.47 (pen), calibrated
empirically so that **x_std + y_std = 2** for the chosen tiles (verified for
N = 64..512, see `tests/check_stats.md`). y_std runs slightly above x_std
because the MPEG7 masks are wider than tall; the sum, not each axis, is the
invariant. If the mask preprocessing, `target_on`, or the translation range
changes, re-measure with `tests/check_stats.py` and rescale `C' = C·(2/sum)²`
(the `check-stats` subagent in `.cursor/agents/` automates this check).

## Usage

```bash
python masks.py                 # inspect in-memory masks and write diagnostics
python canvas.py 6 96 448 448 31941.5 2.0
python sampler.py               # smoke test both symmetries
```

```python
from sampler import SpurSampler

sampler = SpurSampler(
    symmetry=5,
    num_tiles=96,
    translation_canvas=2.0,
    seed=0,
)
batch = sampler.sample_batch(64)
batch["xya"]      # (64, 96, 3): zero-mean x/y per sample, unit-variance scaled angle
batch["colors"]   # (64, 96)  hex: dark/light, pen: 1 = thin rhombus
batch["labels"]   # (64,)     MPEG7 class ids (70 classes)
batch["indices"]  # (64, 96)  mother-canvas tile ids
batch["inness"]   # (64, 96)  soft tile inness at selection time
batch["vertex_in"] # (64, 96, V+1) thresholded center/vertex mask probes

noise = sampler.sample_noise(64)  # N(0, I) positions
```

Pass `mask_idx` to `sample_batch` for class-conditioned sampling, and
`return_vertices=True` to also get the polygon vertices `(B, N, V, 2)` for
rendering. `transform_and_inness(mask_idx)` exposes the per-tile soft inness and
Boolean `vertex_in` probes of all M mother tiles for one rotation/translation
draw.

After tile selection and augmentation, each returned sample is translated so
its tile-center mean is exactly `(0, 0)`. Returned vertices receive the same
translation.

Angles returned by both `sample_batch` and `sample_noise` have unit variance:
radian angles are wrapped to `[-pi, pi)` and scaled by `sqrt(3)/pi`, so noise
angles are uniform on `[-sqrt(3), sqrt(3)]`.

`sample_noise` uses unit-scale Gaussian XY coordinates, `N(0, I)`, with no
distribution or radius override.

## Lattice loss

`lattice_loss.py` provides a color-aware nearest-neighbour metric for batched
XY or XYA tensors:

```python
from lattice_loss import lattice_loss

loss = lattice_loss(symmetry, side, xya, colors)
```

For every tile, the loss finds its nearest centre and assigns the exact target
distance from the pair's colors. Hexagons always target `sqrt(3) * side`.
Penrose targets are `sin(2*pi/5) * side` for thick/thick (`0/0`),
`sin(pi/5) * side` for thin/thin (`1/1`), and `sin(3*pi/10) * side` for a
mixed pair.

The default `multiplicative` error uses `r = d/d*`, `epsilon = 1/side`, and
`max(r, (1 + epsilon)/(r + epsilon)) - 1`. Set `algo="quadratic"` for
`MSE(d, d*)`, or `algo="logarithmic"` for the Itakura-Saito form
`r - log(r) - 1`. All modes return the mean over tiles and batches.

## Matching

`match.py` matches noise rows to sampled data with exact LSA or GPU-native
Sinkhorn outputs:

```python
from match import match

noise = sampler.sample_noise(64)
matched = match(batch["xya"], noise, method="lsa")
matched_argmax = match(
    batch["xya"], noise, method="argmax", epsilon=0.03, iterations=7
)
barycenters = match(
    batch["xya"], noise, method="barycenter", epsilon=0.03, iterations=7
)
```

`lsa` returns a true permutation. `argmax` can reuse the same noise row, and
`barycenter` returns a row-normalized soft weighted average rather than a
permutation. Sinkhorn defaults to epsilon `0.03` and 10 iterations.

## Sanity checks and reports

```bash
python tests/show_masks.py            # grid of normalized masks -> tests/output/
python tests/show_samples.py          # rendered samples + std/inness/color stats
python tests/show_canvas.py           # mother canvas + extremal masks (spread, x/y-range)
python tests/show_inness.py butterfly # one mask on the canvas with per-tile inness
python tests/check_stats.py [copies]  # full stats sweep over N; findings in tests/check_stats.md
```

## Files

- `masks.py` — builds the mask tensor in memory
- `canvas.py` — builds mother canvas tensors in memory
- `sampler.py` — `SpurSampler`, the on-the-fly batch generator
- `lattice_loss.py` — color-aware nearest-neighbour lattice metrics
- `match.py` — exact LSA and Sinkhorn-based noise matching
- `tests/show_masks.py`, `tests/show_samples.py`, `tests/show_canvas.py`,
  `tests/show_inness.py` — visual sanity checks (output in `tests/output/`)
- `tests/check_stats.py`, `tests/check_stats.md` — statistics sweep and findings
- `requirements.txt` — numpy, pillow, scipy, torch, matplotlib, plotly

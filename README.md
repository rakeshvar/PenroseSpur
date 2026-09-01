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

Set `num_cool_classes` to sample only from a prefix of the 30-class preference
list in `cool_classes.py`:

```python
sampler = SpurSampler(5, 96, num_cool_classes=10)
sampler.num_cool_classes  # 10
sampler.cool_class_ids    # exact resolved MPEG7 IDs, in preference order
```

The default `num_cool_classes=None` uses all 70 classes. Filtered batches retain
the original MPEG7 label IDs rather than remapping them. Slim-project configs
should save both `spur.num_cool_classes` and `spur.cool_class_ids`; pass both
back to `SpurSampler` when restoring so a later change to the global preference
list cannot change an existing run's mask pool. Explicit `cool_class_ids` are
authoritative, and `num_cool_classes` must match their length.

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

`lattice_loss.py` provides position, orientation, and combined metrics for
batched scaled XYA tensors:

```python
from lattice_loss import lattice_loss, lattice_loss_angle, lattice_loss_xy

xy_loss = lattice_loss_xy(symmetry, side, xya, colors)
angle_loss = lattice_loss_angle(symmetry, xya)
loss = lattice_loss(symmetry, side, xya, colors)  # xy_loss + angle_loss
```

For every tile, the XY loss finds its nearest centre and assigns the exact
target distance from the pair's colors. Hexagons always target
`sqrt(3) * side`. Penrose targets are `sin(2*pi/5) * side` for thick/thick
(`0/0`), `sin(pi/5) * side` for thin/thin (`1/1`), and
`sin(3*pi/10) * side` for a mixed pair.

The default `multiplicative` error uses `r = d/d*`, `epsilon = 1/side`, and
`max(r, (1 + epsilon)/(r + epsilon)) - 1`. Set `algo="quadratic"` for
`MSE(d, d*)`, or `algo="logarithmic"` for the Itakura-Saito form
`r - log(r) - 1`.

The angle loss is `0.1 * (1 - |mean(exp(i*k*theta))|^2)`, evaluated
independently per sample and then averaged over the batch. The `0.1` scale
makes it comparable in magnitude to the multiplicative XY loss. It uses `k=6`
for hexagons and `k=10` for Penrose rhombuses, giving zero for the corresponding
orientation coset and approaching `0.1` as its harmonic phases disperse.
`theta` is recovered from the stored `sqrt(3)/pi`-scaled angle channel.

## Matching

`match.py` matches noise rows to sampled data with exact LSA or GPU-native
Sinkhorn outputs:

```python
import torch

from match import match

generator = torch.Generator(device=sampler.device).manual_seed(0)
noise = sampler.sample_noise(64)
matched = match(
    batch["xya"],
    noise,
    method="lsa",
    colors=batch["colors"],
    generator=generator,
)
matched_argmax = match(
    batch["xya"], noise, method="sinkhorn.argmax", epsilon=0.03, iterations=7
)
barycenters = match(
    batch["xya"], noise, method="sinkhorn.barycenter", epsilon=0.03, iterations=7
)
```

By default, `lsa` independently shuffles each color, balances it into groups
targeting 64 tiles and capped at 80, and solves those groups in parallel. Set
`lsa_target_size`, `lsa_max_size`, and `lsa_workers` to override those defaults.
Groups differ by at most one tile and avoid sizes at or below 40 whenever a
valid partition allows it. Passing `generator` makes grouping reproducible.
LSA returns a true same-color permutation. `sinkhorn.argmax` can reuse the same
noise row, and `sinkhorn.barycenter` returns a row-normalized soft weighted
average rather than a permutation. Sinkhorn defaults to epsilon `0.03` and 10
iterations.

## SVG and MP4 rendering

The `show/` package is the shared renderer for PenroseSpur and its consumers.
It is intentionally separate from sampling, matching, losses, and tile
mathematics. Import stable APIs from the facade:

```python
from show import (
    AssignmentStyle,
    ComparisonLayer,
    LayerStyle,
    VideoOptions,
    save_mp4,
    save_scene_svg,
    save_tiles_svg,
    xya_shared_viewbox,
)
```

Render raw-radian XYA values or a batched Spur sample. Spur stores angles scaled
by `ANGLE_SCALE = sqrt(3) / pi`, so pass that value explicitly:

The XY channels retain the sampler's mask-aligned canvas convention: the first
coordinate maps to image row and the second to image column. SVG presentation
therefore maps canvas `(x, y)` to SVG `(y, x)`, matching source-mask orientation
and legacy PenroseDiffusion output. Rendering does not reorder the stored model
tensor or change its angle convention.

```python
save_tiles_svg(
    "sample.svg",
    batch["xya"],
    batch["colors"],
    symmetry=sampler.symmetry,
    side=sampler.side,
    angle_scale=ANGLE_SCALE,
    scheme="ocean",
    show_arcs=sampler.symmetry == 5,
    opacities=batch["inness"][0],
)
```

`save_tiles_svg` also accepts unbatched `(N,3)` XYA and `(N,)` colors. For
already constructed `(N,4,2)` Penrose or `(N,6,2)` hex polygons, use
`save_polygons`. Both routes support a named scheme, custom `ColorScheme`,
background/stroke overrides, alpha, per-tile opacity, radius and duplicate
markers, fixed view boxes, and standalone SVG strings. Every built-in scheme
renders tile fills at `alpha=0.7` by default; pass an explicit `alpha` to
override it.

Build error or assignment scenes from any number of fill/outline layers:

```python
layers = [
    ComparisonLayer(target, colors, "target", LayerStyle("fill_outline")),
    ComparisonLayer(
        prediction,
        colors,
        "prediction",
        LayerStyle("outline", color_role="aux"),
    ),
]
save_scene_svg(
    "error.svg",
    layers,
    symmetry=6,
    side=side,
    assignments=[(1, 0, None, AssignmentStyle("error", arrows=True))],
    metrics={"mean displacement": displacement},
)
```

For stable animation frames, calculate one view box across the trajectory and
reuse it in every SVG. `save_mp4` normalizes the canvases, preserves
non-scaling stroke appearance, rasterizes ordered SVG frames, and writes H.264
with `yuv420p` and `faststart`:

```python
viewbox = xya_shared_viewbox(
    trajectory, colors, symmetry, side, angle_scale=ANGLE_SCALE
)
for index, state in enumerate(trajectory):
    save_tiles_svg(
        f"frames/frame_{index:04d}.svg",
        state,
        colors,
        symmetry=symmetry,
        side=side,
        angle_scale=ANGLE_SCALE,
        viewbox=viewbox,
    )
save_mp4(
    sorted(Path("frames").glob("frame_*.svg")),
    "trajectory.mp4",
    options=VideoOptions(fps=30),
    viewbox=viewbox,
)
```

To turn unevenly spaced trajectory keyframes into a smooth, fixed-duration
video, use `save_trajectory_mp4`:

```python
save_trajectory_mp4(
    xya_trajectory,
    colors,
    "smooth.mp4",
    kind="xya",
    symmetry=symmetry,
    side=side,
    angle_scale=ANGLE_SCALE,
    target_duration=10.0,
)

save_trajectory_mp4(
    polygon_trajectory,
    colors,
    "smooth-polygons.mp4",
    kind="polygons",
    target_duration=10.0,
)
```

The default rate is 30 FPS. XYA timing uses the mean per-tile movement
`hypot(dx, dy) + abs(wrapped_angle_delta)` and interpolates angles along the
shortest wrapped path. Polygon timing uses RMS vertex displacement and
interpolates corresponding vertices directly. Cumulative movement determines
how many output frames each source interval receives, so distant keyframes
take proportionally longer than nearby ones. Per-keyframe opacity arrays are
interpolated on the same schedule. Without `target_duration`, each trajectory
keyframe remains one output video frame for compatibility.

MP4 creation requires `ffmpeg` on `PATH`; SVG rendering requires only NumPy and
the standard library. Available styles are listed by `scheme_names(5)` and
`scheme_names(6)`. Render all twelve built-ins with:

```bash
~/.aivenv/bin/python tests/show_gallery.py
```

Generated SVG/MP4/HTML files remain local artifacts and are not logged to
WandB.

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
- `cool_classes.py` — ordered preferred MPEG7 class IDs and names
- `canvas.py` — builds mother canvas tensors in memory
- `sampler.py` — `SpurSampler`, the on-the-fly batch generator
- `lattice_loss.py` — nearest-neighbour XY, harmonic angle, and total metrics
- `match.py` — exact LSA and Sinkhorn-based noise matching
- `show/` — color schemes, SVG scenes, shared view boxes, and MP4 encoding
- `tests/show_masks.py`, `tests/show_samples.py`, `tests/show_canvas.py`,
  `tests/show_inness.py` — visual sanity checks (output in `tests/output/`)
- `tests/test_show_svg.py`, `tests/test_show_video.py`, `tests/show_gallery.py`
  — rendering verification and the twelve-scheme gallery
- `tests/check_stats.py`, `tests/check_stats.md` — statistics sweep and findings
- `requirements.txt` — numpy, pillow, scipy, torch, matplotlib, plotly

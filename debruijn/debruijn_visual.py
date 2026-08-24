"""Visualize the de Bruijn pentagrid construction.

The geometry follows ``canvas.pen_grid_debruijn``: all ten unordered grid
family pairs and all line-index pairs are evaluated in broadcasted NumPy
arrays.  The plotting data is deliberately kept explicit so this file is easy
to use as a visual scratchpad.

Usage:
    python debruijn/debruijn_visual.py [kmax] [dirs_to_show] [--seed SEED]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from convert import pen_vertices


FAMILY_COLORS = ("#c026d3", "#dc2626", "#16a34a", "#2563eb", "#ca8a04")
TILE_COLORS = ("#4c78a8", "#f28e2b")  # thick, thin
DUAL_MAP_SCALE = 5 / 2
OUTPUT_DIR = Path(__file__).resolve().parent


def debruijn_grid(kmax, γ=None, rng=None):
    """Build plot-friendly pentagrid and dual-rhombus arrays without pair loops."""
    if kmax < 1:
        raise ValueError("kmax must be at least 1")
    if rng is None:
        rng = np.random.default_rng()
    if γ is None:
        γ = rng.uniform(0, 1, size=5)
    γ = np.asarray(γ, dtype=np.float64)
    if γ.shape != (5,):
        raise ValueError(f"γ must have shape (5,), got {γ.shape}")

    θ = 2 * np.pi * np.arange(5) / 5
    U = np.stack([np.cos(θ), np.sin(θ)], axis=1)  # (5, 2)
    K = np.arange(-kmax, kmax + 1)  # (L = 2kmax + 1,)

    # All 10 unordered family pairs crossed with all line-index pairs.
    I, J = np.triu_indices(5, k=1)  # (10,), (10,)
    KI, KJ = np.meshgrid(K, K, indexing="ij")
    KI = KI.ravel()[None, :]  # (1, L²)
    KJ = KJ.ravel()[None, :]

    determinant = (
        U[I, 0] * U[J, 1] - U[I, 1] * U[J, 0]
    )[:, None]  # (10, 1)
    CI = KI - γ[I, None]
    CJ = KJ - γ[J, None]
    Xx = (CI * U[J, 1, None] - CJ * U[I, 1, None]) / determinant
    Yx = (U[I, 0, None] * CJ - U[J, 0, None] * CI) / determinant

    # Match canvas.py's radial pentagrid crop.
    if False:
        keep = np.hypot(Xx, Yx) <= kmax
        intersections = np.stack([Xx[keep], Yx[keep]], axis=1)
        I = np.broadcast_to(I[:, None], keep.shape)[keep]
        J = np.broadcast_to(J[:, None], keep.shape)[keep]
        KI = np.broadcast_to(KI, keep.shape)[keep]
        KJ = np.broadcast_to(KJ, keep.shape)[keep]
    else:
        keep = np.ones_like(Xx, dtype=bool)
        intersections = np.stack([Xx.ravel(), Yx.ravel()], axis=1)
        I = np.broadcast_to(I[:, None], keep.shape).ravel()
        J = np.broadcast_to(J[:, None], keep.shape).ravel()
        KI = np.broadcast_to(KI, keep.shape).ravel()
        KJ = np.broadcast_to(KJ, keep.shape).ravel()

    is_thin = ((J - I == 2) | (J - I == 3))

    # Map each pentagrid intersection to its integer-valued 5D region.
    A5D = np.ceil(intersections @ U.T + γ[None, :])
    # Just in case ciel rounded 3+ε to 4 (say)
    family = np.arange(5)[None, :]
    A5D = np.where(family == I[:, None], KI[:, None], A5D)
    A5D = np.where(family == J[:, None], KJ[:, None], A5D)

    # Project and construct centers/vertices exactly as canvas.py does.
    F = A5D @ U
    AC = U[I] + U[J]
    centers = F + AC / 2
    arc_parity = (A5D.sum(axis=1).astype(np.int64) % 2).astype(bool)
    θ1 = np.arctan2(AC[:, 1], AC[:, 0])
    θ2 = np.arctan2(-AC[:, 1], -AC[:, 0])
    angles = np.where(arc_parity, θ1, θ2)
    colors = is_thin.astype(np.uint8)
    degree = np.max(np.abs(A5D), axis=1).astype(np.int64)
    rank = np.abs(np.min(A5D, axis=1)).astype(np.int64)
    centers -= np.mean(centers, axis=0)
    vertices = pen_vertices(centers, angles, colors, side=1)

    return {
        "γ": γ,
        "U": U,
        "line_indices": K,
        "intersections": intersections,
        "family_i": I,
        "family_j": J,
        "line_i": KI,
        "line_j": KJ,
        "regions_5d": A5D,
        "arc_parity": arc_parity,
        "is_thin": is_thin,
        "centers": centers,
        "angles": angles,
        "colors": colors,
        "degree": degree,
        "rank": rank,
        "vertices": vertices,
        "candidate_count": keep.size,
        "keep_mask": keep,
    }


def normalize_dirs_to_show(dirs_to_show):
    """Normalize an integer count, digit string, or iterable of family IDs."""
    if isinstance(dirs_to_show, (int, np.integer)):
        return tuple(range(int(dirs_to_show)))
    if isinstance(dirs_to_show, str):
        if not dirs_to_show.isdigit():
            raise ValueError("dirs_to_show must contain only digits")
        if len(dirs_to_show) == 1:
            return tuple(range(int(dirs_to_show)))
        return tuple(map(int, dirs_to_show))
    return tuple(dirs_to_show)


def selected_data(data, dirs_to_show):
    """Return the tile mask and pentagrid segments used by both renderers."""
    dirs_to_show = normalize_dirs_to_show(dirs_to_show)
    if not dirs_to_show:
        raise ValueError("dirs_to_show must contain at least one family")
    if len(set(dirs_to_show)) != len(dirs_to_show):
        raise ValueError("dirs_to_show must not contain duplicates")
    if any(direction not in range(5) for direction in dirs_to_show):
        raise ValueError("dirs_to_show values must be between 0 and 4")

    selected = (
        np.isin(data["family_i"], dirs_to_show)
        & np.isin(data["family_j"], dirs_to_show)
    )

    U = data["U"][list(dirs_to_show)]
    line_indices = data["line_indices"]
    normals = np.repeat(U, len(line_indices), axis=0)
    offsets = (
        np.tile(line_indices, len(dirs_to_show))
        - np.repeat(data["γ"][list(dirs_to_show)], len(line_indices))
    )
    tangents = np.stack([-normals[:, 1], normals[:, 0]], axis=1)
    extent = 2.2 * data["line_indices"][-1]
    grid_segments = np.stack(
        [offsets[:, None] * normals - extent * tangents,
         offsets[:, None] * normals + extent * tangents],
        axis=1,
    )
    grid_families = np.repeat(dirs_to_show, len(line_indices))
    return selected, grid_segments, grid_families


def _paths_with_breaks(paths):
    """Convert (N, P, 2) paths to Plotly x/y arrays separated by None."""
    if len(paths) == 0:
        return np.array([]), np.array([])
    separated = np.full((len(paths), len(paths[0]) + 1, 2), np.nan)
    separated[:, :-1] = paths
    flattened = separated.reshape(-1, 2)
    return flattened[:, 0], flattened[:, 1]


def _threshold_alpha(metric, kmax):
    """Map the selected metric's kmax cutoff to low/high opacity."""
    return np.where(metric > kmax, 0.15, 0.75)


def build_figure(
    data, selected, grid_segments, grid_families, kmax, alpha_by,
):
    """Build the single Plotly figure used for both HTML and SVG output."""
    if alpha_by not in ("degree", "rank"):
        raise ValueError("alpha_by must be 'degree' or 'rank'")
    fig = go.Figure()

    for family in np.unique(grid_families):
        color = FAMILY_COLORS[family]
        x, y = _paths_with_breaks(grid_segments[grid_families == family])
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=f"Family θ{'⁰¹²³⁴'[family]}",
            line={"color": color, "width": 1},
            opacity=0.22,
            hoverinfo="skip",
        ))

    intersections = data["intersections"][selected]
    # The dual map expands distances asymptotically by 5/2. Undo that expansion
    # for an overlay in the pentagrid's coordinate system.
    vertices = data["vertices"][selected] / DUAL_MAP_SCALE
    thin = data["is_thin"][selected]
    family_i = data["family_i"][selected]
    family_j = data["family_j"][selected]
    line_i = data["line_i"][selected]
    line_j = data["line_j"][selected]
    regions = data["regions_5d"][selected].astype(np.int64)
    degree = data["degree"][selected]
    rank = data["rank"][selected]
    alpha_metric = data[alpha_by][selected]
    alpha = _threshold_alpha(alpha_metric, kmax)

    closed_vertices = np.concatenate([vertices, vertices[:, :1]], axis=1)
    for tile_type, label, color in (
        (False, "Fat", TILE_COLORS[0]),
        (True, "Thin", TILE_COLORS[1]),
    ):
        type_mask = thin == tile_type
        metric_values = np.unique(alpha_metric[type_mask])
        for metric_index, metric_value in enumerate(metric_values):
            metric_mask = type_mask & (alpha_metric == metric_value)
            x, y = _paths_with_breaks(closed_vertices[metric_mask])
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=label,
                legendgroup=label,
                showlegend=metric_index == 0,
                line={"color": color, "width": 1},
                opacity=float(_threshold_alpha(metric_value, kmax)),
                hoverinfo="skip",
            ))

        customdata = np.column_stack([
            family_i[type_mask],
            family_j[type_mask],
            line_i[type_mask],
            line_j[type_mask],
            regions[type_mask],
            degree[type_mask],
            rank[type_mask],
        ])
        fig.add_trace(go.Scatter(
            x=intersections[type_mask, 0],
            y=intersections[type_mask, 1],
            mode="markers",
            name=f"{label}",
            marker={
                "color": color,
                "size": 5,
                "opacity": alpha[type_mask],
            },
            customdata=customdata,
            hovertemplate=(
                "intersection=(%{x:.3f}, %{y:.3f})"
                "<br>families=(%{customdata[0]:.0f}, %{customdata[1]:.0f})"
                "<br>lines=(%{customdata[2]:.0f}, %{customdata[3]:.0f})"
                "<br>region=[%{customdata[4]:.0f}, %{customdata[5]:.0f}, "
                "%{customdata[6]:.0f}, %{customdata[7]:.0f}, "
                "%{customdata[8]:.0f}]"
                "<br>degree=%{customdata[9]:.0f}"
                "<br>rank=%{customdata[10]:.0f}<extra></extra>"
            ),
        ))

    limit = np.max(np.abs(degree))+1
    fig.update_layout(
        title=(
            f"De Bruijn Pentagrid and Dual Rhombi, k=-({kmax}...{kmax}), "
            f"alpha by {alpha_by}"
        ),
        template="plotly_white",
        width=900,
        height=900,
        hovermode="closest",
        xaxis={"range": [-limit, limit], "showgrid": False, "zeroline": False},
        yaxis={
            "range": [-limit, limit],
            "showgrid": False,
            "zeroline": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
    )
    return fig


def print_rhombus_metrics(data, metric_name):
    """Print cumulative fat/thin frequencies for one region metric."""
    metric = data[metric_name]
    kmax = data["line_indices"].max()
    is_thin = data["is_thin"]
    inverse_phi = (np.sqrt(5) - 1) / 2
    inverse_phi_squared = inverse_phi ** 2

    print(
        f"\nCumulative rhombus frequencies by {metric_name} "
        f"(expected fat=1/φ={inverse_phi:.4%}, "
        f"thin=1/φ²={inverse_phi_squared:.4%})"
    )
    print(
        f"{metric_name:>8} {'total':>8} {'fat':>8} {'thin':>8} "
        f"{'fat %':>9} {'thin %':>9} {'fat err':>9} {'thin err':>9}"
    )
    for cutoff in np.unique(metric):
        keep = metric <= cutoff
        total = int(keep.sum())
        thin = int(is_thin[keep].sum())
        fat = total - thin
        fat_fraction = fat / total
        thin_fraction = thin / total
        print(
            f"{cutoff:>8d} {total:>8d} {fat:>8d} {thin:>8d} "
            f"{fat_fraction:>8.2%} {thin_fraction:>8.2%} "
            f"{abs(fat_fraction - inverse_phi):>8.2%} "
            f"{abs(thin_fraction - inverse_phi_squared):>8.2%}"
            f"{' *' if cutoff == kmax else ' '}"
        )
    print()


def visualize_debruijn_grid(
    kmax, γ=None, rng=None, dirs_to_show=(0, 1, 2, 3, 4),
):
    """Build the geometry and write degree- and rank-alpha plot pairs."""
    dirs_to_show = normalize_dirs_to_show(dirs_to_show)
    data = debruijn_grid(kmax, γ=γ, rng=rng)
    print_rhombus_metrics(data, "degree")
    print_rhombus_metrics(data, "rank")
    selected, grid_segments, grid_families = selected_data(data, dirs_to_show)

    visual_dir = OUTPUT_DIR / "visual"
    visual_dir.mkdir(exist_ok=True)
    directions_tag = "".join(map(str, dirs_to_show))
    output_paths = []
    for alpha_by in ("degree", "rank"):
        output_stem = f"k{kmax}_d{directions_tag}_{alpha_by}"
        svg_path = visual_dir / f"{output_stem}.svg"
        html_path = visual_dir / f"{output_stem}.html"
        fig = build_figure(
            data, selected, grid_segments, grid_families, kmax, alpha_by
        )
        fig.write_html(html_path, include_plotlyjs=True, full_html=True)
        fig.write_image(svg_path)
        output_paths.extend((svg_path, html_path))

    kept = len(data["intersections"])
    shown = int(selected.sum())
    print(
        f"Kept {kept}/{data['candidate_count']} intersections; "
        f"showing {shown} from families {dirs_to_show}"
    )
    for output_path in output_paths:
        print(f"Saved {output_path}")
    return data["centers"], data["angles"], data["colors"], data["vertices"]


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the tensorized de Bruijn pentagrid construction."
    )
    parser.add_argument("kmax", type=int, nargs="?", default=2)
    parser.add_argument(
        "dirs_to_show",
        type=str,
        nargs="?",
        default="01234",
        help="Digit selection such as 024; one digit i means families range(i).",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    dirs_to_show = normalize_dirs_to_show(args.dirs_to_show)

    print(f"K: (-{args.kmax}...{args.kmax})")
    print(f"Directions Shown: {dirs_to_show}")
    visualize_debruijn_grid(
        kmax=args.kmax,
        dirs_to_show=dirs_to_show,
        rng=np.random.default_rng(args.seed),
    )


if __name__ == "__main__":
    main()

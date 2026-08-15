"""
On-the-fly sample generation.

Builds the mask tensor and mother canvas once; 
then each batch is fully vectorized on the device:
  1. pick B masks, draw a rotation θ ~ U(-pi, pi) and a translation
     ~ U(-T, T)^2 (T = translation range in canvas units) per sample,
  2. rotate + translate all M canvas tiles (center + V vertices = V+1 inness points per tile),
  3. map points to mask pixels, average soft values -> inness 0..1, and
     threshold each probe -> vertex_in,
  4. add U(0, 1) noise to break ties, torch.topk to keep the best N tiles.

Outputs per batch (all on device):
    labels (B,),
    xya (B, N, 3) with per-sample zero-mean xy and a unit-variance scaled angle,
    colors (B, N), 
    inness (B, N), 
    vertex_in (B, N, V+1),
    indices (B, N), 
    mask_idx (B,),
    rotation_canvas (B,),
    rotation_mask (B,),
"""
import math
import torch

from canvas import build_canvas_for_mask
from masks import build_masks


ANGLE_SCALE = math.sqrt(3.) / math.pi


def _scaled_angle(angle):
    """Wrap radians to [-pi, pi), then scale to unit variance."""
    wrapped = torch.remainder(angle + math.pi, 2. * math.pi) - math.pi
    return wrapped * ANGLE_SCALE


class SpurSampler:
    def __init__(
        self,
        symmetry,
        num_tiles,
        translation_canvas=0.,
        num_ret_tiles=None,
        seed=None,
        device=None,
        rotation_canvas=math.pi,
        rotation_mask=math.pi/4,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.rotation_canvas = float(rotation_canvas)
        self.rotation_mask = float(rotation_mask)

        masks_data = build_masks()
        mask_hw = tuple(masks_data["masks"].shape[1:])
        canvas_data = build_canvas_for_mask(
            symmetry,
            num_tiles,
            mask_hw,
            masks_data["target_on"],
            translation_canvas,
            seed=seed,
            return_indices=True,
        )

        self.masks = masks_data["masks"].to(self.device)          # (1400, H, W) float32
        self.labels = masks_data["labels"].to(self.device)        # (1400,)
        self.inclass_ids = masks_data["inclass_ids"].to(self.device)
        self.class_names = masks_data["class_names"]
        
        K, H, W = self.masks.shape
        self.mask_flat = self.masks.reshape(K, H * W)
        self.H, self.W = H, W

        self.symmetry = canvas_data["symmetry"]
        self.num_tiles = canvas_data["num_tiles"]
        self.side = canvas_data["side"]
        self.scaling = canvas_data["scaling"]                     # canvas units per pixel
        self.translation_cu = canvas_data["translation"]          # canvas units
        self.num_ret_tiles = num_ret_tiles if num_ret_tiles is not None else self.num_tiles

        centers = canvas_data["centers"].to(self.device)          # (M, 2)
        vertices = canvas_data["vertices"].to(self.device)        # (M, V, 2)
        self.angles = canvas_data["angles"].to(self.device)       # (M,)
        self.colors = canvas_data["colors"].to(self.device)       # (M,)
        self.indices = canvas_data["indices"].to(self.device)     # (M,)

        # inness points: center first, then the V vertices -> (M, V+1, 2)
        self.cvertices = torch.cat([centers[:, None, :], vertices], dim=1)
        self.M, self.V1, _ = self.cvertices.shape

    def __len__(self):
        return len(self.masks)

    def transform_and_inness(self, mask_idx, generator=None):
        """
        Rotate + translate the canvas randomly (one draw per sample) and inness
        every mother tile against the given masks.

        Returns pts (B, M, V+1, 2) transformed 'cvertices' (index 0 is the
        tile center), θ (B,) rotations, float inness (B, M), and Boolean
        vertex_in (B, M, V+1).
        """
        B, dev = len(mask_idx), self.device

        #---------------------------------
        # Rotate + translate the canvas
        #---------------------------------
        trans = (torch.rand(B, 1, 1, 2, device=dev, generator=generator) * 2 - 1) * self.translation_cu
        θ = (torch.rand(B, device=dev, generator=generator) * 2 - 1) * self.rotation_canvas
        cos, sin = torch.cos(θ), torch.sin(θ)
        rot_canvas = torch.stack([
                torch.stack([cos, sin], -1),
                torch.stack([-sin, cos], -1),], -2,)
        cvertices = torch.einsum("mvc,bcd->bmvd", self.cvertices, rot_canvas) + trans  # (B, M, V+1, 2)
        
        #---------------------------------
        # inness: mask hits at V+1 'cvertices'
        #---------------------------------
        pxlxs = torch.round(cvertices[..., 0] / self.scaling + (self.H - 1) / 2).long()      # (B, M, V+1)
        pxlys = torch.round(cvertices[..., 1] / self.scaling + (self.W - 1) / 2).long()      # (B, M, V+1)
        flatpxxy = (pxlxs.clamp(0, self.H - 1) * self.W + pxlys.clamp(0, self.W - 1))        # (B, M, V+1)
        vals = torch.gather(self.mask_flat[mask_idx], 1, 
                            flatpxxy.view(B, -1)).view(B, self.M, self.V1)

        in_bounds = (pxlxs >= 0) & (pxlxs < self.H) & (pxlys >= 0) & (pxlys < self.W)
        vals = vals * in_bounds
        inness = vals.sum(-1) / self.V1                                 # (B, M) in 0..1
        vertex_in = vals > 0.5

        return cvertices, θ, inness, vertex_in

    def sample_batch(self, batch_size, mask_idx=None, generator=None, return_vertices=False):
        """
        mask_idx: optional (B,) tensor of mask indices; random if None.
        Returns dict with unit-variance-angle xya (B, N, 3), colors (B, N),
        indices (B, N), labels (B,), inness (B, N), and Boolean vertex_in
        (B, N, V+1); vertices (B, N, V, 2) if requested.
        """
        B, dev = batch_size, self.device
        if mask_idx is None:
            mask_idx = torch.randint(len(self.masks), (B,), device=dev, generator=generator)
        else:
            mask_idx = mask_idx.to(dev)

        cvertices, θ_canvas, inness, vertex_in = self.transform_and_inness(mask_idx, generator=generator)

        #---------------------------------
        # Top-N with random tie-breaking
        #---------------------------------
        noisy = inness.float() + 1e-3 * torch.rand(B, self.M, device=dev, generator=generator)
        top = torch.topk(noisy, self.num_ret_tiles, dim=1).indices                # (B, N)

        #---------------------------------
        # Gather outputs
        #---------------------------------
        xy = cvertices[:, :, 0, :]                                       # (B, M, 2)
        xy = torch.gather(xy, 1, top[..., None].expand(-1, -1, 2))
        ang = self.angles[top] + θ_canvas[:, None]
        if return_vertices:
            vertices = cvertices[:, :, 1:, :]                                      # (B, M, V, 2)
            topv = top[:, :, None, None].expand(-1, -1, vertices.shape[2], 2)
            vertices = torch.gather(vertices, 1, topv)                             # (B, N, V, 2)

        #---------------------------------
        # Rotate the mask
        #---------------------------------
        θ_mask = (torch.rand(B, device=dev, generator=generator) * 2 - 1) * self.rotation_mask
        cos, sin = torch.cos(θ_mask), torch.sin(θ_mask)
        rot_mask = torch.stack([
                torch.stack([cos, sin], -1),
                torch.stack([-sin, cos], -1),], -2,)
        xy = torch.einsum("bmc,bcd->bmd", xy, rot_mask)
        ang = ang + θ_mask[:, None]
        if return_vertices:
            vertices = torch.einsum("bmvc,bcd->bmvd", vertices, rot_mask)

        sample_center = xy.mean(dim=1, keepdim=True)
        xy = xy - sample_center
        if return_vertices:
            vertices = vertices - sample_center[:, :, None, :]

        #---------------------------------
        # Return outputs
        #---------------------------------
        xya = torch.cat([xy, _scaled_angle(ang)[..., None]], dim=-1)
        inness = torch.gather(inness, 1, top)                           
        vertex_in = torch.gather(vertex_in, 1, top[..., None].expand(-1, -1, self.V1))
        out = {
            "xya": xya,                                                       # (B, N, 3)
            "colors": self.colors[top],                                       # (B, N)
            "indices": self.indices[top],                                     # (B, N)
            "labels": self.labels[mask_idx],                                  # (B,)
            "inness": inness,                                                 # (B, N)
            "vertex_in": vertex_in,                                           # (B, N, V+1)
            "mask_idx": mask_idx,
            "rotation_canvas": θ_canvas,
            "rotation_mask": θ_mask}

        if return_vertices:
            out["vertices"] = vertices

        return out

    def sample_noise(self, batch_size, generator=None):
        """Sample unit-variance Gaussian-XY noise with shape (B, N, 3)."""
        shape = (batch_size, self.num_ret_tiles)
        kwargs = {"device": self.device, "dtype": self.cvertices.dtype, "generator": generator}
        xy = torch.randn((*shape, 2), **kwargs)
        angle = (torch.rand((*shape, 1), **kwargs) * 2. - 1.) * math.sqrt(3.)
        return torch.cat((xy, angle), dim=-1)

    def sample_data_noise(self, batch_size, mask_idx=None, generator=None):
        """Sample data and noise independently and return them unchanged."""
        data = self.sample_batch(batch_size, mask_idx=mask_idx, generator=generator)["xya"]
        noise = self.sample_noise(batch_size, generator=generator)
        return data, noise

    def sample_matched_data_noise(self, batch_size, method="lsa", mask_idx=None, generator=None, **match_kwargs):
        """Sample data and noise, then match noise to data (LSA by default)."""
        from match import match

        data, noise = self.sample_data_noise(batch_size, mask_idx=mask_idx, generator=generator)
        matched_noise = match(data, noise, method=method, **match_kwargs)
        return data, matched_noise


if __name__ == "__main__":
    import time
    from pathlib import Path

    from svg import save_polygons

    output_dir = Path(__file__).resolve().parent / "tests" / "output"
    for symmetry in (6, 5):
        print(f"\n=== Symmetry {symmetry} ===")
        sampler = SpurSampler(symmetry, 96, 2.0, seed=0)
        batch = sampler.sample_batch(8, return_vertices=True)
        for i in range(8):
            label = sampler.class_names[batch["labels"][i]]
            output = output_dir / f"samples/{symmetry}_{i}_{label}.svg"
            save_polygons(output, batch["vertices"][i], batch["colors"][i], show_arcs=symmetry == 5)
            print(f"Saved {output}")
    
        xy = batch["xya"][..., :2]
        t0 = time.time()
        for _ in range(10):
            sampler.sample_batch(64)
        dt = (time.time() - t0) / 10
        print("--------------------------------")
        print(f"xya {tuple(batch['xya'].shape)} colors {tuple(batch['colors'].shape)}"
              f" labels {tuple(batch['labels'].shape)}")
        print(f"x std {xy[..., 0].std():.3f}  y std {xy[..., 1].std():.3f}"
              f"  min inness {batch['inness'].min().item()}")
        print(f"batch of 64 on {sampler.device}: {dt*1000:.1f} ms")

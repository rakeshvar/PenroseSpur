"""Focused tests for cool-class mask selection."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sampler as sampler_module
from cool_classes import (
    COOL_CLASSES,
    COOL_CLASS_IDS,
    resolve_cool_classes,
    validate_cool_classes,
)
from sampler import SpurSampler


EXPECTED_COOL_CLASS_IDS = (
    47, 40, 4, 2, 0, 7, 12, 31, 34, 35,
    61, 11, 1, 49, 60, 9, 62, 29, 30, 26,
    27, 68, 24, 45, 50, 51, 56, 59, 66, 46,
)


def fake_class_names():
    names = [f"class_{class_id}" for class_id in range(70)]
    for class_id, class_name in COOL_CLASSES:
        names[class_id] = class_name
    return names


def fake_masks():
    labels = torch.arange(70).repeat_interleave(20)
    inclass_ids = torch.arange(20, 0, -1).repeat(70)
    return {
        "masks": torch.zeros(1400, 2, 2),
        "labels": labels,
        "inclass_ids": inclass_ids,
        "class_names": fake_class_names(),
        "target_on": 1.0,
    }


def fake_metadata():
    data = fake_masks()
    return {
        key: data[key]
        for key in ("labels", "inclass_ids", "class_names")
    }


def fake_canvas(symmetry, num_tiles, mask_hw, target_on, translation, **kwargs):
    vertices_per_tile = 6 if symmetry == 6 else 4
    offset = kwargs["rng"].uniform() if kwargs.get("rng") is not None else 0.0
    centers = torch.zeros(num_tiles, 2)
    centers[:, 0] = offset
    return {
        "symmetry": symmetry,
        "num_tiles": num_tiles,
        "side": 1.0,
        "scaling": 1.0,
        "translation": translation,
        "centers": centers,
        "vertices": torch.zeros(num_tiles, vertices_per_tile, 2),
        "angles": torch.zeros(num_tiles),
        "colors": torch.zeros(num_tiles, dtype=torch.uint8),
        "indices": torch.arange(num_tiles),
    }


class CoolClassSamplerTest(unittest.TestCase):
    def setUp(self):
        metadata_patch = patch.object(
            sampler_module,
            "load_mask_metadata",
            side_effect=fake_metadata,
        )
        masks_patch = patch.object(
            sampler_module,
            "build_masks",
            side_effect=fake_masks,
        )
        canvas_patch = patch.object(
            sampler_module,
            "build_canvas_for_mask",
            side_effect=fake_canvas,
        )
        self.load_metadata = metadata_patch.start()
        self.build_masks = masks_patch.start()
        self.build_canvas = canvas_patch.start()
        for active_patch in (metadata_patch, masks_patch, canvas_patch):
            self.addCleanup(active_patch.stop)

    def test_cool_class_ranking_matches_requested_order(self):
        self.assertEqual(COOL_CLASS_IDS, EXPECTED_COOL_CLASS_IDS)
        self.assertEqual(len(COOL_CLASS_IDS), 30)
        self.assertEqual(len(set(COOL_CLASS_IDS)), 30)
        validate_cool_classes(fake_class_names())

    def test_num_cool_classes_selects_preference_prefix(self):
        for count in (10, 30):
            with self.subTest(count=count):
                sampler = SpurSampler(
                    6,
                    2,
                    device="cpu",
                    num_cool_classes=count,
                )
                expected_ids = COOL_CLASS_IDS[:count]
                self.assertEqual(len(sampler), count * 20)
                self.assertEqual(sampler.num_cool_classes, count)
                self.assertEqual(sampler.cool_class_ids, expected_ids)
                self.assertEqual(
                    sampler.labels.tolist(),
                    [class_id for class_id in expected_ids for _ in range(20)],
                )
                self.assertEqual(
                    sampler.inclass_ids.tolist(),
                    list(range(1, 21)) * count,
                )
                self.assertTrue(
                    all(
                        sampler.class_names[class_id] == class_name
                        for class_id, class_name in COOL_CLASSES
                    )
                )

    def test_none_keeps_all_masks(self):
        sampler = SpurSampler(6, 2, device="cpu")

        self.assertEqual(len(sampler), 1400)
        self.assertIsNone(sampler.num_cool_classes)
        self.assertIsNone(sampler.cool_class_ids)
        self.assertEqual(
            sampler.labels.tolist(),
            torch.arange(70).repeat_interleave(20).tolist(),
        )

    def test_construction_and_noise_do_not_decode_masks(self):
        sampler = SpurSampler(6, 2, device="cpu")

        self.assertFalse(sampler._data_ready)
        self.assertEqual(sampler.V1, 7)
        self.assertEqual(sampler.side, sampler_module.target_side_for_unit_var(6, 2))
        self.assertEqual(sampler.sample_noise(3).shape, (3, 2, 3))
        self.build_masks.assert_not_called()
        self.build_canvas.assert_not_called()

    def test_first_data_access_loads_once(self):
        sampler = SpurSampler(6, 2, device="cpu")

        self.assertEqual(tuple(sampler.masks.shape), (1400, 2, 2))
        self.assertTrue(sampler._data_ready)
        self.assertIs(sampler.warmup(), sampler)
        sampler.sample_batch(1)
        self.build_masks.assert_called_once_with()
        self.build_canvas.assert_called_once()

    def test_penrose_builds_one_fresh_canvas_per_batch(self):
        sampler = SpurSampler(5, 2, device="cpu", seed=7)

        sampler.sample_batch(1)
        first_canvas = sampler.cvertices.clone()
        self.assertEqual(self.build_canvas.call_count, 1)

        sampler.sample_batch(1)
        second_canvas = sampler.cvertices.clone()
        self.assertEqual(self.build_canvas.call_count, 2)
        self.assertFalse(torch.equal(first_canvas, second_canvas))
        self.build_masks.assert_called_once_with()

    def test_penrose_warmup_grid_is_refreshed_for_first_batch(self):
        sampler = SpurSampler(5, 2, device="cpu", seed=7).warmup()
        warmup_canvas = sampler.cvertices.clone()

        sampler.sample_batch(1)

        self.assertEqual(self.build_canvas.call_count, 2)
        self.assertFalse(torch.equal(warmup_canvas, sampler.cvertices))

    def test_seed_reproduces_penrose_canvas_sequence(self):
        first = SpurSampler(5, 2, device="cpu", seed=11)
        second = SpurSampler(5, 2, device="cpu", seed=11)

        first_sequence = []
        second_sequence = []
        for sampler, sequence in (
            (first, first_sequence),
            (second, second_sequence),
        ):
            for _ in range(2):
                sampler.sample_batch(1)
                sequence.append(sampler.cvertices.clone())

        self.assertTrue(torch.equal(first_sequence[0], second_sequence[0]))
        self.assertTrue(torch.equal(first_sequence[1], second_sequence[1]))
        self.assertFalse(torch.equal(first_sequence[0], first_sequence[1]))

    def test_hex_reuses_canvas_and_default_translation_is_two(self):
        sampler = SpurSampler(6, 2, device="cpu")

        sampler.sample_batch(1)
        sampler.sample_batch(1)

        self.assertEqual(self.build_canvas.call_count, 1)
        self.assertEqual(sampler.translation_cu, 2.0)

    def test_seeded_data_sampling_remains_deterministic(self):
        first = SpurSampler(6, 2, device="cpu")
        second = SpurSampler(6, 2, device="cpu")
        first_batch = first.sample_batch(
            3,
            generator=torch.Generator().manual_seed(17),
        )
        second_batch = second.sample_batch(
            3,
            generator=torch.Generator().manual_seed(17),
        )

        for key in first_batch:
            self.assertTrue(torch.equal(first_batch[key], second_batch[key]), key)

    def test_cool_subset_preserves_metadata_and_mask_order(self):
        sampler = SpurSampler(
            6,
            2,
            device="cpu",
            cool_class_ids=(68, 0, 47),
        )

        expected_labels = [68] * 20 + [0] * 20 + [47] * 20
        self.assertEqual(sampler.labels.tolist(), expected_labels)
        self.assertEqual(len(sampler), 60)
        self.assertEqual(len(sampler.masks), 60)
        self.assertEqual(sampler.labels.tolist(), expected_labels)

    def test_saved_cool_class_ids_are_authoritative(self):
        saved_ids = (68, 0, 47)
        sampler = SpurSampler(
            6,
            2,
            device="cpu",
            num_cool_classes=3,
            cool_class_ids=saved_ids,
        )

        self.assertEqual(sampler.num_cool_classes, 3)
        self.assertEqual(sampler.cool_class_ids, saved_ids)
        self.assertEqual(
            sampler.labels.tolist(),
            [class_id for class_id in saved_ids for _ in range(20)],
        )
        self.assertEqual(sampler.inclass_ids.tolist(), list(range(1, 21)) * 3)

    def test_invalid_cool_class_settings_are_rejected(self):
        cases = (
            (0, None),
            (31, None),
            (2, (47,)),
            (None, (47, 47)),
            (None, (70,)),
            (None, ()),
        )
        for count, class_ids in cases:
            with self.subTest(count=count, class_ids=class_ids):
                with self.assertRaises(ValueError):
                    resolve_cool_classes(count, class_ids)


if __name__ == "__main__":
    unittest.main()

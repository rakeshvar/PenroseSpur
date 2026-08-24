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


def fake_canvas(symmetry, num_tiles, mask_hw, target_on, translation, **kwargs):
    return {
        "symmetry": symmetry,
        "num_tiles": num_tiles,
        "side": 1.0,
        "scaling": 1.0,
        "translation": translation,
        "centers": torch.zeros(num_tiles, 2),
        "vertices": torch.zeros(num_tiles, 4, 2),
        "angles": torch.zeros(num_tiles),
        "colors": torch.zeros(num_tiles, dtype=torch.uint8),
        "indices": torch.arange(num_tiles),
    }


class CoolClassSamplerTest(unittest.TestCase):
    def setUp(self):
        masks_patch = patch.object(sampler_module, "build_masks", fake_masks)
        canvas_patch = patch.object(
            sampler_module,
            "build_canvas_for_mask",
            fake_canvas,
        )
        masks_patch.start()
        canvas_patch.start()
        self.addCleanup(masks_patch.stop)
        self.addCleanup(canvas_patch.stop)

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

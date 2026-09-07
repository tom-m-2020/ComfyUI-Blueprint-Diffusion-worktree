import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from zimage_terminal_refine_discriminator import (
    CHANNELS,
    F_HW,
    G_HW,
    H_HW,
    LOCAL_SIGMA,
    SIGMAS,
    StreamingAssembler,
    bounded_transfer,
    complete_map,
    region_noise,
    regions,
    restrict_working,
)


class TestZImageTerminalRefineDiscriminator(unittest.TestCase):
    def test_fixed_geometry_and_coverage(self):
        planned = regions()
        self.assertEqual(len(planned), 25)
        self.assertEqual((planned[0].y, planned[0].x, planned[0].height, planned[0].width),
                         (0, 0, 64, 64))
        self.assertEqual((planned[-1].y, planned[-1].x,
                          planned[-1].height, planned[-1].width),
                         (192, 192, 64, 64))
        assembler = StreamingAssembler(planned)
        for region in planned:
            assembler.add(torch.ones((1, CHANNELS, *F_HW)), region)
        result, coverage = assembler.finish()
        self.assertTrue(torch.equal(result, torch.ones_like(result)))
        self.assertGreater(float(coverage.min()), 0.0)

    def test_bounded_transfer_matches_complete_map(self):
        generator = torch.Generator().manual_seed(44)
        value = torch.randn((1, CHANNELS, *G_HW), generator=generator)
        mapped = complete_map(value)
        for region in regions():
            working, source_hw = bounded_transfer(value, region)
            restricted = restrict_working(working)
            expected = mapped[..., region.y:region.y2, region.x:region.x2]
            self.assertLessEqual(float((restricted - expected).abs().max()), 1.2e-5)
            self.assertLessEqual(source_hw[0], 34)
            self.assertLessEqual(source_hw[1], 34)

    def test_noise_is_deterministic_and_regional(self):
        planned = regions()
        a = region_noise(planned[0], "cpu", torch.float32)
        b = region_noise(planned[0], "cpu", torch.float32)
        c = region_noise(planned[1], "cpu", torch.float32)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))
        self.assertEqual(tuple(a.shape), (1, CHANNELS, 128, 128))

    def test_fixed_schedule_and_local_member(self):
        self.assertEqual(len(SIGMAS), 9)
        self.assertEqual(SIGMAS[-2], LOCAL_SIGMA)
        self.assertEqual(LOCAL_SIGMA, 0.3000000119)
        self.assertTrue(all(left > right for left, right in zip(SIGMAS[:-1], SIGMAS[1:])))

    def test_shape_contracts_fail_closed(self):
        with self.assertRaises(ValueError):
            bounded_transfer(torch.zeros((1, 128, *G_HW)), regions()[0])
        with self.assertRaises(ValueError):
            restrict_working(torch.zeros((1, CHANNELS, *H_HW)))


if __name__ == "__main__":
    unittest.main()

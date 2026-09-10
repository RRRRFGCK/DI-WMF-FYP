import unittest

import torch
from torch.utils.data import TensorDataset

from domain_mf.corruptions import (
    CORRUPTIONS,
    CorruptedDataset,
    apply_corruption_batch,
    from_pixel_space,
    gaussian_noise,
    to_pixel_space,
)


class CorruptionTests(unittest.TestCase):
    def test_gaussian_noise_is_deterministic_per_sample(self):
        image = torch.full((1, 8, 8), 0.5)
        first = gaussian_noise(image, 0.1, noise_seed=3, sample_index=7)
        second = gaussian_noise(image, 0.1, noise_seed=3, sample_index=7)
        other = gaussian_noise(image, 0.1, noise_seed=4, sample_index=7)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)

    def test_cifar_pixel_round_trip(self):
        pixels = torch.rand(3, 5, 5)
        normalised = from_pixel_space(pixels, "cifar10")
        restored = to_pixel_space(normalised, "cifar10")
        self.assertTrue(torch.allclose(restored, pixels, atol=1e-6))

    def test_zero_severity_preserves_dataset_input(self):
        images = torch.rand(3, 1, 4, 4)
        labels = torch.arange(3)
        wrapped = CorruptedDataset(
            TensorDataset(images, labels), "fashion", "gaussian", 0.0, 0
        )
        for index in range(3):
            image, label = wrapped[index]
            self.assertTrue(torch.equal(image, images[index]))
            self.assertEqual(int(label), index)

    def test_all_corruptions_are_deterministic_and_bounded(self):
        pixels = torch.linspace(0, 1, 3 * 12 * 12).reshape(3, 12, 12)
        for corruption in CORRUPTIONS:
            first = apply_corruption_batch(
                pixels,
                corruption,
                0.2,
                torch.Generator().manual_seed(7),
            )
            second = apply_corruption_batch(
                pixels,
                corruption,
                0.2,
                torch.Generator().manual_seed(7),
            )
            self.assertTrue(torch.equal(first, second), corruption)
            self.assertEqual(first.shape, pixels.shape)
            self.assertGreaterEqual(float(first.min()), 0.0)
            self.assertLessEqual(float(first.max()), 1.0)


if __name__ == "__main__":
    unittest.main()

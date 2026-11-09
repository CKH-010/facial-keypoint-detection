import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data_pipeline import FacialKeypointsDataset


class FacialKeypointsDatasetTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        image = np.zeros((128, 192), dtype=np.uint8)
        image[64:66, 48:50] = 255
        Image.fromarray(image).save(root / "face.png")
        pd.DataFrame([["face.png", 48, 64]], columns=["image", "x", "y"]).to_csv(
            root / "labels.csv", index=False
        )
        self.root = root

    def test_normalizes_annotation_coordinates(self):
        sample = FacialKeypointsDataset(self.root / "labels.csv", self.root)[0]
        torch.testing.assert_close(sample["keypoints"][0], torch.tensor([0.25, 0.5]))
        self.assertGreater(sample["image"][0, 48, 24].item(), 0.5)
        self.assertEqual(sample["mask"].sum().item(), 1)
        self.assertTrue(torch.all(sample["keypoints"][1:] == 0))

    def test_normalizes_224_square_image(self):
        Image.fromarray(np.zeros((224, 224), dtype=np.uint8)).save(
            self.root / "face.png"
        )
        pd.DataFrame([["face.png", 56, 112]], columns=["image", "x", "y"]).to_csv(
            self.root / "labels.csv", index=False
        )
        sample = FacialKeypointsDataset(self.root / "labels.csv", self.root)[0]
        torch.testing.assert_close(sample["keypoints"][0], torch.tensor([0.25, 0.5]))

    def test_flip_keeps_image_and_keypoints_aligned(self):
        dataset = FacialKeypointsDataset(
            self.root / "labels.csv", self.root, augment=True
        )
        with patch("src.data_pipeline.random.random", return_value=0.0):
            sample = dataset[0]
        self.assertGreater(sample["image"][0, 48, 71].item(), 0.5)
        torch.testing.assert_close(
            sample["keypoints"][0] * 96, torch.tensor([71.0, 48.0])
        )
        self.assertTrue(torch.all(sample["keypoints"][1:] == 0))


if __name__ == "__main__":
    unittest.main()

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from models.naimishnet import NaimishNet
from src.alignment_workflow import plot_prediction, transfer_backbone


class AlignmentWorkflowTest(unittest.TestCase):
    def test_transfer_keeps_head_and_copies_backbone(self):
        baseline = NaimishNet()
        advanced = NaimishNet(use_bam_and_coord=True)
        head_before = advanced.classifier[-1].weight.detach().clone()
        transfer_backbone(baseline, advanced)
        torch.testing.assert_close(advanced.features[0].weight[:, :1], baseline.features[0].weight)
        self.assertEqual(torch.count_nonzero(advanced.features[0].weight[:, 1:]).item(), 0)
        torch.testing.assert_close(advanced.features[4].weight, baseline.features[4].weight)
        torch.testing.assert_close(advanced.classifier[-1].weight, head_before)

    def test_plot_uses_matching_mask_and_image_bounds(self):
        class FakeModel(torch.nn.Module):
            def forward(self, image):
                result = torch.zeros((1, 90, 2))
                result[0, 0] = torch.tensor([0.25, 0.5])
                result[0, 1:] = 10
                return result

        images = torch.zeros((1, 1, 96, 96))
        targets = torch.zeros((1, 90, 2))
        targets[0, 0] = torch.tensor([0.5, 0.25])
        masks = torch.zeros((1, 90))
        masks[0, 0] = 1
        fig, ax = plot_prediction(FakeModel(), (images, targets, masks), "cpu", "test")
        self.addCleanup(plt.close, fig)
        self.assertEqual(len(ax.collections[0].get_offsets()), 1)
        self.assertEqual(len(ax.collections[1].get_offsets()), 1)
        self.assertEqual(tuple(ax.collections[0].get_offsets()[0]), (24, 48))
        self.assertEqual(tuple(ax.collections[1].get_offsets()[0]), (48, 24))
        self.assertEqual(ax.get_xlim(), (-0.5, 95.5))
        self.assertEqual(ax.get_ylim(), (95.5, -0.5))


if __name__ == "__main__":
    unittest.main()

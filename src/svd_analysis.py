"""SVD interventions for the facial-keypoint regression head."""

import copy
import torch
from src.evalution import evaluate_held_out_nme

def evaluate_svd_ranks(model, test_loader, device):
    """Return test NME for every retained rank from 1 to 136."""
    images_checked = 0

    for batch in test_loader:
        masks = batch["mask"]

        if masks.ndim != 2 or masks.shape[1] != 90:
            raise ValueError(f"Expected masks with shape (N, 90), got {masks.shape}")

        expected = torch.arange(90, device=masks.device) < 68
        if not (masks.bool() == expected).all():
            raise ValueError("Expected the first 68 landmarks to be valid.")

        images_checked += masks.shape[0]

    if images_checked == 0:
        raise ValueError("The test loader yielded no images.")

    modified = copy.deepcopy(model).to(device).eval()
    layer = modified.classifier[5]

    if layer.weight.shape != (180, 1000):
        raise ValueError(f"Unexpected weight shape: {layer.weight.shape}")

    weights = layer.weight[:136].detach().cpu()
    u, singular_values, vh = torch.linalg.svd(weights, full_matrices=False)

    results = {}
    for rank in range(1, 137):
        reconstructed = (u[:, :rank] * singular_values[:rank]) @ vh[:rank, :]

        with torch.no_grad():
            layer.weight[:136].copy_(reconstructed.to(device))

        results[rank] = evaluate_held_out_nme(
            modified, test_loader, device
        )["nme"]

        if rank % 10 == 0 or rank == 136:
            print(f"Completed rank {rank}/136")

    return results

def plot_svd_mode_direction(model, test_loader):
    import matplotlib.pyplot as plt

    weight = model.classifier[5].weight[:136].detach().cpu()
    u, _, _ = torch.linalg.svd(weight, full_matrices=False)

    batch = next(iter(test_loader))
    image = batch["image"][0, 0].cpu()
    landmarks = batch["keypoints"][0, :68].cpu() * 96

    direction = u[:, 0].reshape(68, 2)
    direction = direction / direction.norm(dim=1).max() * 15

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image, cmap="gray")
    ax.scatter(landmarks[:, 0], landmarks[:, 1], c="lime", s=10)
    ax.quiver(
        landmarks[:, 0], landmarks[:, 1],
        direction[:, 0], direction[:, 1],
        color="red", angles="xy", scale_units="xy", scale=1,
    )
    ax.set_title("Mode 1 output direction")
    return fig, ax
"""Helpers for training and inspecting normalized facial keypoints."""

import torch



def transfer_backbone(baseline, advanced):
    """Copy baseline convolutions, leaving the advanced regression head fresh."""
    source = baseline.state_dict()
    destination = advanced.state_dict()
    transferred = {}
    for name, value in source.items():
        if not name.startswith("features.") or name not in destination:
            continue
        if name == "features.0.weight":
            if (value.shape[1] != 1 or destination[name].shape[1] != 3
                    or value.shape[0] != destination[name].shape[0]
                    or value.shape[2:] != destination[name].shape[2:]):
                raise ValueError("Incompatible baseline and CoordConv first convolutions")
            weight = torch.zeros_like(destination[name])
            weight[:, :1] = value
            transferred[name] = weight
        elif value.shape == destination[name].shape:
            transferred[name] = value
        else:
            raise ValueError(f"Incompatible backbone parameter: {name}")
    advanced.load_state_dict(transferred, strict=False)


def plot_prediction(model, batch, device, title):
    """Plot valid predictions and matching targets on the same input image."""
    import matplotlib.pyplot as plt

    images, targets, masks = batch
    model.eval()
    with torch.no_grad():
        prediction = model(images[:1].to(device)).reshape(-1, 2).cpu()
    valid = masks[0].bool().cpu()
    prediction = prediction[valid].numpy() * 96
    target = targets[0].cpu()[valid].numpy() * 96

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(images[0, 0].cpu().numpy(), cmap="gray")
    ax.scatter(prediction[:, 0], prediction[:, 1], c="red", s=15, label="Predicted")
    ax.scatter(target[:, 0], target[:, 1], c="lime", s=15, label="Ground truth")
    ax.set_xlim(-0.5, 95.5)
    ax.set_ylim(95.5, -0.5)
    ax.legend()
    ax.set_title(title)
    ax.axis("off")
    return fig, ax

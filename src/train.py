import copy
import math
import torch
def train_with_early_stopping(
    model, train_loader, val_loader, optimizer, device,
    max_epochs=50, patience=5
):
    """Train with masked MSE and restore the best validation weights."""
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive.")

    def run_epoch(loader, training):
        model.train(training)
        total_error = 0.0
        total_coordinates = 0

        with torch.set_grad_enabled(training):
            for images, targets, masks in loader:
                images = images.to(device)
                targets = targets.to(device)
                masks = masks.to(device)

                valid_coordinates = (masks.sum() * 2).item()
                if valid_coordinates == 0:
                    raise ValueError("Batch has no valid landmarks.")

                if training:
                    optimizer.zero_grad()

                outputs = model(images).reshape(-1, 90, 2)
                error = (((outputs - targets) ** 2) * masks.unsqueeze(-1)).sum()
                loss = error / valid_coordinates

                if not torch.isfinite(loss):
                    raise ValueError("Loss is not finite.")

                if training:
                    loss.backward()
                    optimizer.step()

                total_error += error.item()
                total_coordinates += valid_coordinates

        if total_coordinates == 0:
            raise ValueError("DataLoader yielded no valid landmarks.")

        return total_error / total_coordinates

    best_loss = float("inf")
    best_weights = None
    no_improvement = 0
    history = []

    for epoch in range(max_epochs):
        train_loss = run_epoch(train_loader, training=True)
        val_loss = run_epoch(val_loader, training=False)
        history.append({"train_loss": train_loss, "val_loss": val_loss})

        print(f"Epoch {epoch + 1}: train={train_loss:.6f}, val={val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Early stopping")
                break

    model.load_state_dict(best_weights)
    return history
import numpy as np
import torch


def _normalize_coords(coords, name):
    """Normalizes (N, K, 2) or (N, 2*K) coordinate arrays into shape (N, K, 2)."""
    coords = np.asarray(coords, dtype=np.float64)

    if coords.ndim == 2:
        if coords.shape[1] % 2 != 0:
            raise ValueError(f"{name} flattened coordinate dimension must be even, got {coords.shape[1]}.")
        return coords.reshape(coords.shape[0], -1, 2)

    if coords.ndim == 3 and coords.shape[-1] == 2:
        return coords

    raise ValueError(f"Expected {name} of shape (N, K, 2) or (N, 2*K), got {coords.shape}.")


def _normalize_landmark_mask(mask, num_samples, num_landmarks):
    """Converts (N, K), (N, K, 2), or (N, 2*K) masks to a (N, K) boolean landmark mask."""
    if mask is None:
        raise ValueError("The dataset mask must be explicitly passed; padded landmarks are zeros.")

    mask = np.asarray(mask)

    if mask.ndim == 3 and mask.shape == (num_samples, num_landmarks, 2):
        # Landmark is valid only when BOTH x and y coordinates are valid
        return mask.astype(bool).all(axis=-1)
    if mask.ndim == 2 and mask.shape == (num_samples, num_landmarks * 2):
        return mask.reshape(num_samples, num_landmarks, 2).astype(bool).all(axis=-1)
    if mask.ndim == 2 and mask.shape == (num_samples, num_landmarks):
        return mask.astype(bool)

    raise ValueError(
        f"Mask shape {mask.shape} is incompatible with ({num_samples}, {num_landmarks}) "
        f"or ({num_samples}, {num_landmarks}, 2)."
    )


def compute_nme(y_true, y_pred, mask, reduction="per_image"):
    """
    Computes Normalized Mean Error (NME) in [0, 1] space (diagonal = sqrt(2)).

    Args:
        y_true:    array of shape (N, K, 2) or (N, 2*K), e.g. (N, 90, 2), normalized to [0, 1]
        y_pred:    array of shape (N, K, 2) or (N, 2*K), e.g. (N, 180), normalized to [0, 1]
        mask:      required boolean or 0/1 array of shape (N, K), (N, K, 2), or (N, 2*K)
        reduction: 'per_image' (each valid image weighted equally) or
                   'per_landmark' (each valid landmark weighted equally)

    Returns:
        dict with 'nme', 'reduction', 'valid_landmarks', 'evaluated_images', and 'excluded_images'
    """
    if reduction not in ("per_image", "per_landmark"):
        raise ValueError(f"Unsupported reduction '{reduction}'. Use 'per_image' or 'per_landmark'.")

    # Normalize both arrays independently to (N, K, 2) before checking shape equality
    y_true = _normalize_coords(y_true, "y_true")
    y_pred = _normalize_coords(y_pred, "y_pred")

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Normalized shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}.")

    num_samples, num_landmarks, _ = y_true.shape
    valid_mask = _normalize_landmark_mask(mask, num_samples, num_landmarks)

    # Identify images with at least one valid landmark
    landmarks_per_image = valid_mask.sum(axis=1)  # shape: (N,)
    has_valid_landmark = landmarks_per_image > 0

    evaluated_images = int(np.sum(has_valid_landmark))
    excluded_images = int(np.sum(~has_valid_landmark))
    valid_landmarks = int(np.sum(landmarks_per_image))

    # Raise an error instead of returning a misleading 0.0 when no valid landmarks exist
    if valid_landmarks == 0 or evaluated_images == 0:
        raise ValueError(
            f"Cannot compute NME: 0 valid landmarks found across all {num_samples} images."
        )

    # Validate finite values at all valid landmark positions
    if not np.isfinite(y_true[valid_mask]).all():
        raise ValueError("y_true contains NaN or Inf values at valid landmark positions.")
    if not np.isfinite(y_pred[valid_mask]).all():
        raise ValueError("y_pred contains NaN or Inf values at valid landmark positions.")

    # Euclidean distance per landmark in [0, 1] space
    distances = np.linalg.norm(y_pred - y_true, axis=-1)  # shape: (N, K)

    if reduction == "per_image":
        # Average valid landmark distances within each valid image first, then average across images
        masked_distances = np.where(valid_mask, distances, 0.0)
        per_image_mean = (
            masked_distances[has_valid_landmark].sum(axis=1)
            / landmarks_per_image[has_valid_landmark]
        )
        nme = float(np.mean(per_image_mean) / np.sqrt(2.0))
    else:
        # Weight every valid landmark equally across the dataset
        nme = float(np.mean(distances[valid_mask]) / np.sqrt(2.0))

    return {
        "nme": nme,
        "reduction": reduction,
        "valid_landmarks": valid_landmarks,
        "evaluated_images": evaluated_images,
        "excluded_images": excluded_images,
    }


def evaluate_held_out_nme(model, dataloader, device, reduction="per_image"):
    """
    Runs PyTorch model inference on a held-out DataLoader and computes NME.
    Expects dictionary batches with keys 'image', 'keypoints', and 'mask'.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["keypoints"]
            masks = batch["mask"]

            preds = model(images)  # shape: (B, 180)

            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(
                targets.detach().cpu().numpy() if torch.is_tensor(targets) else np.asarray(targets)
            )
            all_masks.append(
                masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
            )

    if not all_preds:
        raise ValueError("Held-out dataloader yielded 0 batches.")

    y_pred = np.concatenate(all_preds, axis=0)      # (N, 180)
    y_true = np.concatenate(all_targets, axis=0)    # (N, 90, 2)
    mask = np.concatenate(all_masks, axis=0)        # (N, 90) or (N, 90, 2)

    return compute_nme(y_true, y_pred, mask, reduction=reduction)

if __name__ == "__main__":
    truth = np.zeros((2, 2, 2))
    pred = truth.copy()
    pred[0, 0, 0] = 1
    mask = np.array([[1, 0], [1, 1]])

    print(compute_nme(truth, pred.reshape(2, 4), mask))
    print(compute_nme(truth, pred, mask, reduction="per_landmark"))
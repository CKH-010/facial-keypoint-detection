from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import urllib.request
import zipfile

import os
import torch
from src.data_pipeline import FacialKeypointsDataset

from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import urllib.request
import zipfile

def create_loaders(data_dir="data", batch_size=32, seed=42):
    from torch.utils.data import DataLoader, TensorDataset, Subset

    download_udacity_dataset(data_dir)

    dataset = FacialKeypointsDataset(
        csv_file=os.path.join(data_dir, "training_frames_keypoints.csv"),
        root_dir=os.path.join(data_dir, "training"),
    )
    test_dataset = FacialKeypointsDataset(
        csv_file=os.path.join(data_dir, "test_frames_keypoints.csv"),
        root_dir=os.path.join(data_dir, "test"),
    )

    if len(dataset) < 2:
        raise ValueError("Need at least two training images for a train/validation split.")

    images, keypoints, masks = [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        images.append(batch["image"])
        keypoints.append(batch["keypoints"])
        masks.append(batch["mask"])

    cached = TensorDataset(torch.cat(images), torch.cat(keypoints), torch.cat(masks))
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    split = int(0.9 * len(indices))

    train_loader = DataLoader(
        Subset(cached, indices[:split]), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        Subset(cached, indices[split:]), batch_size=batch_size
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    sample = dataset[indices[0]]
    plot_batch = (
        sample["image"].unsqueeze(0),
        sample["keypoints"].unsqueeze(0),
        sample["mask"].unsqueeze(0),
    )
    return train_loader, val_loader, test_loader, plot_batch

def download_udacity_dataset(data_dir="data"):
    data_dir = Path(data_dir)
    required = (
        "training",
        "test",
        "training_frames_keypoints.csv",
        "test_frames_keypoints.csv",
    )
    missing = [name for name in required if not (data_dir / name).exists()]

    if not missing:
        print("Dataset already exists. Skipping download.")
        return

    url = (
        "https://s3.amazonaws.com/video.udacity-data.com/"
        "topher/2018/May/5aea1b91_train-test-data/train-test-data.zip"
    )
    data_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        archive = temp_dir / "dataset.zip"
        extracted = temp_dir / "extracted"

        print("Downloading dataset...")
        urllib.request.urlretrieve(url, archive)

        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(extracted)

        for name in missing:
            if not (extracted / name).exists():
                raise FileNotFoundError(f"Downloaded archive is missing {name}")

        for name in missing:
            shutil.move(str(extracted / name), str(data_dir / name))

    print(f"Dataset ready in {data_dir}")
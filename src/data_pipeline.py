import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class FacialKeypointsDataset(Dataset):
    """Custom Dataset for loading facial keypoints data."""

    def __init__(self, csv_file: str, root_dir: str, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.keypoints_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.keypoints_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.keypoints_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('L')  # Convert to grayscale
        keypoints = self.keypoints_frame.iloc[idx, 1:].values.astype('float').reshape(-1, 2)

        keypoints = keypoints / 96.0 #normalize keypoints to [0, 1] range

        arr = np.zeros((1, 96, 96), dtype=np.float32)
        image = image.resize((96, 96))
        arr[0, :, :] = np.array(image, dtype=np.float32) / 255.0

        padded_keypoint = np.zeros((90, 2), dtype=np.float32)
        padded_keypoint[:keypoints.shape[0], :] = keypoints

        mask = np.zeros((90,), dtype=np.float32)
        mask[:keypoints.shape[0]] = 1.0

        sample = {'image': arr, 'keypoints': padded_keypoint, 'mask': mask}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        #convert to torch tensors
        sample['image'] = torch.tensor(sample['image'], dtype=torch.float32)
        sample['keypoints'] = torch.tensor(sample['keypoints'], dtype=torch.float32)
        sample['mask'] = torch.tensor(sample['mask'], dtype=torch.float32)


        return sample
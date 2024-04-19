import os

import cv2
from PIL import Image
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None,
                 is_single_channel: bool = True):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.is_single_channel = is_single_channel

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = np.array(Image.open(os.path.join(self.img_dir, img_id + self.img_ext)))

        mask = []
        for i in range(self.num_classes):
            mask_path = os.path.join(self.mask_dir, str(i), img_id + self.mask_ext) # e.g., inputs/矿石图像分割/Annotations/1563.png
            assert os.path.exists(mask_path), f'Mask path {mask_path} does not exist.'
            # m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            m = np.array(Image.open(mask_path))
            mask.append(m[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            img_3channel = np.repeat(img[..., None], 3, axis=-1)
            augmented = self.transform(image=img_3channel, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        if self.is_single_channel:
            img = img.transpose(2, 0, 1)[:1, :, :]
        else:
            img = img.transpose(2, 0, 1)
        mask = mask.astype('float32')
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}

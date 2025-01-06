import numpy as np
import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import scipy.ndimage as ndimage


def norm(img: np.ndarray) -> np.ndarray:
    return (img - img.min()) / (img.max() - img.min())


class RegPro(Dataset):
    def __init__(self,
                 img_dir,
                 transform=None):
        self.img_dir = img_dir
        self.case_list = os.listdir(self.img_dir)
        self.len_ = len(self.case_list)
        self.transform = transform

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        # mr be fixed, us be moving
        img_name = self.case_list[idx]

        img_path = os.path.join(self.img_dir, self.case_list[idx])
        label_path = img_path.replace('images', 'labels')
        # label_prostate_path = img_path.replace('images', 'labels_prostate')

        # nib
        img = nib.load(img_path).get_fdata()
        label = nib.load(label_path).get_fdata()
        _, num_components = ndimage.label(label)
        if num_components > 1:
            label[label == 3] = 0
        label[label > 1] = 1


        # centered pad to [128, 128, 128]
        img = norm(img)
        img = np.pad(img, pad_width=((4, 4), (0, 0), (0, 0)), mode="constant")
        label = np.pad(label, pad_width=((4, 4), (0, 0), (0, 0)), mode="constant")

        # normalization and to tensor
        sample = {'name': img_name, 'volume': img, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        name, volume, label = sample['name'], sample['volume'], sample['label']
        volume = torch.from_numpy(np.expand_dims(volume, axis=0).copy())
        label = torch.from_numpy(np.expand_dims(label, axis=0).copy())
        sample = {'name': name, 'volume': volume, 'label': label}
        return sample
import os
import sys

# import nibabel as nib
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

data_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(data_root, os.path.pardir))

def ToTensor(img):
    return torch.from_numpy(img).float()


def ToTensor1(img):
    return torch.from_numpy((img - img.min()) / (img.max() - img.min())).float()


def norm(img):
    return (img - img.min()) / (img.max() - img.min())


class LPBADataset(Dataset):
    def __init__(self,
                 mr_dir,
                 us_dir,
                 for_test=False):

        self.for_test = for_test
        self.mr_dir = mr_dir
        self.us_dir = us_dir
        self.case_list = os.listdir(self.mr_dir)
        self.len_ = len(self.case_list)

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        if self.for_test:
            # mr be moving, us be fixed
            img_name = self.case_list[idx]

            moving_img_path = os.path.join(self.mr_dir, self.case_list[idx])
            moving_label_path = moving_img_path.replace('images', 'labels')
            fixed_img_path = os.path.join(self.us_dir, self.case_list[idx])
            fixed_label_path = fixed_img_path.replace('images', 'labels')  # 'labels_prostate'
            moving_prostate_label_path = moving_img_path.replace('images', 'labels_prostate')  # _dilated10
            fixed_prostate_label_path = fixed_img_path.replace('images',
                                                               'labels_prostate_vnet')  # _vnet_dilated8

            moving_imgs = sitk.GetArrayFromImage(sitk.ReadImage(moving_img_path)).astype(np.float32)
            moving_labels = sitk.GetArrayFromImage(sitk.ReadImage(moving_label_path)).astype(np.uint8)

            fixed_imgs = sitk.GetArrayFromImage(sitk.ReadImage(fixed_img_path)).astype(np.float32)
            fixed_labels = sitk.GetArrayFromImage(sitk.ReadImage(fixed_label_path)).astype(np.uint8)

            moving_prostate_label = sitk.GetArrayFromImage(sitk.ReadImage(moving_prostate_label_path)).astype(np.uint8)
            fixed_prostate_label = sitk.GetArrayFromImage(sitk.ReadImage(fixed_prostate_label_path)).astype(np.uint8)

            # exclude the region outside prostate
            moving_imgs = norm(moving_imgs)
            fixed_imgs = norm(fixed_imgs)
            moving_imgs_roi = moving_imgs * moving_prostate_label
            fixed_imgs_roi = fixed_imgs * fixed_prostate_label

            moving_imgs = np.pad(moving_imgs, pad_width=((0, 0),
                                                         (0, 0),
                                                         (4, 4)),
                                 mode="constant")[np.newaxis, ...]
            moving_labels = np.pad(moving_labels, pad_width=((0, 0),
                                                             (0, 0),
                                                             (4, 4)),
                                   mode="constant")[np.newaxis, ...]
            fixed_imgs = np.pad(fixed_imgs, pad_width=((0, 0),
                                                       (0, 0),
                                                       (4, 4)),
                                mode="constant")[np.newaxis, ...]
            fixed_labels = np.pad(fixed_labels, pad_width=((0, 0),
                                                           (0, 0),
                                                           (4, 4)),
                                  mode="constant")[np.newaxis, ...]
            moving_imgs_roi = np.pad(moving_imgs_roi, pad_width=((0, 0),
                                                                 (0, 0),
                                                                 (4, 4)),
                                     mode="constant")[np.newaxis, ...]
            fixed_imgs_roi = np.pad(fixed_imgs_roi, pad_width=((0, 0),
                                                               (0, 0),
                                                               (4, 4)),
                                    mode="constant")[np.newaxis, ...]
            moving_prostate_label = np.pad(moving_prostate_label, pad_width=((0, 0),
                                                                             (0, 0),
                                                                             (4, 4)),
                                           mode="constant")[np.newaxis, ...]
            # moving_labels[np.where(moving_labels >= 5)] = 1
            # fixed_labels[np.where(fixed_labels >= 5)] = 1

            # normalization
            moving_imgs = ToTensor(moving_imgs)
            moving_labels = torch.from_numpy(moving_labels).long()
            fixed_imgs = ToTensor(fixed_imgs)
            fixed_labels = torch.from_numpy(fixed_labels).long()
            moving_imgs_roi = ToTensor(moving_imgs_roi)
            fixed_imgs_roi = ToTensor(fixed_imgs_roi)
            moving_prostate_label = torch.from_numpy(moving_prostate_label).long()

            return (moving_imgs, moving_labels, fixed_imgs, fixed_labels, moving_imgs_roi, fixed_imgs_roi,
                    moving_prostate_label, img_name)
        else:
            # mr be moving, us be fixed
            moving_img_path = os.path.join(self.mr_dir, self.case_list[idx])
            moving_label_path = moving_img_path.replace('images', 'labels')
            fixed_img_path = os.path.join(self.us_dir, self.case_list[idx])
            fixed_label_path = fixed_img_path.replace('images', 'labels')  # 'labels_prostate'
            moving_prostate_label_path = moving_img_path.replace('images', 'labels_prostate')  # _dilated10
            fixed_prostate_label_path = fixed_img_path.replace('images', 'labels_prostate')  # _dilated8
            # moving_pro_label_path = moving_img_path.replace('images', 'labels_prostate')  # _dilated
            # fixed_pro_label_path = fixed_img_path.replace('images', 'labels_prostate')

            moving_imgs = sitk.GetArrayFromImage(sitk.ReadImage(moving_img_path)).astype(np.float32)
            moving_labels = sitk.GetArrayFromImage(sitk.ReadImage(moving_label_path)).astype(np.uint8)

            fixed_imgs = sitk.GetArrayFromImage(sitk.ReadImage(fixed_img_path)).astype(np.float32)
            fixed_labels = sitk.GetArrayFromImage(sitk.ReadImage(fixed_label_path)).astype(np.uint8)

            moving_prostate_label = sitk.GetArrayFromImage(sitk.ReadImage(moving_prostate_label_path)).astype(np.uint8)
            fixed_prostate_label = sitk.GetArrayFromImage(sitk.ReadImage(fixed_prostate_label_path)).astype(np.uint8)

            # moving_pro_label = sitk.GetArrayFromImage(sitk.ReadImage(moving_pro_label_path)).astype(np.uint8)
            # fixed_pro_label = sitk.GetArrayFromImage(sitk.ReadImage(fixed_pro_label_path)).astype(np.uint8)

            moving_imgs = norm(moving_imgs)
            fixed_imgs = norm(fixed_imgs)
            moving_imgs = moving_imgs * moving_prostate_label
            fixed_imgs = fixed_imgs * fixed_prostate_label
            moving_labels = moving_labels * moving_prostate_label  # exclude the region outside prostate
            fixed_labels = fixed_labels * fixed_prostate_label

            # pad to [128, 128, 128]
            moving_imgs = np.pad(moving_imgs, pad_width=((0, 0),
                                                         (0, 0),
                                                         (4, 4)),
                                 mode="constant")[np.newaxis, ...]
            moving_labels = np.pad(moving_labels, pad_width=((0, 0),
                                                             (0, 0),
                                                             (4, 4)),
                                   mode="constant")[np.newaxis, ...]
            fixed_imgs = np.pad(fixed_imgs, pad_width=((0, 0),
                                                       (0, 0),
                                                       (4, 4)),
                                mode="constant")[np.newaxis, ...]
            fixed_labels = np.pad(fixed_labels, pad_width=((0, 0),
                                                           (0, 0),
                                                           (4, 4)),
                                  mode="constant")[np.newaxis, ...]

            # moving_prostate_label = np.pad(moving_pro_label, pad_width=((0, 0),
            #                                                             (0, 0),
            #                                                             (4, 4)),
            #                                mode="constant")[np.newaxis, ...]
            # fixed_prostate_label = np.pad(fixed_pro_label, pad_width=((0, 0),
            #                                                           (0, 0),
            #                                                           (4, 4)),
            #                               mode="constant")[np.newaxis, ...]

            # random select a label
            # label_nums = np.max(fixed_labels)
            # label_to_select = random.randint(1, label_nums)
            # moving_labels = (moving_labels == label_to_select).astype(np.uint8)
            # fixed_labels = (fixed_labels == label_to_select).astype(np.uint8)

            moving_labels[moving_labels == 6] = 1
            fixed_labels[fixed_labels == 6] = 1

            # normalization and to tensor
            moving_imgs = ToTensor(moving_imgs)
            moving_labels = torch.from_numpy(moving_labels).long()
            fixed_imgs = ToTensor(fixed_imgs)
            fixed_labels = torch.from_numpy(fixed_labels).long()
            # moving_prostate_label = torch.from_numpy(moving_prostate_label).long()
            # fixed_prostate_label = torch.from_numpy(fixed_prostate_label).long()

            return moving_imgs, moving_labels, fixed_imgs, fixed_labels  #, moving_prostate_label, fixed_prostate_label

    def get_labels_num(self):
        # a_label = nib.load(self.label_paths[0]).get_fdata()
        a_label = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(self.us_dir, self.case_list[0]).replace('images', 'labels_prostate')))

        return int(len(np.unique(a_label)))


class RegProDataset2(Dataset):
    def __init__(self,
                 mr_dir,
                 us_dir,
                 for_test=False):

        self.for_test = for_test
        self.mr_dir = mr_dir
        self.us_dir = us_dir
        self.case_list = os.listdir(self.mr_dir)
        self.len_ = len(self.case_list)

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        if self.for_test:
            # mr be moving, us be fixed
            img_name = self.case_list[idx]

            moving_img_path = os.path.join(self.mr_dir, self.case_list[idx])
            moving_label_path = moving_img_path.replace('images', 'labels')
            fixed_img_path = os.path.join(self.us_dir, self.case_list[idx])
            fixed_label_path = fixed_img_path.replace('images', 'labels')  # 'labels_prostate'
            moving_prostate_label_path = moving_img_path.replace('images', 'labels_prostate')  # _dilated
            fixed_prostate_label_path = fixed_img_path.replace('images', 'labels_prostate_vnet')

            moving_imgs = sitk.GetArrayFromImage(sitk.ReadImage(moving_img_path)).astype(np.float32)
            moving_labels = sitk.GetArrayFromImage(sitk.ReadImage(moving_label_path)).astype(np.uint8)

            fixed_imgs = sitk.GetArrayFromImage(sitk.ReadImage(fixed_img_path)).astype(np.float32)
            fixed_labels = sitk.GetArrayFromImage(sitk.ReadImage(fixed_label_path)).astype(np.uint8)

            moving_prostate_label = sitk.GetArrayFromImage(sitk.ReadImage(moving_prostate_label_path)).astype(np.uint8)
            fixed_prostate_label = sitk.GetArrayFromImage(sitk.ReadImage(fixed_prostate_label_path)).astype(np.uint8)

            moving_imgs = np.pad(moving_imgs, pad_width=((0, 0),
                                                         (0, 0),
                                                         (4, 4)),
                                 mode="constant")[np.newaxis, ...]
            moving_labels = np.pad(moving_labels, pad_width=((0, 0),
                                                             (0, 0),
                                                             (4, 4)),
                                   mode="constant")[np.newaxis, ...]
            fixed_imgs = np.pad(fixed_imgs, pad_width=((0, 0),
                                                       (0, 0),
                                                       (4, 4)),
                                mode="constant")[np.newaxis, ...]
            fixed_labels = np.pad(fixed_labels, pad_width=((0, 0),
                                                           (0, 0),
                                                           (4, 4)),
                                  mode="constant")[np.newaxis, ...]
            fixed_prostate_labels = np.pad(fixed_prostate_label, pad_width=((0, 0),
                                                                            (0, 0),
                                                                            (4, 4)),
                                           mode="constant")[np.newaxis, ...]
            moving_prostate_labels = np.pad(moving_prostate_label, pad_width=((0, 0),
                                                                              (0, 0),
                                                                              (4, 4)),
                                            mode="constant")[np.newaxis, ...]
            # moving_labels[np.where(moving_labels >= 5)] = 1
            # fixed_labels[np.where(fixed_labels >= 5)] = 1

            # normalization
            moving_imgs = norm(moving_imgs)
            fixed_imgs = norm(fixed_imgs)
            moving_imgs = ToTensor(moving_imgs)
            moving_labels = torch.from_numpy(moving_labels).float()
            fixed_imgs = ToTensor(fixed_imgs)
            fixed_labels = torch.from_numpy(fixed_labels).float()
            moving_prostate_labels = torch.from_numpy(moving_prostate_labels).float()
            fixed_prostate_labels = torch.from_numpy(fixed_prostate_labels).float()

            return (moving_imgs, moving_labels, moving_prostate_labels, fixed_imgs, fixed_labels, fixed_prostate_labels,
                    img_name)
        else:
            # mr be moving, us be fixed
            moving_img_path = os.path.join(self.mr_dir, self.case_list[idx])
            fixed_img_path = os.path.join(self.us_dir, self.case_list[idx])
            moving_label_path = moving_img_path.replace('images', 'labels')
            fixed_label_path = fixed_img_path.replace('images', 'labels')
            moving_prostate_label_path = moving_img_path.replace('images', 'labels_prostate')  # _dilated
            fixed_prostate_label_path = fixed_img_path.replace('images', 'labels_prostate')

            moving_label = sitk.GetArrayFromImage(sitk.ReadImage(moving_label_path)).astype(np.uint8)
            fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(fixed_label_path)).astype(np.uint8)

            moving_prostate_label = sitk.GetArrayFromImage(sitk.ReadImage(moving_prostate_label_path)).astype(np.uint8)
            fixed_prostate_label = sitk.GetArrayFromImage(sitk.ReadImage(fixed_prostate_label_path)).astype(np.uint8)
            # pad to [128, 128, 128]
            moving_prostate_labels = np.pad(moving_prostate_label, pad_width=((0, 0),
                                                                              (0, 0),
                                                                              (4, 4)),
                                            mode="constant")[np.newaxis, ...]
            moving_labels = np.pad(moving_label, pad_width=((0, 0),
                                                            (0, 0),
                                                            (4, 4)),
                                   mode="constant")[np.newaxis, ...]
            fixed_prostate_labels = np.pad(fixed_prostate_label, pad_width=((0, 0),
                                                                            (0, 0),
                                                                            (4, 4)),
                                           mode="constant")[np.newaxis, ...]
            fixed_labels = np.pad(fixed_label, pad_width=((0, 0),
                                                          (0, 0),
                                                          (4, 4)),
                                  mode="constant")[np.newaxis, ...]

            moving_labels = torch.from_numpy(moving_labels).float()
            fixed_labels = torch.from_numpy(fixed_labels).float()
            moving_prostate_labels = torch.from_numpy(moving_prostate_labels).float()
            fixed_prostate_labels = torch.from_numpy(fixed_prostate_labels).float()

            return moving_prostate_labels, moving_labels, fixed_prostate_labels, fixed_labels

    def get_labels_num(self):
        # a_label = nib.load(self.label_paths[0]).get_fdata()
        a_label = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(self.us_dir, self.case_list[0]).replace('images', 'labels_prostate')))

        return int(len(np.unique(a_label)))


class RegProDataset3(Dataset):
    def __init__(self,
                 mr_dir,
                 us_dir,
                 for_test=False):

        self.for_test = for_test
        self.mr_dir = mr_dir
        self.us_dir = us_dir
        self.case_list = os.listdir(self.mr_dir)
        self.len_ = len(self.case_list)

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        if self.for_test:
            img_name = self.case_list[idx]

            moving_img_path = os.path.join(self.mr_dir, self.case_list[idx])
            moving_label_path = moving_img_path.replace('images', 'labels')
            fixed_img_path = os.path.join(self.us_dir, self.case_list[idx])
            fixed_label_path = fixed_img_path.replace('images', 'labels')  # 'labels_prostate'
            moving_prostate_label_path = moving_img_path.replace('images', 'labels_prostate')  # _dilated
            fixed_prostate_label_path = fixed_img_path.replace('images', 'labels_prostate_vnet')

            moving_imgs = sitk.GetArrayFromImage(sitk.ReadImage(moving_img_path)).astype(np.float32)
            moving_labels = sitk.GetArrayFromImage(sitk.ReadImage(moving_label_path)).astype(np.uint8)

            fixed_imgs = sitk.GetArrayFromImage(sitk.ReadImage(fixed_img_path)).astype(np.float32)
            fixed_labels = sitk.GetArrayFromImage(sitk.ReadImage(fixed_label_path)).astype(np.uint8)

            moving_prostate_label = sitk.GetArrayFromImage(sitk.ReadImage(moving_prostate_label_path)).astype(np.uint8)
            fixed_prostate_label = sitk.GetArrayFromImage(sitk.ReadImage(fixed_prostate_label_path)).astype(np.uint8)

            moving_imgs = np.pad(moving_imgs, pad_width=((0, 0),
                                                         (0, 0),
                                                         (4, 4)),
                                 mode="constant")[np.newaxis, ...]
            moving_labels = np.pad(moving_labels, pad_width=((0, 0),
                                                             (0, 0),
                                                             (4, 4)),
                                   mode="constant")[np.newaxis, ...]
            fixed_imgs = np.pad(fixed_imgs, pad_width=((0, 0),
                                                       (0, 0),
                                                       (4, 4)),
                                mode="constant")[np.newaxis, ...]
            fixed_labels = np.pad(fixed_labels, pad_width=((0, 0),
                                                           (0, 0),
                                                           (4, 4)),
                                  mode="constant")[np.newaxis, ...]
            fixed_prostate_labels = np.pad(fixed_prostate_label, pad_width=((0, 0),
                                                                            (0, 0),
                                                                            (4, 4)),
                                           mode="constant")[np.newaxis, ...]
            moving_prostate_labels = np.pad(moving_prostate_label, pad_width=((0, 0),
                                                                              (0, 0),
                                                                              (4, 4)),
                                            mode="constant")[np.newaxis, ...]
            # moving_labels[np.where(moving_labels >= 5)] = 1
            # fixed_labels[np.where(fixed_labels >= 5)] = 1

            # normalization
            moving_imgs = norm(moving_imgs)
            fixed_imgs = norm(fixed_imgs)
            moving_imgs = ToTensor(moving_imgs)
            moving_labels = torch.from_numpy(moving_labels).float()
            fixed_imgs = ToTensor(fixed_imgs)
            fixed_labels = torch.from_numpy(fixed_labels).float()
            moving_prostate_labels = torch.from_numpy(moving_prostate_labels).float()
            fixed_prostate_labels = torch.from_numpy(fixed_prostate_labels).float()

            return (moving_imgs, moving_labels, moving_prostate_labels, fixed_imgs, fixed_labels, fixed_prostate_labels,
                    img_name)
        else:
            # mr be moving, us be fixed
            moving_img_path = os.path.join(self.mr_dir, self.case_list[idx])
            fixed_img_path = os.path.join(self.us_dir, self.case_list[idx])
            moving_prostate_label_path = moving_img_path.replace('images', 'labels_prostate')  # _dilated
            fixed_prostate_label_path = fixed_img_path.replace('images', 'labels_prostate')

            moving_imgs = sitk.GetArrayFromImage(sitk.ReadImage(moving_img_path)).astype(np.float32)

            fixed_imgs = sitk.GetArrayFromImage(sitk.ReadImage(fixed_img_path)).astype(np.float32)

            moving_prostate_label = sitk.GetArrayFromImage(sitk.ReadImage(moving_prostate_label_path)).astype(np.uint8)
            fixed_prostate_label = sitk.GetArrayFromImage(sitk.ReadImage(fixed_prostate_label_path)).astype(np.uint8)

            moving_imgs = np.pad(moving_imgs, pad_width=((0, 0),
                                                         (0, 0),
                                                         (4, 4)),
                                 mode="constant")[np.newaxis, ...]
            fixed_imgs = np.pad(fixed_imgs, pad_width=((0, 0),
                                                       (0, 0),
                                                       (4, 4)),
                                mode="constant")[np.newaxis, ...]
            fixed_prostate_labels = np.pad(fixed_prostate_label, pad_width=((0, 0),
                                                                            (0, 0),
                                                                            (4, 4)),
                                           mode="constant")[np.newaxis, ...]
            moving_prostate_labels = np.pad(moving_prostate_label, pad_width=((0, 0),
                                                                              (0, 0),
                                                                              (4, 4)),
                                            mode="constant")[np.newaxis, ...]
            # moving_labels[np.where(moving_labels >= 5)] = 1
            # fixed_labels[np.where(fixed_labels >= 5)] = 1

            # normalization
            moving_imgs = norm(moving_imgs)
            fixed_imgs = norm(fixed_imgs)
            moving_imgs = ToTensor(moving_imgs)
            fixed_imgs = ToTensor(fixed_imgs)
            moving_prostate_labels = torch.from_numpy(moving_prostate_labels).float()
            fixed_prostate_labels = torch.from_numpy(fixed_prostate_labels).float()

            return moving_imgs, moving_prostate_labels, fixed_imgs, fixed_prostate_labels

    def get_labels_num(self):
        # a_label = nib.load(self.label_paths[0]).get_fdata()
        a_label = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(self.us_dir, self.case_list[0]).replace('images', 'labels_prostate')))

        return int(len(np.unique(a_label)))


def regpro(logger,
         mr_dir,
         us_dir,
         batch_size=2,
         is_shuffle=True,
         for_test=False,
         num_workers=2):
    if for_test:
        test_dataset = LPBADataset(mr_dir, us_dir,
                                   for_test=for_test)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=1, num_workers=num_workers)
        return test_loader

    else:

        train_dataset = LPBADataset(mr_dir, us_dir)

        val_dataset = LPBADataset(mr_dir.replace('train', 'val'), us_dir.replace('train', 'val'), for_test=False)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=is_shuffle,
                                  drop_last=True,
                                  num_workers=num_workers)

        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=1, num_workers=num_workers)

        # num_labels = train_dataset.get_labels_num()
        logger.info(f'Training set sizes: {len(train_dataset)}, Train loader size: {len(train_loader)}, '
                    f'Validation set sizes: {len(val_dataset)}')

        return train_loader, val_loader

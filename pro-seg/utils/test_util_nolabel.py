import os
import math
# import xlwt
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from medpy import metric
import torch.nn.functional as F


def test_all_case(net, image_list, num_classes, patch_size, stride_x, stride_y, stride_z,
                  save_result, test_save_path, use_mirror_ensemble):
    for index, image_path in enumerate(image_list):
        affine = nib.load(image_path).affine
        image = nib.load(image_path).get_fdata()
        name = os.path.basename(image_path).split('.')[0]

        prediction, score_map = test_single_case(net, image, stride_x, stride_y, stride_z, patch_size, num_classes)

        if use_mirror_ensemble:
            prediction_flip_reverse_list = []
            for flip_axis in range(3):
                image_flip = np.flip(image, axis=flip_axis)
                prediction_flip, _ = test_single_case(net, image_flip, stride_x, stride_y, stride_z, patch_size,
                                                      num_classes)
                prediction_flip_reverse = np.flip(prediction_flip, axis=flip_axis)
                prediction_flip_reverse_list.append(prediction_flip_reverse)
            # ensemble
            prediction_flip_reverse_sum = 0
            for i in range(3):
                prediction_flip_reverse_sum += prediction_flip_reverse_list[i]
            prediction += prediction_flip_reverse_sum
            prediction = (prediction >= 3).astype(np.uint8)
        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.uint8), affine), test_save_path + "/%s.nii.gz" % name)
            # nib.save(nib.Nifti1Image(image.astype(np.float32), affine), test_save_path + "/%simg.nii.gz" % name)


def norm(img: np.ndarray) -> np.ndarray:
    return (img - img.min()) / (img.max() - img.min())


def test_single_case(net, image, stride_x, stride_y, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant',
                       constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_x) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_y) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in tqdm(range(0, sx)):
        xs = min(stride_x * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_y * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)
                    # ensemble
                    y = F.softmax(y1, dim=1)

                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map

import os
import math
import xlwt
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from medpy import metric
import torch.nn.functional as F
import pandas as pd
from scipy import ndimage


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def test_all_case(net, image_list, num_classes, patch_size, stride_x, stride_y, stride_z,
                  save_result, test_save_path, use_mirror_ensemble, roi=False):
    # loader = tqdm(image_list)
    all_case_dice = []
    all_case_jc = []
    all_case_asd = []
    all_case_95hd = []
    dice_scores = {'DICE ' + str(i): [] for i in range(1, num_classes)}
    all_case_sen = []
    all_case_pre = []
    case_names = {'case': []}
    for index, image_path in enumerate(image_list):
        case_names['case'].append(os.path.basename(image_path))
        affine = nib.load(image_path).affine
        image = nib.load(image_path).get_fdata()
        label_path = image_path.replace('images', 'labels')
        label = nib.load(label_path).get_fdata()
        _, num_components = ndimage.label(label)
        if num_components > 1:
            label[label == 3] = 0
        label[label > 1] = 1

        if roi:
            label_prostate_path = image_path.replace('images', 'labels_prostate')
            label_prostate = nib.load(label_prostate_path).get_fdata()
            image = image * label_prostate
            label = label * label_prostate
            label[label > 4] = 0  # keep landmarks 2-4
            label[label >= 1] = label[label >= 1] - 1
        image = np.pad(image, pad_width=((4, 4), (0, 0), (0, 0)), mode="constant")
        # label = np.pad(label, pad_width=((4, 4), (0, 0), (0, 0)), mode="constant")
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
            prediction = (prediction >= 2).astype(np.uint8)

        prediction = prediction[4:-4, :, :]
        score_map = score_map[1, 4:-4, :, :]
        if roi:
            label = label[4:-4, :, :]
        # for class_idx in range(1, num_classes):
        #     pred_class = (prediction == class_idx).astype(np.uint8)
        #     label_class = (label == class_idx).astype(np.uint8)
        #     if np.sum(label_class) != 0:
        #         dice_scores['DICE ' + str(class_idx)].append(metric.binary.dc(pred_class, label_class))
        #     else:
        #         dice_scores['DICE ' + str(class_idx)].append(None)

        single_case_dc = metric.binary.dc(prediction, label)
        single_case_jc = metric.binary.jc(prediction, label)
        single_case_asd = metric.binary.asd(prediction, label)
        single_case_hd95 = metric.binary.hd95(prediction, label)
        single_case_sen = metric.binary.sensitivity(prediction, label)
        single_case_pre = metric.binary.precision(prediction, label)

        #
        all_case_dice.append(single_case_dc)
        all_case_jc.append(single_case_jc)
        all_case_asd.append(single_case_asd)
        all_case_95hd.append(single_case_hd95)
        all_case_sen.append(single_case_sen)
        all_case_pre.append(single_case_pre)

        print('%2d/%d %s: Dice=%.4f Jaccard=%.4f ASD=%.4f 95HD=%.4f' % (index + 1, len(image_list), name,
                                                                        single_case_dc, single_case_jc,
                                                                        single_case_asd, single_case_hd95))
        print('%2d/%d %s' % (index+1, len(image_list), name))

        if save_result:
            # nib.save(nib.Nifti1Image(prediction.astype(np.uint8), affine), test_save_path + "/%s.nii.gz" % name)
            nib.save(nib.Nifti1Image(score_map.astype(np.float32), affine), test_save_path + "/%s.nii.gz" % name)
            # nib.save(nib.Nifti1Image(image.astype(np.float32), affine), test_save_path + "/%simg.nii.gz" % name)
            # nib.save(nib.Nifti1Image(label.astype(np.uint8), affine), test_save_path + "/%sgt.nii.gz" % name)

    # df = case_names | dice_scores
    # df = pd.DataFrame(df)
    # df.to_excel(os.path.join(test_save_path, 'seg_result.xlsx'), index=False)

    mean_dice = np.array(all_case_dice).mean()
    mean_jc = np.array(all_case_jc).mean()
    mean_asd = np.array(all_case_asd).mean()
    mean_95hd = np.array(all_case_95hd).mean()
    mean_sen = np.array(all_case_sen).mean()
    mean_pre = np.array(all_case_pre).mean()

    std_dice = np.array(all_case_dice).std()
    std_jc = np.array(all_case_jc).std()
    std_asd = np.array(all_case_asd).std()
    std_95hd = np.array(all_case_95hd).std()
    std_sen = np.array(all_case_sen).std()
    std_pre = np.array(all_case_pre).std()

    print('mean: Dice=%.4f Jaccard=%.4f ASD=%.4f 95HD=%.4f' % (mean_dice, mean_jc, mean_asd, mean_95hd))
    print('std:  Dice=%.4f Jaccard=%.4f ASD=%.4f 95HD=%.4f' % (std_dice, std_jc, std_asd, std_95hd))

    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('result', cell_overwrite_ok=True)
    i = 1
    worksheet.write(0, 0, 'name')
    worksheet.write(0, 1, 'Dice')
    worksheet.write(0, 2, 'Jaccard')
    worksheet.write(0, 3, 'ASD')
    worksheet.write(0, 4, '95HD')
    worksheet.write(0, 5, 'Sensitivity')
    worksheet.write(0, 6, 'Precision')
    for img_, dice, jc, asd, hd95, sen, pre in zip(image_list, all_case_dice, all_case_jc, all_case_asd, all_case_95hd, all_case_sen, all_case_pre):
        img_name = os.path.basename(img_)
        worksheet.write(i, 0, img_name)
        worksheet.write(i, 1, str(dice))
        worksheet.write(i, 2, str(jc))
        worksheet.write(i, 3, str(asd))
        worksheet.write(i, 4, str(hd95))
        worksheet.write(i, 5, sen)
        worksheet.write(i, 6, pre)
        i += 1
    worksheet.write(i, 0, 'mean')
    worksheet.write(i, 1, str(mean_dice))
    worksheet.write(i, 2, str(mean_jc))
    worksheet.write(i, 3, str(mean_asd))
    worksheet.write(i, 4, str(mean_95hd))
    worksheet.write(i, 5, mean_sen)
    worksheet.write(i, 6, mean_pre)

    worksheet.write(i + 1, 0, 'std')
    worksheet.write(i + 1, 1, str(std_dice))
    worksheet.write(i + 1, 2, str(std_jc))
    worksheet.write(i + 1, 3, str(std_asd))
    worksheet.write(i + 1, 4, str(std_95hd))
    worksheet.write(i + 1, 5, std_sen)
    worksheet.write(i + 1, 6, std_pre)
    workbook.save(os.path.join(test_save_path, 'origin_semi_result.xls'))

    print('done')

    return mean_dice, mean_jc, mean_asd, mean_95hd
    # return dice_scores

def norm(img: np.ndarray) -> np.ndarray:
    return (img - img.min()) / (img.max() - img.min())


def test_single_case(net, image, stride_x, stride_y, stride_z, patch_size, num_classes=1):
    image = norm(image)
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
                    if isinstance(y1, list):
                        y1 = y1[0]
                    elif isinstance(y1, tuple):
                        y1 = y1[1]
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

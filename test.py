# from __future__ import print_function
import argparse
import os
import pathlib

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch

from models import SpatialTransformer, SRMNet
from utils import setup_seed
from utils.metrics import Get_Jac
from utils.metrics2 import compute_tre_from_masks
from datasets import regpro

setup_seed(2024)


def remove_padding(img_array: np.ndarray):
    return img_array[:, :, 4:-4]


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def write_image(new_array, ori_image, path):
    new_img = sitk.GetImageFromArray(new_array)
    new_img.SetOrigin(ori_image.GetOrigin()[:3])
    new_img.SetSpacing(ori_image.GetSpacing()[:3])
    new_img.SetDirection(ori_image.GetDirection())
    sitk.WriteImage(new_img, path)


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-model", help="filename of pytorch pth model",
                        default='')
    parser.add_argument("-mr_dir", help="MR images folder",
                        default=r'')
    parser.add_argument("-us_dir", help="US images folder",
                        default=r'')
    parser.add_argument("-save_path", help="output nii.gz prediction",
                        default="")
    parser.add_argument("-save_result", default=False)

    args = parser.parse_args()
    save_path = args.save_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(save_path):
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    val_loader = regpro(logger=None,
                      mr_dir=args.mr_dir,
                      us_dir=args.us_dir,
                      batch_size=1,
                      num_workers=2,
                      for_test=True)

    img_size = (128, 128, 128)
    stn_val = SpatialTransformer(size=img_size)
    stn_val = stn_val.to(device)

    # reg_net = VxmDense.load(args.model, device)
    reg_net = SRMNet.load(args.model, device)
    reg_net.to(device)
    reg_net.eval()
    # dice = []
    all_tres = []
    # all_stdJac = []
    # all_negJac = []
    num_classes = 7
    dice_scores = {'DICE ' + str(i): [] for i in range(1, num_classes)}
    tres = {'TRE ' + str(i): [] for i in range(1, num_classes)}
    stdJac = {'stdJac': []}
    case_names = {'case': []}
    negJac = {'negJac': []}
    # struct_elem = create_struct_elem(dilation_radius=10)
    for val_idx, (moving_img, moving_label, fixed_img, fixed_label,
                  moving_img_roi, fixed_img_roi, moving_prostate_label, case_name) in enumerate(val_loader):
        case_names['case'].append(case_name[0])
        moving_img, moving_label, fixed_img, fixed_label = (moving_img.to(device),
                                                            moving_label.to(device),
                                                            fixed_img.to(device),
                                                            fixed_label.to(device))
        moving_img_roi = moving_img_roi.to(device)  # prostate roi
        fixed_img_roi = fixed_img_roi.to(device)
        moving_label_image = sitk.ReadImage(os.path.join(args.mr_dir, case_name[0]).replace('images', 'labels'))
        moving_prostate_label = moving_prostate_label.to(device)

        with torch.no_grad():
            moved_img_roi, moved_label, flow_field = reg_net(moving_img_roi,
                                                             fixed_img_roi,
                                                             mov_seg=moving_label.float())

            moved_prostate_label = stn_val(moving_prostate_label.float(), flow_field, mode='nearest')

            moved_img = stn_val(moving_img, flow_field)

            moved_label = moved_label.short().squeeze().detach().cpu().numpy()
            # dsc_case = DSC(moved_label, fixed_label)[0]
            # dice.append(dsc_case.mean())
            tre_case = []
            fixed_label = fixed_label.short().squeeze().detach().cpu().numpy()
            for class_idx in range(1, num_classes):
                fixed_mask_class = (fixed_label == class_idx).astype(np.uint8)
                moved_mask_class = (moved_label == class_idx).astype(np.uint8)
                if np.sum(fixed_mask_class) < 50 or np.sum(moved_mask_class) < 50:
                    dice_scores['DICE ' + str(class_idx)].append(None)
                    tres['TRE ' + str(class_idx)].append(None)
                else:
                    dice_scores['DICE ' + str(class_idx)].append(dice_coefficient(fixed_mask_class, moved_mask_class))
                    tre = compute_tre_from_masks(moved_mask_class, fixed_mask_class)
                    tres['TRE ' + str(class_idx)].append(tre)
                    tre_case.append(tre)
            all_tres.append(np.mean(tre_case))

            jacdet = Get_Jac(flow_field.permute(0, 2, 3, 4, 1)).cpu().numpy()
            Jac_std = jacdet.std()
            Jac_neg = 100 * ((jacdet <= 0.).sum() / jacdet.size)
            stdJac['stdJac'].append(Jac_std)
            negJac['negJac'].append(Jac_neg)
            # all_stdJac.append(Jac_std)
            # all_negJac.append(Jac_neg)

            if args.save_result:
                flow_field = flow_field.detach().cpu()[0].numpy()
                flow_field = flow_field.transpose((1, 2, 3, 0))
                flow_field = flow_field[:, :, 4:-4, :]
                flow_field = sitk.GetImageFromArray(flow_field)
                flow_field.SetSpacing(moving_label_image.GetSpacing()[:3])
                flow_field.SetOrigin(moving_label_image.GetOrigin()[:3])
                flow_field.SetDirection(moving_label_image.GetDirection())

                moved_img = moved_img[0].squeeze().detach().cpu().numpy()
                moved_img = remove_padding(moved_img)
                moved_label = remove_padding(moved_label)
                moved_prostate_label = moved_prostate_label[0].short().squeeze().detach().cpu().numpy()
                moved_prostate_label = remove_padding(moved_prostate_label)

                moved_img_roi = moved_img_roi[0].squeeze().detach().cpu().numpy()
                moved_img_roi = remove_padding(moved_img_roi)
                fixed_img_roi = fixed_img_roi[0].squeeze().detach().cpu().numpy()
                fixed_img_roi = remove_padding(fixed_img_roi)
                moving_img_roi = moving_img_roi[0].squeeze().detach().cpu().numpy()
                moving_img_roi = remove_padding(moving_img_roi)

                moved_img_roi_save_path = os.path.join(save_path, 'moved_img_roi')
                make_dir(moved_img_roi_save_path)
                fixed_img_roi_save_path = os.path.join(save_path, 'fixed_img_roi')
                make_dir(fixed_img_roi_save_path)
                moving_img_roi_save_path = os.path.join(save_path, 'moving_img_roi')
                make_dir(moving_img_roi_save_path)

                moved_image_save_path = os.path.join(save_path, 'moved_images')
                make_dir(moved_image_save_path)

                moved_landmarks_save_path = os.path.join(save_path, 'moved_labels')
                make_dir(moved_landmarks_save_path)

                flow_field_save_path = os.path.join(save_path, 'flow_fields')
                make_dir(flow_field_save_path)

                moved_prostate_label_save_path = os.path.join(save_path, 'moved_labels_prostate')
                make_dir(moved_prostate_label_save_path)

                sitk.WriteImage(flow_field, os.path.join(flow_field_save_path, case_name[0]))
                write_image(moved_img, moving_label_image, os.path.join(moved_image_save_path, case_name[0]))
                write_image(moved_label, moving_label_image, os.path.join(moved_landmarks_save_path, case_name[0]))
                write_image(moved_prostate_label, moving_label_image,
                            os.path.join(moved_prostate_label_save_path, case_name[0]))
                write_image(moved_img_roi, moving_label_image, os.path.join(moved_img_roi_save_path, case_name[0]))
                write_image(fixed_img_roi, moving_label_image, os.path.join(fixed_img_roi_save_path, case_name[0]))
                write_image(moving_img_roi, moving_label_image, os.path.join(moving_img_roi_save_path, case_name[0]))

            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            # print(dice, jacdet, Jac_std, Jac_neg)

            print(f"Prostate Dice: {dice_scores['DICE 1'][val_idx]}, "
                  f"stdJac {Jac_std :.3f}, Jac<=0: {Jac_neg :.5f}%, "
                  f"TRE {np.mean(tre_case)}")
    print(f"Mean Dice: {np.mean(dice_scores['DICE 1'])}\nMean TRE: {np.mean(all_tres)}\n"
          f"Mean stdJac: {np.mean(stdJac['stdJac'])}\nMean negJac: {np.mean(negJac['negJac'])}")
    df = case_names | dice_scores | tres | stdJac | negJac
    df = pd.DataFrame(df)
    # exp_name = save_path.split('/')[1]
    df.to_excel(os.path.join(save_path, save_path.split('/')[1] + '_test.xlsx'), index=False)


if __name__ == '__main__':
    main()

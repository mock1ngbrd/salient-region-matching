import argparse
import os

import numpy as np
import torch
from medpy import metric

from network.vnet import VNet
from utils.test_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
parser.add_argument('--model_type', type=str, default='iter_num', choices=['best', 'iter_num'])
parser.add_argument("--img_dir", type=str, default='/datasets/RegPro2/val/mr_images',
                    help='dataset root path')
parser.add_argument('--model_root_path', type=str, default='./work_dir', help='model root path')
parser.add_argument('--exp_name', type=str, default='vnet_dice_mr_e-3lr_100e', help='experiment name')
parser.add_argument('--iter_num', type=int, default=1200, help='model iteration')
parser.add_argument('--save_result', type=bool, default=False, help='save result?')
parser.add_argument('--use_mirror_ensemble', type=bool, default=False, help='use mirror for ensemble?')
args = parser.parse_args()


def multi_class_dice(pred, label, num_classes=16):
    total_dsc = []

    for i in range(1, num_classes):
        label_one = label == i
        pred_one = pred == i

        dsc = metric.binary.dc(pred_one, label_one)
        total_dsc.append(dsc)
    total_dsc = np.array(total_dsc).sum() / (num_classes - 1)
    return total_dsc


def test_calculate_metric(model_type, model_root_path, exp_name, iter_num, test_path_list,
                          num_classes, patch_size, stride_x, stride_z, stride_y, save_result, use_mirror_ensemble):
    # load model
    if model_type == 'iter_num':
        model_path = os.path.join(model_root_path, exp_name, 'model_%d.pth' % iter_num)
    elif model_type == 'best':
        model_path = os.path.join(model_root_path, exp_name, 'model_%s.pth' % model_type)
    else:
        raise ValueError('model_type')

    net = VNet(n_channels=1, n_classes=num_classes, n_filters=16, has_dropout=False,
                   normalization='instancenorm').cuda()

    print("init weight from {}".format(model_path))
    net.load_state_dict(torch.load(model_path))
    net.eval()

    if model_type == 'iter_num':
        test_save_path = os.path.join(model_root_path, exp_name, str(iter_num) + 'iterations2')
    elif model_type == 'best':
        test_save_path = os.path.join(model_root_path, exp_name, model_type)
    else:
        raise ValueError('test_save_path')
    if use_mirror_ensemble:
        test_save_path += '_mirror_ensemble'
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    print(test_save_path)

    test_path_list = sorted(test_path_list)

    avg_metric = test_all_case(net, test_path_list, num_classes=num_classes, patch_size=patch_size,
                               stride_x=stride_x, stride_y=stride_y, stride_z=stride_z, save_result=save_result,
                               test_save_path=test_save_path, use_mirror_ensemble=use_mirror_ensemble, roi=False)

    return avg_metric


def data_path_list(img_dir):
    img_name_list = os.listdir(img_dir)
    for i in range(len(img_name_list)):
        img_name_list[i] = os.path.join(img_dir, img_name_list[i])

    return img_name_list


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    num_classes = 2
    patch_size = (128, 128, 128)
    stride_x = int(patch_size[0] // 1.5)
    stride_y = int(patch_size[1] // 1.5)
    stride_z = int(patch_size[2] // 1.5)

    test_path_list = data_path_list(args.img_dir)
    metric = test_calculate_metric(args.model_type, args.model_root_path, args.exp_name, args.iter_num,
                                   args.backbone, test_path_list, num_classes,
                                   patch_size, stride_x, stride_y, stride_z, args.save_result, args.use_mirror_ensemble)

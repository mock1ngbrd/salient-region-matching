import nibabel as nib
# import xlwt
# import xlrd
import os
import numpy as np
import torch
import random
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as distance
# from skimage import segmentation as skimage_seg


def divide_data2train_valid(total_volume_path, total_label_path, total_edge_path, index, seed):
    """
    used to divide all data into training dataset and validation dataset
    :param total_volume_path: the path to total volume
    :param total_label_path: the path to total label
    :param total_edge_path: the path to total edge
    :param index: indicate how many data set as training dataset
    :param seed: for random
    :return:
    """
    random.seed(seed)
    random.shuffle(total_volume_path)
    random.seed(seed)
    random.shuffle(total_label_path)
    random.seed(seed)
    random.shuffle(total_edge_path)

    train_volume_path = total_volume_path[0:index]
    train_label_path = total_label_path[0:index]
    train_edge_path = total_edge_path[0:index]

    valid_volume_path = total_volume_path[index:]
    valid_label_path = total_label_path[index:]
    valid_edge_path = total_edge_path[index:]

    return train_volume_path, train_label_path, train_edge_path, valid_volume_path, valid_label_path, valid_edge_path


def onehot(tensor, label_list, device="cuda:0"):
    """
    one hot encoder
    :param tensor:
    :param label_list:
    :param device: cuda:?
    :return:
    """
    tensor = tensor.float()
    shape = list(tensor.shape)
    # print(shape)
    shape[1] = len(label_list)
    # print(shape)
    result = torch.zeros(shape).to(device)
    for index, label_class in enumerate(label_list):
        label_mask = torch.full(size=list(tensor.shape), fill_value=label_class, dtype=torch.long).to(device)
        label_seg = (label_mask == tensor).float()
        result[:, index, :, :, :] = label_seg.squeeze(dim=1)
    return result


def standardized_seg(seg, label_list, device="cuda:0"):
    """
    standardized seg_tensor with label list to generate a tensor,
    which can be put into nn.CrossEntropy(input, target) as "target"
    :param seg:
    :param label_list: (include 0)
    :param device: cuda device
    :return:
    """
    seg = torch.squeeze(seg, dim=1)
    result = torch.zeros(seg.shape, dtype=torch.long).to(device)
    for index, label_class in enumerate(label_list):
        label_mask = torch.full(size=list(seg.shape), fill_value=label_class, dtype=torch.long).to(device)
        label_seg = (label_mask == seg).long()
        label_seg = label_seg * index
        result = torch.add(result, label_seg)
    return result


# def cross_validation_path(fold_root_path, data_root_path, fold_index=0):
#     fold_txt_path = os.path.join(fold_root_path, 'foldYNbreast48_%d.txt' % fold_index)
#     # fold_txt_path = os.path.join(fold_root_path, 'fold_%d.txt' % fold_index)
#     f = open(fold_txt_path, encoding='gbk')
#     txt = []
#     for line in f:
#         txt.append(line.strip())
#     test_index = txt.index('test:')
#     train_list = txt[1:test_index]
#     test_list = txt[test_index + 1:]
#
#     train_list = [data_root_path+'/img/'+i for i in train_list]
#     test_list = [data_root_path + '/img/' + i for i in test_list]
#     # train_list = [data_root_path + '/' + i for i in train_list]
#     # test_list = [data_root_path + '/' + i for i in test_list]
#     return train_list, test_list

def cross_validation_path(fold_root_path, data_root_path, fold_index=0):
    fold_txt_path = os.path.join(fold_root_path, 'fold_%d.txt' % fold_index)
    f = open(fold_txt_path, 'r', encoding='utf-8')
    txt = []
    for line in f:
        txt.append(line.strip())
    test_index = txt.index('test:')
    train_list = txt[1:test_index]
    test_list = txt[test_index + 1:]

    train_list = [data_root_path + '/imagesTr/' + i for i in train_list]
    test_list = [data_root_path + '/imagesTr/' + i for i in test_list]
    # train_list = [data_root_path + '/' + i for i in train_list]
    # test_list = [data_root_path + '/' + i for i in test_list]
    return train_list, test_list








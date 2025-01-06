import os
from glob import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import label
import SimpleITK as sitk
from medpy import metric
from tqdm import tqdm
import xlwt


def remove_all_but_the_largest_connected_component(image: np.ndarray, for_which_classes: list, volume_per_voxel: float,
                                                   minimum_valid_object_size: dict = None):
    """
    removes all but the largest connected component, individually for each class
    :param image:
    :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
    Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
    to use all foreground classes together)
    :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed. Keys in
    minimum_valid_object_size must match entries in for_which_classes
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]

    assert 0 not in for_which_classes, "cannot remove background"
    largest_removed = {}
    kept_size = {}
    for c in for_which_classes:
        if isinstance(c, (list, tuple)):
            c = tuple(c)  # otherwise it cant be used as key in the dict
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int))

        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum() * volume_per_voxel

        largest_removed[c] = None
        kept_size[c] = None

        if num_objects > 0:
            # we always keep the largest object. We could also consider removing the largest object if it is smaller
            # than minimum_valid_object_size in the future but we don't do that now.
            maximum_size = max(object_sizes.values())
            kept_size[c] = maximum_size

            for object_id in range(1, num_objects + 1):
                # we only remove objects that are not the largest
                if object_sizes[object_id] != maximum_size:
                    # we only remove objects that are smaller than minimum_valid_object_size
                    remove = True
                    if minimum_valid_object_size is not None:
                        remove = object_sizes[object_id] < minimum_valid_object_size[c]
                    if remove:
                        image[(lmap == object_id) & mask] = 0
                        if largest_removed[c] is None:
                            largest_removed[c] = object_sizes[object_id]
                        else:
                            largest_removed[c] = max(largest_removed[c], object_sizes[object_id])
    return image, largest_removed, kept_size


def localization_metrics(target_path_list, prediction_path_list):
    target_num = 0
    hit_num = 0
    cases_result = {}
    mean_dice = 0
    for target_path in target_path_list:
        target = nib.load(target_path).get_fdata()
        _, num_object = label(target.astype(int))
        target_num += num_object
    for prediction_path in tqdm(prediction_path_list):

        single_result = {}

        prediction_sitk = sitk.ReadImage(prediction_path)
        volume_per_voxel = float(np.prod(prediction_sitk.GetSpacing(), dtype=np.float64))

        prediction = nib.load(prediction_path).get_fdata()
        prediction, _, _ = remove_all_but_the_largest_connected_component(prediction, [1], volume_per_voxel, {1: 500})
        prediction_map, num_object = label(prediction.astype(int))

        target_path = prediction_path.replace('pred', 'gt')
        target = nib.load(target_path).get_fdata()
        hit_flag = False
        # dice_list = []
        for i in range(num_object):
            prediction_single_target = (prediction_map == (i+1)).astype(np.float64)
            dice = metric.binary.dc(prediction_single_target, target)
            # dice_list.append(dice)
            if dice > 0.4:
                hit_num += 1
                hit_flag = True
                break
        if not hit_flag:
            print('missed: ', os.path.basename(prediction_path))
        single_result['Hit'] = hit_flag
        single_result['FP_num'] = num_object - int(hit_flag)
        single_result['Dice'] = dice
        mean_dice += dice
        cases_result[os.path.basename(prediction_path).replace('pred.nii.gz', '')] = single_result

    mean_dice /= len(target_path_list)
    sensitivity = hit_num / target_num
    return cases_result, mean_dice, sensitivity


def write_to_xls(cases_result:dict, mean_dice, sensitivity, save_path):
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('result', cell_overwrite_ok=True)
    style = xlwt.XFStyle()
    al = xlwt.Alignment()
    al.horz = 0x02
    al.vert = 0x01
    style.alignment = al
    worksheet.write(0, 0, 'name', style)
    worksheet.write(0, 1, 'Hit', style)
    worksheet.write(0, 2, 'FP_num', style)
    worksheet.write(0, 3, 'Dice', style)

    i = 1
    for key, value in cases_result.items():
        worksheet.write(i, 0, key, style)
        worksheet.write(i, 1, str(value['Hit']), style)
        worksheet.write(i, 2, value['FP_num'], style)
        worksheet.write(i, 3, value['Dice'], style)
        i += 1
    worksheet.write(i, 0, 'sensitivity', style)
    worksheet.write(i, 1, sensitivity, style)
    worksheet.write(i, 3, mean_dice, style)
    workbook.save(os.path.join(save_path, 'result.xls'))


if __name__ == '__main__':
    root_path = '../data/inference/nnunet_SpacingOrigin_fold1/fold_1/30000_mirror_ensemble'
    target_path_list = glob(root_path + '/*gt*')
    prediction_path_list = glob(root_path + '/*pred*')
    cases_result, mean_dice, sensitivity = localization_metrics(target_path_list, prediction_path_list)
    write_to_xls(cases_result, mean_dice, sensitivity, root_path)
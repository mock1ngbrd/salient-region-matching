import ants
import os
import shutil


def save_image(img, ref_img, name, result_dir: str):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    img.set_direction(ref_img.direction)
    img.set_origin(ref_img.origin)
    img.set_spacing(ref_img.spacing)
    ants.image_write(img, os.path.join(result_dir, name))
    print(f"warped img saved to {result_dir}")


def ants_reg(us_image_path, us_mask_path, mr_image_path, mr_mask_path, trans_type, if_pad=False):
    # Load images and masks
    save_path = '/data/fzt/EXP/datasets/RegPro2/val/ants_rigid_mr_vnet2'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # us_image_path = "/home/fzt/EXP/datasets/RegPro/val/us_images/case000065.nii.gz"
    # mr_image_path = us_image_path.replace('us', 'mr')
    # us_mask_path = us_image_path.replace('images', 'labels_prostate')
    # mr_mask_path = mr_image_path.replace('images', 'labels_prostate')
    us_landmarks_path = us_image_path.replace('images', 'labels')
    mr_landmarks_path = mr_image_path.replace('images', 'labels')

    us_image = ants.image_read(us_image_path)
    mr_image = ants.image_read(mr_image_path)
    us_mask = ants.image_read(us_mask_path)
    mr_mask = ants.image_read(mr_mask_path)
    us_landmarks = ants.image_read(us_landmarks_path)
    mr_landmarks = ants.image_read(mr_landmarks_path)

    if if_pad:
        us_mask = us_mask[4:-4, :, :]
        mr_mask = mr_mask[4:-4, :, :]

    us_roi = us_image * us_mask
    mr_roi = mr_image * mr_mask

    case_name = os.path.basename(us_image_path)
    if trans_type == 'Rigid' or trans_type == 'Affine':
        rigid_transform = ants.registration(
            fixed=us_mask,
            moving=mr_mask,
            type_of_transform=trans_type,
            # aff_metric='meansquares'
        )
        rigid_moved_mask = ants.apply_transforms(
            fixed=us_mask,
            moving=mr_mask,
            transformlist=rigid_transform['fwdtransforms'],
            interpolator='nearestNeighbor'
        )
        rigid_moved_image = ants.apply_transforms(
            fixed=us_image,
            moving=mr_image,
            transformlist=rigid_transform['fwdtransforms']
        )
        rigid_moved_landmarks = ants.apply_transforms(
            fixed=us_landmarks,
            moving=mr_landmarks,
            transformlist=rigid_transform['fwdtransforms'],
            interpolator='nearestNeighbor'
        )

        moved_image_save_path = os.path.join(save_path, 'moved_images')
        if not os.path.exists(moved_image_save_path):
            os.makedirs(moved_image_save_path)

        moved_landmarks_save_path = os.path.join(save_path, 'moved_labels')
        if not os.path.exists(moved_landmarks_save_path):
            os.makedirs(moved_landmarks_save_path)

        moved_mask_save_path = os.path.join(save_path, 'moved_labels_prostate')
        if not os.path.exists(moved_mask_save_path):
            os.makedirs(moved_mask_save_path)

        save_image(rigid_moved_image, mr_image, case_name, moved_image_save_path)
        save_image(rigid_moved_mask, mr_mask, case_name, moved_mask_save_path)
        save_image(rigid_moved_landmarks, mr_landmarks, case_name, moved_landmarks_save_path)

    elif trans_type == 'SyN':
        non_rigid_transform = ants.registration(
            fixed=us_roi,
            moving=mr_roi,
            type_of_transform='SyN'
        )

        moved_image = ants.apply_transforms(
            fixed=us_image,
            moving=mr_image,
            transformlist=non_rigid_transform['fwdtransforms']
        )
        moved_mask = ants.apply_transforms(
            fixed=us_mask,
            moving=mr_mask,
            transformlist=non_rigid_transform['fwdtransforms'],
            interpolator="nearestNeighbor"
        )
        moved_landmarks = ants.apply_transforms(
            fixed=us_landmarks,
            moving=mr_landmarks,
            transformlist=non_rigid_transform['fwdtransforms'],
            interpolator="nearestNeighbor"
        )

        moved_image_save_path = os.path.join(save_path, 'moved_images')
        if not os.path.exists(moved_image_save_path):
            os.makedirs(moved_image_save_path)

        moved_landmarks_save_path = os.path.join(save_path, 'moved_labels')
        if not os.path.exists(moved_landmarks_save_path):
            os.makedirs(moved_landmarks_save_path)

        moved_mask_save_path = os.path.join(save_path, 'moved_labels_prostate')
        if not os.path.exists(moved_mask_save_path):
            os.makedirs(moved_mask_save_path)

        save_image(moved_image, us_image, case_name, moved_image_save_path)
        save_image(moved_mask, us_mask, case_name, moved_mask_save_path)
        save_image(moved_landmarks, us_landmarks, case_name, moved_landmarks_save_path)

        flow_field = non_rigid_transform['fwdtransforms'][0]
        flow_field_save_path = os.path.join(save_path, 'flow_field')
        if not os.path.exists(flow_field_save_path):
            os.makedirs(flow_field_save_path)
        shutil.copy(flow_field, os.path.join(flow_field_save_path, case_name))


if __name__ == '__main__':
    us_image_dir = '/data_path'
    us_label_dir = us_image_dir.replace('images', 'labels_prostate')
    mr_image_dir = us_image_dir.replace('us', 'mr')
    mr_label_dir = mr_image_dir.replace('images', 'labels_prostate')
    us_image_list = os.listdir(us_image_dir)

    for us_image_filename in us_image_list:
        us_image_path = os.path.join(us_image_dir, us_image_filename)
        us_mask_path = os.path.join(us_label_dir, us_image_filename)
        mr_image_path = os.path.join(mr_image_dir, us_image_filename)
        mr_mask_path = os.path.join(mr_label_dir, us_image_filename)
        ants_reg(us_image_path, us_mask_path, mr_image_path, mr_mask_path, 'Rigid')

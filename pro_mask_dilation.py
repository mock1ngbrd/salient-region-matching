import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation, label
import os


def load_nifti_image(file_path):
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata(), nifti_img.affine


def save_nifti_image(data, affine, file_path):
    nifti_img = nib.Nifti1Image(data.astype(np.uint8), affine)
    nib.save(nifti_img, file_path)


def process_mask_3d(prostate_mask, landmarks):
    distance_transform = distance_transform_edt(~(prostate_mask).astype(bool))
    distance_transform = distance_transform[landmarks > 0]
    max_distance = np.max(distance_transform)
    dilation_radius = int(np.ceil(max_distance))

    # Create a 3D spherical structuring element for dilation
    struct_elem = np.zeros((dilation_radius * 2 + 1, dilation_radius * 2 + 1, dilation_radius * 2 + 1), dtype=np.uint8)
    center = (dilation_radius, dilation_radius, dilation_radius)
    for x in range(struct_elem.shape[0]):
        for y in range(struct_elem.shape[1]):
            for z in range(struct_elem.shape[2]):
                if np.linalg.norm(np.array([x, y, z]) - center) <= dilation_radius:
                    struct_elem[x, y, z] = 1

    dilated_prostate_mask = binary_dilation(prostate_mask, structure=struct_elem)

    # combined_mask = np.logical_or(dilated_prostate_mask, landmarks)

    included = np.all(dilated_prostate_mask[landmarks > 0])
    if included:
        print("All landmarks are included in the combined mask.")
    else:
        print("Some landmarks are not included.")

    return dilated_prostate_mask


def create_struct_elem(dilation_radius):
    struct_elem = np.zeros((dilation_radius * 2 + 1, dilation_radius * 2 + 1, dilation_radius * 2 + 1), dtype=np.uint8)
    center = (dilation_radius, dilation_radius, dilation_radius)
    for x in range(struct_elem.shape[0]):
        for y in range(struct_elem.shape[1]):
            for z in range(struct_elem.shape[2]):
                if np.linalg.norm(np.array([x, y, z]) - center) <= dilation_radius:
                    struct_elem[x, y, z] = 1
    return struct_elem


def dilate_prostate_mask(prostate_mask, struct_elem):
    # Ensure the mask is boolean
    prostate_mask = prostate_mask.astype(bool)

    # Dilate the prostate mask
    dilated_prostate_mask = binary_dilation(prostate_mask, structure=struct_elem)

    return dilated_prostate_mask


def compute_max_distances(prostate_mask, landmark):
    # Ensure masks are boolean
    prostate_mask = prostate_mask.astype(bool)

    # Compute the distance transform of the prostate mask
    distance_transform = distance_transform_edt(~prostate_mask)

    # Calculate distances from landmarks to the nearest prostate mask edge
    distances = distance_transform[landmark > 0]

    return np.max(distances)


def check_inclusion(dilated_prostate_mask, anatomical_masks):
    # Ensure the masks are boolean
    dilated_prostate_mask = dilated_prostate_mask.astype(bool)
    anatomical_masks = anatomical_masks.astype(bool)

    # Check if all anatomical masks are included in the dilated prostate mask
    inclusion = np.all(dilated_prostate_mask[anatomical_masks >= 1])

    return inclusion


def compute_dilation_radius(prostate_mask_dir, landmark_dir):
    all_distances = []
    landmark_cases = os.listdir(landmark_dir)
    for case in landmark_cases:
        landmark_path = os.path.join(landmark_dir, case)
        prostate_mask_path = os.path.join(prostate_mask_dir, case)

        prostate_mask, _ = load_nifti_image(prostate_mask_path)
        landmarks, _ = load_nifti_image(landmark_path)

        labeled_array, num_components = label(landmarks)
        if num_components > 1:
            landmarks[landmarks == 3] = 0
        landmarks[prostate_mask == 1] = 0  # exclude prostate
        if landmarks.sum() != 0:
            all_distances.append(compute_max_distances(prostate_mask, landmarks))

    all_distances = np.array(all_distances)
    dilation_radius = np.percentile(all_distances, 95)
    print(f"Suggested dilation radius: {dilation_radius} round: {int(np.ceil(dilation_radius))}")
    return dilation_radius, all_distances


if __name__ == '__main__':
    landmark_dir = r'F:\datasets\registration\RegPro2\val\mr_labels'
    prostate_mask_dir = landmark_dir.replace('labels', 'labels_prostate_vnet')  # _vnet

    landmark_cases = os.listdir(landmark_dir)

    # compute dilation_radius
    # dilation_radius, all_distances = compute_dilation_radius(prostate_mask_dir, landmark_dir)
    dilation_radius = 10
    #
    struct_elem = create_struct_elem(dilation_radius)
    case_out_of_roi = []
    prostate_dilated_dir = landmark_dir.replace('labels', 'labels_prostate_vnet_dilated10')
    if not os.path.exists(prostate_dilated_dir):
        os.makedirs(prostate_dilated_dir)

    for case in landmark_cases:
        landmark_path = os.path.join(landmark_dir, case)
        prostate_mask_path = os.path.join(prostate_mask_dir, case)

        prostate_mask, prostate_affine = load_nifti_image(prostate_mask_path)
        landmarks, landmarks_affine = load_nifti_image(landmark_path)

        labeled_array, num_components = label(landmarks)
        if num_components > 1:
            landmarks[landmarks == 3] = 0
        landmarks[prostate_mask == 1] = 0  # exclude prostate

        dillated_prostate_mask = dilate_prostate_mask(prostate_mask, struct_elem)
        if check_inclusion(dillated_prostate_mask, landmarks):
            print(f"{case} All landmarks are included in the dilated mask.")
        else:
            print(f"{case} Some landmarks are not included.")
            case_out_of_roi.append(case)

        save_nifti_image(dillated_prostate_mask, prostate_affine, os.path.join(prostate_dilated_dir, case))

    print(case_out_of_roi)

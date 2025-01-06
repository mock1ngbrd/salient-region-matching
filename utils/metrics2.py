import numpy as np
from scipy.ndimage import center_of_mass


def DSC(outputs, labels, logger=None):
    """
    Evaluation function for Dice score of segmentation overlap
    """
    # organ_labels = {0: "background", 1: "liver", 2: "spleen", 3: "r_kidney", 4: "l_kidney"}
    organ_labels = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}
    label_nums = np.unique(labels)
    # print("labels:", label_nums)
    dice = []
    # dice = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}
    for label in label_nums[1:]:
        iflat = (outputs == label).reshape(-1)
        tflat = (labels == label).reshape(-1)
        intersection = (iflat * tflat).sum()
        dsc = (2. * intersection) / (iflat.sum() + tflat.sum())
        if logger:
            try:
                logger.info(f"{organ_labels[label]}: {dsc :.3f}")
            except:
                pass
        dice.append(dsc)
    return np.asarray(dice)


def RDSC(y_true, y_pred):
    top = 2 * np.sum(y_true * y_pred, axis=tuple(range(1, y_true.ndim)))
    bottom = np.sum(y_true + y_pred, axis=tuple(range(1, y_true.ndim)))

    dice_scores_all = np.divide(top, bottom, out=np.zeros_like(top), where=bottom != 0)
    sorted_indices = np.argsort(dice_scores_all)[::-1]
    num_samples_dsc = int(np.round(len(sorted_indices) * 0.68))
    dsc_scores_for_robustness = dice_scores_all[sorted_indices[:num_samples_dsc]]
    rdsc = np.mean(dsc_scores_for_robustness)
    return rdsc


def compute_centroid(mask):
    """
    Compute the centroid of a binary mask.

    Parameters:
    mask (np.array): Binary mask of shape (H, W) for 2D or (D, H, W) for 3D.

    Returns:
    np.array: Centroid coordinates of shape (2,) for 2D or (3,) for 3D.
    """
    return np.array(center_of_mass(mask))


def centroid_maes(y_true, y_pred):
    y_true_centroids, y_pred_centroids = compute_centroid(y_true), compute_centroid(y_pred)
    maes = np.mean(np.abs(y_true_centroids - y_pred_centroids), axis=0)
    return maes


def RMS(tensor):
    return np.sqrt(np.mean(np.square(tensor)))


def TRE(all_label_maes):
    TREs = [RMS(case) for case in all_label_maes]
    mTRE = np.mean(TREs)
    return mTRE


def RTRE(all_label_maes):
    TREs = [RMS(case) for case in all_label_maes]
    sorted_indices = np.argsort(TREs)
    num_samples_tre = int(np.round(len(sorted_indices) * 0.68))
    tres_for_robustness = TREs[:num_samples_tre]
    rtre = np.mean(tres_for_robustness)
    return rtre


def RTs(all_label_maes):
    TREs = []
    for case in all_label_maes:
        sorted_indices = np.argsort(case)
        num_samples_case_rts = int(np.round(len(sorted_indices) * 0.68))
        case_for_rts = case[sorted_indices[:num_samples_case_rts]]
        case_rms = RMS(case_for_rts)
        TREs.append(case_rms)
    mTRE = np.mean(TREs)
    return mTRE


def compute_tre_from_masks(fixed_mask, moved_mask):
    """
    Compute the Target Registration Error (TRE) from binary masks.

    Parameters:
    fixed_mask (np.array): Binary mask for the fixed image.
    moving_mask (np.array): Binary mask for the moving image.
    transform (np.array): Transformation matrix of shape (3, 3) for 2D or (4, 4) for 3D.

    Returns:
    float: The computed TRE.
    """
    # Compute centroids
    fixed_centroid = compute_centroid(fixed_mask)
    moved_centroid = compute_centroid(moved_mask)

    # Compute the Euclidean distance
    distance = np.linalg.norm(moved_centroid - fixed_centroid)

    return distance

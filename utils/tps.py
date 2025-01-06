import torch
import torch.nn.functional as F
import SimpleITK as sitk
import os


def tps_weights(points, values, lambd=0.01):
    """
    Compute the Thin Plate Spline (TPS) weights.

    Args:
        points: Tensor of shape (N, 3) with control points.
        values: Tensor of shape (N, 3) with values at control points.
        lambd: Regularization parameter.

    Returns:
        Weights for TPS transformation.
    """
    N = points.shape[0]

    # Compute the pairwise distance matrix
    pairwise_dists = torch.cdist(points, points, p=2)
    K = pairwise_dists ** 2 * torch.log(pairwise_dists + 1e-6)

    # Construct the TPS kernel matrix
    P = torch.cat([torch.ones(N, 1), points], dim=1)
    A = torch.cat([torch.cat([K, P], dim=1), torch.cat([P.T, torch.zeros(4, 4)], dim=1)], dim=0)

    # Compute the weights
    V = torch.cat([values, torch.zeros(4, 3)], dim=0)
    weights = torch.linalg.solve(A + lambd * torch.eye(N + 4), V)

    return weights


def tps_interpolation(points, values, query_points):
    """
    Apply the Thin Plate Spline (TPS) interpolation.

    Args:
        points: Tensor of shape (N, 3) with control points.
        values: Tensor of shape (N, 3) with values at control points.
        query_points: Tensor of shape (M, 3) with points to interpolate.

    Returns:
        Interpolated values at query points.
    """
    N = points.shape[0]
    M = query_points.shape[0]
    dim = 3

    # Compute the pairwise distance matrix
    # pairwise_dists = torch.cdist(query_points, points, p=2)
    pairwise_dists = torch.sqrt(
            torch.square(query_points - points).sum(-1) + 1e-6
        )
    K = pairwise_dists ** 2 * torch.log(pairwise_dists + 1e-6)

    # Construct the TPS kernel matrix
    # P = torch.cat([torch.ones(M, 1), query_points], dim=1)

    # Apply the TPS transformation
    # Construct the TPS kernel matrix
    P = torch.cat([torch.ones(N, 1).to('cuda'), points], dim=1)
    # A = torch.cat([torch.cat([K, P], dim=1), torch.cat([P.T, torch.zeros(4, 4)], dim=1)], dim=0)
    A = torch.zeros((N + dim + 1, N + dim + 1)).float()
    A[:, :N, :N] = K
    A[:, :N, -(dim + 1):] = P
    A[:, -(dim + 1):, :N] = P.transpose(1, 2)

    # Compute the weights
    lambd = 0.01
    V = torch.cat([values, torch.zeros(4, 3).to('cuda')], dim=0)
    weights = torch.linalg.solve(A + lambd * torch.eye(N + 4), V)

    interpolated_values = torch.matmul(K, weights[:N]) + torch.matmul(P, weights[N:])

    return interpolated_values


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
flow_dir = '/home/fzt/EXP/registration/voxelmorph-unofficial/work_dir/vxm_1MI_1DICE_4l2_rigid-reg_roi_5e-5cosine-lr_200e_v2/iter_4384_vnet/flow_fields'
origin_img_dir = r'/home/fzt/EXP/datasets/RegPro/val/ants_rigid_mr_vnet/moved_images'
prostate_dir = origin_img_dir.replace('moved_images', 'moved_labels_prostate')
label_dir = origin_img_dir.replace('images', 'labels')
case = 'case000072.nii.gz'
prostate_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(prostate_dir, case)))
prostate_label = torch.from_numpy(prostate_label).long().to(device)
flow = sitk.ReadImage(os.path.join(flow_dir, case))
flow_array = sitk.GetArrayFromImage(flow)
flow_array = torch.from_numpy(flow_array).to(device)
flow_array = flow_array.permute((3, 0, 1, 2)).reshape((3, -1))

# Example usage
D, H, W = prostate_label.shape
x, y, z = torch.meshgrid(torch.arange(W), torch.arange(H), torch.arange(D), indexing='ij')
grid_points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1).float().to(device)

# Flatten the points and values for TPS interpolation
points = grid_points[prostate_label.flatten()]
values = flow_array[:, prostate_label.flatten()]
values = torch.transpose(values, 0, 1)
# Interpolate to the whole volume or the boundary area
interpolated_field = tps_interpolation(points, values, grid_points).T.reshape(3, D, H, W)

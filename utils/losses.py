import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from monai.losses import DiceCELoss

import warnings
from typing import List

warnings.filterwarnings("ignore")


class OHEMLoss(torch.nn.NLLLoss):
    """ Online Hard Example Mining Loss.
    Needs input from nn.LogSoftmax() """

    def __init__(self, ratio, weights):
        super(OHEMLoss, self).__init__(None, True)
        self.ratio = ratio
        self.weights = weights

    def forward(self, x, y):
        if len(x.size()) == 5:
            x = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, x.size(1))
        if len(x.size()) == 4:
            x = x.permute(0, 2, 3, 1).contiguous().view(-1, x.size(1))
        if len(x.size()) == 3:
            x = x.permute(0, 2, 1).contiguous().view(-1, x.size(1))
        y = y.reshape(-1)
        num_inst = x.size(0)
        num_hns = int(self.ratio * num_inst)
        x_ = x.clone()
        inst_losses = F.cross_entropy(x_, y, reduce=False)
        _, idxs = inst_losses.topk(num_hns)
        x_hn = x.index_select(0, idxs)
        y_hn = y.index_select(0, idxs)
        return torch.nn.functional.nll_loss(x_hn, y_hn, weight=self.weights)


def multi_class_dice_loss(soft_pred, target, num_labels, weights=None):
    loss = 0
    target = target.float()
    smooth = 1e-6
    for i in range(num_labels):
        score = soft_pred[:, i]
        target_ = target == i
        intersect = torch.sum(score * target_)
        y_sum = torch.sum(target_ * target_)
        z_sum = torch.sum(score * score)
        loss += ((2 * intersect + smooth) / (z_sum + y_sum + smooth))
        if weights is not None:
            loss *= weights[i]
    loss = 1 - (loss / num_labels)
    return loss


def binary_dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, ignore_index=None):
        super(DiceLoss, self).__init__()
        # self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target, weight=None):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        # dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        # predict = F.softmax(predict, dim=1)

        for i in range(0, target.shape[1]):
            if i not in self.ignore_index:
                dice_loss = binary_dice_loss(predict[:, i], target[:, i])
                if weight is not None:
                    assert weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= weight[i]
                total_loss += dice_loss

        return total_loss / (target.shape[1] - len(self.ignore_index)) \
            if self.ignore_index is not None else total_loss / target.shape[1]


class NCCLoss(nn.Module):
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=3, eps=1e-8):
        super(NCCLoss, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device,
                            requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size / 2))
        J_sum = conv_fn(J, weight, padding=int(win_size / 2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size / 2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size / 2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size / 2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)


def gradient_loss(s, penalty='l2'):
    """
    displacement regularization loss
    """
    if len(s.shape) == 5:
        dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
        dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
        dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

        if penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz
        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        res = d / 3.0

    elif len(s.shape) == 4:
        dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
        dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

        if penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
        d = torch.mean(dx) + torch.mean(dy)
        res = d / 2.0

    return res


def local_displacement_energy(ddf, energy_type, energy_weight):
    def gradient_dx(fv):
        return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

    def gradient_dy(fv):
        return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

    def gradient_dz(fv):
        return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

    def gradient_txyz(Txyz, fn):
        return torch.stack([fn(Txyz[:, i, ...]) for i in range(3)], dim=1)

    def compute_gradient_norm(displacement, flag_l1=False):
        norms = torch.norm(displacement, p=1 if flag_l1 else 2, dim=1)
        return torch.mean(norms)

    def compute_bending_energy(displacement):
        dTdx = gradient_txyz(displacement, gradient_dx)
        dTdy = gradient_txyz(displacement, gradient_dy)
        dTdz = gradient_txyz(displacement, gradient_dz)
        dTdxx = gradient_txyz(dTdx, gradient_dx)
        dTdyy = gradient_txyz(dTdy, gradient_dy)
        dTdzz = gradient_txyz(dTdz, gradient_dz)
        dTdxy = gradient_txyz(dTdx, gradient_dy)
        dTdyz = gradient_txyz(dTdy, gradient_dz)
        dTdxz = gradient_txyz(dTdx, gradient_dz)
        return torch.mean(dTdxx**2 + dTdyy**2 + dTdzz**2 + 2*dTdxy**2 + 2*dTdxz**2 + 2*dTdyz**2)

    if energy_weight:
        if energy_type == 'bending':
            energy = compute_bending_energy(ddf)
        elif energy_type == 'gradient-l2':
            energy = compute_gradient_norm(ddf)
        elif energy_type == 'gradient-l1':
            energy = compute_gradient_norm(ddf, flag_l1=True)
        else:
            raise Exception('Not recognized local regularizer!')
    else:
        energy = torch.tensor(0.0, device=ddf.device)

    return energy * energy_weight


def pdist_squared(x):
    xx = (x ** 2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, 255.0)
    return dist


def MINDSSC(img, radius=3, dilation=3, device=torch.device('cuda')):
    """
    *Preliminary* pytorch implementation.
    MIND-SSC Losses for VoxelMorph
    """
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

    # kernel size
    kernel_size = radius * 2 + 1

    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.tensor([[0, 1, 1],
                                      [1, 1, 0],
                                      [1, 0, 1],
                                      [1, 1, 2],
                                      [2, 1, 1],
                                      [1, 2, 1]]).long()

    # squared distances
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
    mask = ((x > y).view(-1) & (dist == 2).view(-1))

    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).to(device)
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).to(device)
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)

    # compute patch-ssd
    ssd = F.avg_pool3d(rpad2(
        (F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
        kernel_size, stride=1)

    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = mind_var.cpu().data
    mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)

    mind_var = mind_var.to(device)  # .to(device)
    mind /= mind_var
    mind = torch.exp(-mind)

    # permute to have same ordering as C++ code
    mind = mind[:, torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

    return mind  # Tensor: (N, 12, 192, 160, 192)


def MIND_loss(x, y):
    """
    The loss is small, even the voxel intensity distribution of fake image is so difference, loss.item < 0.14
    """
    return torch.mean((MINDSSC(x) - MINDSSC(y)) ** 2)


class TI_Loss(nn.Module):
    """
    References: https://github.com/TopoXLab/TopoInteraction

    The proposed topological interaction (TI) module encodes topological interactions by computing the critical voxels map.
    The critical voxels map contains the locations which induce errors in the topological interactions.
    The TI loss is introduced based on the topological interaction module.
    """

    def __init__(self, dim, connectivity, inclusion, exclusion, min_thick=1):
        """
        :param dim: 2 if 2D; 3 if 3D
        :param connectivity: 4 or 8 for 2D; 6 or 26 for 3D
        :param inclusion: list of [A,B] classes where A is completely surrounded by B.
        :param exclusion: list of [A,C] classes where A and C exclude each other.
        :param min_thick: Minimum thickness/separation between the two classes. Only used if connectivity is 8 for 2D or 26 for 3D
        """
        super(TI_Loss, self).__init__()

        self.dim = dim
        self.connectivity = connectivity
        self.min_thick = min_thick
        self.interaction_list = []
        self.sum_dim_list = None
        self.conv_op = None
        self.apply_nonlin = lambda x: torch.nn.functional.softmax(x, 1)
        self.ce_loss_func = torch.nn.CrossEntropyLoss(reduction='none')

        if self.dim == 2:
            self.sum_dim_list = [1, 2, 3]
            self.conv_op = torch.nn.functional.conv2d
        elif self.dim == 3:
            self.sum_dim_list = [1, 2, 3, 4]
            self.conv_op = torch.nn.functional.conv3d

        self.set_kernel()

        if len(inclusion) != 0:
            for inc in inclusion:
                temp_pair = []
                temp_pair.append(True)  # type inclusion
                temp_pair.append(inc[0])
                temp_pair.append(inc[1])
                self.interaction_list.append(temp_pair)

        if len(exclusion) != 0:
            for exc in exclusion:
                temp_pair = []
                temp_pair.append(False)  # type exclusion
                temp_pair.append(exc[0])
                temp_pair.append(exc[1])
                self.interaction_list.append(temp_pair)
        else:
            raise ValueError(f"Prams 'inclusion' or 'exclusion' must have values")

    def set_kernel(self):
        """
        Sets the connectivity kernel based on user's sepcification of dim, connectivity, min_thick
        """
        k = 2 * self.min_thick + 1
        if self.dim == 2:
            if self.connectivity == 4:
                np_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            elif self.connectivity == 8:
                np_kernel = np.ones((k, k))

        elif self.dim == 3:
            if self.connectivity == 6:
                np_kernel = np.array([
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
                ])
            elif self.connectivity == 26:
                np_kernel = np.ones((k, k, k))

        self.kernel = torch_kernel = torch.from_numpy(np.expand_dims(np.expand_dims(np_kernel, axis=0), axis=0))

    def topological_interaction_module(self, P):
        """
        Given a discrete segmentation map and the intended topological interactions, this module computes the critical voxels map.
        :param P: Discrete segmentation map
        :return: Critical voxels map
        """

        for ind, interaction in enumerate(self.interaction_list):
            interaction_type = interaction[0]
            label_A = interaction[1]
            label_C = interaction[2]

            # Get Masks
            mask_A = torch.where(P == label_A, 1.0, 0.0).double()
            if interaction_type:
                mask_C = torch.where(P == label_C, 1.0, 0.0).double()
                mask_C = torch.logical_or(mask_C, mask_A).double()
                mask_C = torch.logical_not(mask_C).double()
            else:
                mask_C = torch.where(P == label_C, 1.0, 0.0).double()

            # Get Neighbourhood Information
            # padding='same' needs pytorch >= 1.9, it means the output shape keep the same with input.
            # When kernel size is 3, padding=1 got the same perform.
            neighbourhood_C = self.conv_op(mask_C, self.kernel.double(), padding=1)
            neighbourhood_C = torch.where(neighbourhood_C >= 1.0, 1.0, 0.0)
            neighbourhood_A = self.conv_op(mask_A, self.kernel.double(), padding=1)
            neighbourhood_A = torch.where(neighbourhood_A >= 1.0, 1.0, 0.0)

            # Get the pixels which induce errors
            violating_A = neighbourhood_C * mask_A
            violating_C = neighbourhood_A * mask_C
            violating = violating_A + violating_C
            violating = torch.where(violating >= 1.0, 1.0, 0.0)

            if ind == 0:
                critical_voxels_map = violating
            else:
                critical_voxels_map = torch.logical_or(critical_voxels_map, violating).double()

        return critical_voxels_map

    def forward(self, x, y):
        """
        The forward function computes the TI loss value.
        :param x: Likelihood map of shape: b, c, x, y(, z) with c = total number of classes
        :param y: GT of shape: b, c, x, y(, z) with c=1. The GT should only contain values in [0,L) range where L is the total number of classes.
        :return:  TI loss value
        """

        if x.device.type == "cuda":
            self.kernel = self.kernel.cuda(x.device.index)

        # Obtain discrete segmentation map
        x_softmax = self.apply_nonlin(x)
        P = torch.argmax(x_softmax, dim=1)
        P = torch.unsqueeze(P.double(), dim=1)
        del x_softmax

        # Call the Topological Interaction Module
        critical_voxels_map = self.topological_interaction_module(P)

        # Compute the TI loss value
        ce_tensor = torch.unsqueeze(self.ce_loss_func(x.double(), y[:, 0].long()), dim=1)
        ce_tensor[:, 0] = ce_tensor[:, 0] * torch.squeeze(critical_voxels_map, dim=1)
        ce_loss_value = ce_tensor.sum(dim=self.sum_dim_list).mean()

        return ce_loss_value


class MutualInformation(torch.nn.Module):
    """
    Mutual Information
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        # print(sigma)

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def forward(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)


class NormalizedMutualInformation(nn.Module):
    """
    Normalized Mutual Information
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(NormalizedMutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = torch.linspace(minval, maxval, num_bin).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def entropy(self, y):
        y = torch.clamp(y, 0., self.max_clip)
        y = y.view(y.shape[0], -1)
        y = torch.unsqueeze(y, 2)

        nb_voxels = y.shape[1]  # total num of voxels

        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I = torch.exp(- self.preterm * torch.square(y - vbc))
        I = I / torch.sum(I, dim=-1, keepdim=True)

        # compute probabilities
        p = torch.mean(I, dim=1, keepdim=True)

        plogp = p * torch.log(p + 1e-6)  # adding epsilon to avoid log(0)
        entropy = -torch.sum(plogp, dim=1)
        return entropy.mean()  # average across batch

    def mi(self, y_true, y_pred):
        y_true = torch.clamp(y_true, 0., self.max_clip)
        y_pred = torch.clamp(y_pred, 0., self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def nmi(self, y_true, y_pred):
        mi_value = self.mi(y_true, y_pred)
        h_true = self.entropy(y_true)
        h_pred = self.entropy(y_pred)
        nmi_value = (2 * mi_value) / (h_true + h_pred + 1e-6)  # adding epsilon to avoid division by zero
        return nmi_value

    def forward(self, y_true, y_pred):
        return -self.nmi(y_true, y_pred)


class localMutualInformation(torch.nn.Module):
    """
    Local Mutual Information for non-overlapping patches
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32, patch_size=5):
        super(localMutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers
        self.patch_size = patch_size

    def local_mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """Making image paddings"""
        if len(list(y_pred.size())[2:]) == 3:
            ndim = 3
            x, y, z = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            z_r = -z % self.patch_size
            padding = (z_r // 2, z_r - z_r // 2, y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        elif len(list(y_pred.size())[2:]) == 2:
            ndim = 2
            x, y = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            padding = (y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        else:
            raise Exception('Supports 2D and 3D but not {}'.format(list(y_pred.size())))
        y_true = F.pad(y_true, padding, "constant", 0)
        y_pred = F.pad(y_pred, padding, "constant", 0)

        """Reshaping images into non-overlapping patches"""
        if ndim == 3:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 3, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 3, 1))
        else:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 3, 5)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 2, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 3, 5)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 2, 1))

        """Compute MI"""
        I_a_patch = torch.exp(- self.preterm * torch.square(y_true_patch - vbc))
        I_a_patch = I_a_patch / torch.sum(I_a_patch, dim=-1, keepdim=True)

        I_b_patch = torch.exp(- self.preterm * torch.square(y_pred_patch - vbc))
        I_b_patch = I_b_patch / torch.sum(I_b_patch, dim=-1, keepdim=True)

        pab = torch.bmm(I_a_patch.permute(0, 2, 1), I_b_patch)
        pab = pab / self.patch_size ** ndim
        pa = torch.mean(I_a_patch, dim=1, keepdim=True)
        pb = torch.mean(I_b_patch, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()

    def forward(self, y_true, y_pred):
        return -self.local_mi(y_true, y_pred)



if __name__ == "__main__":
    """
    Sample usage. In order to test the code, Input and GT are randomly populated with values.
    Set the dim (2 for 2D; 3 for 3D) correctly to run relevant code.

    The samples provided enforce the following interactions:
        Enforce class 1 to be completely surrounded by class 2
        Enforce class 2 to be excluded from class 3
        Enforce class 3 to be excluded from class 4
    """

    # Parameters for creating random input
    num_classes = height = width = depth = 5

    dim = 3

    if dim == 2:
        x = torch.rand(1, num_classes, height, width)
        y = torch.randint(0, num_classes, (1, 1, height, width))

        ti_loss_weight = 1e-4
        ti_loss_func = TI_Loss(dim=2, connectivity=4, inclusion=[[1, 2]], exclusion=[[2, 3], [3, 4]])
        ti_loss_value = ti_loss_func(x, y) if ti_loss_weight != 0 else 0
        ti_loss_value = ti_loss_weight * ti_loss_value
        print("ti_loss_value: ", ti_loss_value)

    elif dim == 3:
        x = torch.rand(1, num_classes, depth, height, width)
        y = torch.randint(0, num_classes, (1, 1, depth, height, width))

        ti_loss_weight = 1e-6
        ti_loss_func = TI_Loss(dim=3, connectivity=26, inclusion=[[1, 2]], exclusion=[[2, 3], [3, 4]], min_thick=1)
        ti_loss_value = ti_loss_func(x, y) if ti_loss_weight != 0 else 0
        ti_loss_value = ti_loss_weight * ti_loss_value
        print("ti_loss_value: ", ti_loss_value)

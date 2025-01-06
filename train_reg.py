import os
import time
import argparse
import pathlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from utils import get_logger, countParam, LinearWarmupCosineAnnealingLR, setup_seed
from utils.losses import gradient_loss, NCCLoss, MIND_loss, DiceLoss, MutualInformation, NormalizedMutualInformation
from utils.metrics import dice_coeff, Get_Jac
from utils.metrics2 import compute_tre_from_masks
from datasets import regpro
from models import SRMNet, SpatialTransformer
from torch.utils.tensorboard import SummaryWriter
import sys
import shutil

setup_seed(2024)


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs) ** exponent


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-mr_dir", help="MR images folder",
                        default=r'MR_image_path')
    parser.add_argument("-us_dir", help="US images folder",
                        default=r'US_image_path')

    parser.add_argument("-output", help="filename (without extension) for output",
                        default="work_dir/srm_wDICE_04l2_rigid-reg_1e-4lr_250e/")

    # training args
    parser.add_argument("-int_steps", help="Number of flow integration steps. "
                                           "The warp is non-diffeomorphic when this value is 0.",
                        type=int, default=7)  # default 7
    parser.add_argument("-int_downsize", help="Integer specifying the flow downsample factor for vector integration. "
                                              "The flow field is not downsampled when this value is 1.",
                        type=int, default=2)
    parser.add_argument("-corner", help="corner", type=bool, default=False)

    parser.add_argument("-batch_size", help="Dataloader batch size", type=int, default=1)
    parser.add_argument("-lr", help="Optimizer learning rate, keep pace with batch_size",
                        type=float, default=1e-4)
    parser.add_argument("-epochs", help="Train epochs", type=int, default=250)
    parser.add_argument("-resume", help="Path to pretrained model to continute training",
                        default=None)
    parser.add_argument("-interval", help="validation and saving interval", type=int, default=1)
    parser.add_argument("-num_workers", help="Dataloader num_workers", type=int, default=4)

    # losses args
    parser.add_argument("-alpha", type=float, help="weight for regularization loss",
                        default=0.4)
    parser.add_argument("-dice_weight", help="Dice loss weight",
                        type=float, default=1.0)
    parser.add_argument("-sim_weight", help="sim loss weight",
                        type=float, default=1.0)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.output):
        pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)

    py_path_old = sys.argv[0]
    py_path_new = os.path.join(args.output, os.path.basename(py_path_old))
    shutil.copy(py_path_old, py_path_new)

    logger = get_logger(args.output)

    logger.info(f"output to {args.output}")

    writer = SummaryWriter(log_dir=args.output)

    train_loader, val_loader = regpro(logger=logger,
                                    mr_dir=args.mr_dir,
                                    us_dir=args.us_dir,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)

    test_loader = regpro(logger=None,
                      mr_dir=args.mr_dir.replace('train/ants_rigid_mr', 'val/ants_rigid_mr_vnet'),  # ('train/ants_rigid_mr', 'val/ants_rigid_mr_vnet')
                      us_dir=args.us_dir.replace('train', 'val'),  # ('train', 'val')
                      batch_size=1,
                      num_workers=2,
                      for_test=True)

    num_labels = 6
    end_epoch = args.epochs  # 300

    logger.info(f'train scan numbers: {len(train_loader)}')
    logger.info(f'val scan numbers: {len(val_loader)}')
    logger.info(f"num of labels: {num_labels}")

    img_shape = [128, 128, 128]

    # initialise trainable network parts
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    reg_net = SRMNet(
        inshape=img_shape,
        nb_unet_features=[enc_nf, dec_nf],
        int_steps=args.int_steps,
        int_downsize=args.int_downsize)

    reg_net.to(device)
    logger.info(f'VM reg_net params: {countParam(reg_net)}')

    if args.resume:
        reg_net = reg_net.load(args.resume, device=device).to(device)
        logger.info(f"Training resume from {args.resume}")

    reg_net.train()

    stn_val = SpatialTransformer(size=img_shape)
    stn_val.to(device)

    # train using Adam with weight decay and exponential LR decay
    lr = args.lr
    optimizer = optim.Adam(reg_net.parameters(), lr=lr)

    # losses
    sim_criterion = MutualInformation()
    weights = torch.tensor([0.1, 0.1, 0.3, 0.3, 0.3, 0.3]).to(device)
    weights_sim = torch.tensor([0.2, 0.4]).to(device)  # bg, pro [0.2, 0.4]
    grad_criterion = gradient_loss
    dce_criterion = DiceLoss(ignore_index=[])  # dice_loss

    steps, best_acc = 0, 14
    best_acc_test = 14
    run_loss = np.zeros([end_epoch, 4])

    for epoch in range(end_epoch):
        t0 = time.time()
        # lr schedule for prostate region
        lr_ = poly_lr(epoch, end_epoch, lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        weights_pro = poly_lr(epoch, end_epoch, 0.1)
        weights[0], weights[1] = weights_pro, weights_pro
        weights[1] = weights_pro
        writer.add_scalar("Pro_weight", weights[1], global_step=epoch+1)
        for i_batch, (moving_imgs, moving_labels, fixed_imgs, fixed_labels) in enumerate(train_loader):
            steps += 1
            moving_imgs, moving_labels = moving_imgs.to(device), moving_labels.to(device)
            fixed_imgs, fixed_labels = fixed_imgs.to(device), fixed_labels.to(device)
            moved_imgs, flow_field = reg_net(moving_imgs, fixed_imgs)

            moving_labels_one_hot = F.one_hot(moving_labels.long().squeeze(1), num_classes=num_labels).permute(
                0, 4, 1, 2, 3).float()
            # N x Num_Labels x H x W x D

            moved_labels = stn_val(moving_labels_one_hot, flow_field)  # 采用线性插值对seg进行warped

            # ce_loss = ce_criterion(moved_labels, fixed_labels)
            fixed_labels = F.one_hot(fixed_labels.long().squeeze(1), num_classes=num_labels).permute(0, 4, 1, 2,
                                                                                                    3).float()
            # print(fixed_labels.shape)
            grad_loss = grad_criterion(flow_field)
            dce_loss = dce_criterion(moved_labels, fixed_labels, weights)

            sim_loss = sim_criterion(moved_imgs * moved_labels[:, 1:2], fixed_imgs * fixed_labels[:, 1:2]) * weights_sim[1]

            total_loss = args.alpha * grad_loss + args.dice_weight * dce_loss + args.sim_weight * sim_loss

            logger.info(("[Epoch: %4d/%d] [Train index: %2d/%d] "
                         "[sim_loss: %f] [dce_loss: %f] [grad_loss: %f] [total_loss: %f]"
                         % (epoch + 1, end_epoch, i_batch + 1, len(train_loader), args.sim_weight * sim_loss.item(),
                            args.dice_weight * dce_loss.item(), args.alpha * grad_loss.item(), total_loss.item())))

            writer.add_scalar("Sim_loss", args.sim_weight * sim_loss.item(), global_step=steps)
            writer.add_scalar("Dice_loss", args.dice_weight * dce_loss.item(), global_step=steps)
            writer.add_scalar("Grad_loss", args.alpha * grad_loss.item(), global_step=steps)
            writer.add_scalar("total_loss", total_loss.item(), global_step=steps)
            writer.add_scalar("Learning_rate", optimizer.state_dict()['param_groups'][0]['lr'], global_step=steps)

            run_loss[epoch, 0] += total_loss.item()
            # run_loss[epoch, 1] += args.sim_weight * sim_loss.item()
            run_loss[epoch, 2] += args.alpha * grad_loss.item()
            # run_loss[epoch, 3] += args.dice_weight * dce_loss.item()
            total_loss.backward()

            optimizer.step()
            optimizer.zero_grad()


        time_t = time.time() - t0

        if epoch % args.interval == 0:
            reg_net.eval()
            Jac_std, Jac_neg = [], []

            dice_mean = []
            dice_prostate = []
            tre_mean = []
            for val_idx, (moving_img, moving_label, fixed_img, fixed_label) in enumerate(val_loader):
                moving_img, moving_label, fixed_img, fixed_label = moving_img.to(device), moving_label.to(
                    device), fixed_img.to(device), fixed_label.to(device)
                t0 = time.time()
                with torch.no_grad():
                    _, moved_label, flow_field = reg_net(moving_img,
                                                         fixed_img,
                                                         mov_seg=moving_label.float())
                    time_i = time.time() - t0

                    # dice_all_val[val_idx] = dice_coeff(fixed_label.cpu(), moved_label.long().cpu())
                    dice_sample = dice_coeff(fixed_label.cpu(), moved_label.long().cpu())
                    dice_prostate.append(dice_sample[0])
                    dice_mean.append(dice_sample[1:].mean())

                    # complexity of transformation and foldings
                    jacdet = Get_Jac(flow_field.permute(0, 2, 3, 4, 1)).cpu()
                    Jac_std.append(jacdet.std())
                    Jac_neg.append(100 * ((jacdet <= 0.).sum() / jacdet.numel()))

                    moved_label = moved_label.short().squeeze().detach().cpu().numpy()
                    # dsc_case = DSC(moved_label, fixed_label)[0]
                    # dice.append(dsc_case.mean())
                    tre_case = []
                    fixed_label = fixed_label.short().squeeze().detach().cpu().numpy()
                    for class_idx in range(1, num_labels):
                        fixed_mask_class = (fixed_label == class_idx).astype(np.uint8)
                        moved_mask_class = (moved_label == class_idx).astype(np.uint8)
                        if np.sum(fixed_mask_class) >= 50 and np.sum(moved_mask_class) >= 50:
                            tre = compute_tre_from_masks(moved_mask_class, fixed_mask_class)
                            tre_case.append(tre)
                            # tre_mean.append(tre)
                    tre_mean.append(np.mean(tre_case))

            # logger some feedback information
            latest_lr = optimizer.state_dict()['param_groups'][0]['lr']

            dice_mean = np.mean(dice_mean)
            dice_prostate = np.mean(dice_prostate)
            tre_mean = np.mean(tre_mean)
            # is_best = dice_mean > best_acc + 0.003
            # best_acc = max(dice_mean, best_acc)
            is_best = tre_mean < best_acc - 0.1
            best_acc = min(tre_mean, best_acc)

            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            logger.info(
                f"[epoch: {epoch + 1}], [step {steps}], [time train {round(time_t, 3)}], [time infer {round(time_i, 3)}],"
                f"[stdJac {np.mean(Jac_std) :.3f}], [Jac<=0 {np.mean(Jac_neg) :.3f}%], [tre_mean {tre_mean :.3f}], "
                f"[dice_mean {dice_mean :.3f}], [dice_prostate {dice_prostate :.3f}], [best_tre {best_acc :.3f}],"
                f"[lr {latest_lr :.8f}]")
            # f"[total loss {run_loss[epoch, 0] :.3f}], "
            # f"[grad loss {run_loss[epoch, 2] :.3f}], [dce loss {run_loss[epoch, 3] :.3f}], "

            writer.add_scalar('val_result/Dice', dice_prostate, global_step=steps)
            writer.add_scalar('val_result/Dice_mean', dice_mean, global_step=steps)
            writer.add_scalar('val_result/TRE_mean', tre_mean, global_step=steps)
            writer.add_scalar('val_result/Jac_std', np.mean(Jac_std), global_step=steps)
            writer.add_scalar('val_result/Jac<=0', np.mean(Jac_neg), global_step=steps)

            dice_mean_test = []
            dice_prostate_test = []
            tre_mean_test = []
            for val_idx, (_, moving_label, _, fixed_label,
                          moving_img_roi, fixed_img_roi, _, _) in enumerate(test_loader):
                moving_label, fixed_label = (moving_label.to(device), fixed_label.to(device))
                moving_img_roi = moving_img_roi.to(device)  # prostate roi
                fixed_img_roi = fixed_img_roi.to(device)
                with torch.no_grad():
                    _, moved_label, flow_field = reg_net(moving_img_roi,
                                                         fixed_img_roi,
                                                         mov_seg=moving_label.float())
                    time_i = time.time() - t0

                    # dice_all_val[val_idx] = dice_coeff(fixed_label.cpu(), moved_label.long().cpu())
                    dice_sample = dice_coeff(fixed_label.cpu(), moved_label.long().cpu())
                    dice_prostate_test.append(dice_sample[0])
                    dice_mean_test.append(dice_sample[1:].mean())

                    moved_label = moved_label.short().squeeze().detach().cpu().numpy()
                    # dsc_case = DSC(moved_label, fixed_label)[0]
                    # dice.append(dsc_case.mean())
                    tre_case = []
                    fixed_label = fixed_label.short().squeeze().detach().cpu().numpy()
                    for class_idx in range(1, num_labels):
                        fixed_mask_class = (fixed_label == class_idx).astype(np.uint8)
                        moved_mask_class = (moved_label == class_idx).astype(np.uint8)
                        if np.sum(fixed_mask_class) >= 50 and np.sum(moved_mask_class) >= 50:
                            tre = compute_tre_from_masks(moved_mask_class, fixed_mask_class)
                            tre_case.append(tre)
                    tre_mean_test.append(np.mean(tre_case))

            dice_mean_test = np.mean(dice_mean_test)
            dice_prostate_test = np.mean(dice_prostate_test)
            tre_mean_test = np.mean(tre_mean_test)
            # is_best = dice_mean > best_acc + 0.003
            # best_acc = max(dice_mean, best_acc)
            is_best_test = tre_mean_test < best_acc_test - 0.01
            best_acc_test = min(tre_mean_test, best_acc_test)

            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            logger.info(
                f"[test_tre_mean {tre_mean_test :.3f}], "
                f"[test_dice_mean {dice_mean_test :.3f}], [test_dice_prostate {dice_prostate_test :.3f}], [best_tre {best_acc_test :.3f}],"
                f"[lr {latest_lr :.8f}]")
            # f"[total loss {run_loss[epoch, 0] :.3f}], "
            # f"[grad loss {run_loss[epoch, 2] :.3f}], [dce loss {run_loss[epoch, 3] :.3f}], "

            writer.add_scalar('test_result/Dice', dice_prostate_test, global_step=steps)
            writer.add_scalar('test_result/Dice_mean', dice_mean_test, global_step=steps)
            writer.add_scalar('test_result/TRE_mean', tre_mean_test, global_step=steps)

            if is_best_test and steps >= 5000:
                reg_net.save("%siter_%d_dice_%.3f.pth" % (args.output, steps, best_acc_test))
                logger.info(f"saved the best model at epoch {epoch}, with best acc {best_acc_test :.3f}")

            reg_net.train()

    reg_net.save("%siter_%d.pth" % (args.output, steps))
    logger.info("save model : %siter_%d.pth" % (args.output, steps))
    writer.close()


if __name__ == '__main__':
    main()

import os
import argparse
import sys
import time
import datetime
import codecs
import logging
import shutil
import torch.nn.functional
from tqdm import tqdm
from medpy import metric
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

# internal imports
from utils import util
from utils.sliding_window_inference import test_single_case
from utils.losses import multi_class_dice_loss
from regpro import RegPro, ToTensor
from utils.data_augmentation import *
from network.vnet import VNet  # residual block


# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default='vnet_dice_mr_e-3lr_100e', help='experiment name used for save')
parser.add_argument("--fold_index", type=int, default=0, help="index of fold validation")
parser.add_argument("--gpu", type=str, default='1', help="gpu id")
parser.add_argument("--batch_size", type=int, default='5', help="batch size")
parser.add_argument("--seed1", type=int, default='2024', help="seed 1")
parser.add_argument("--max_iterations", type=int, dest="max_iterations", default=1200,
                    help="number of iterations of training")
parser.add_argument("--lr", type=float, dest="lr", default=0.001, help="adam: learning rate")
parser.add_argument("--n_save_iter", type=int, dest="n_save_iter", default=12, help="Save the model every time")
parser.add_argument("--img_path", type=str, default='/image_path', help='dataset root path')
parser.add_argument("--model_dir_root_path", type=str, dest="model_dir_root_path", default=r'./work_dir',
                    help="root path to save the model")
parser.add_argument("--note", type=str, dest="note", default="baseline", help="note")
arg = parser.parse_args()


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def train(exp_name: str,
          fold_index,
          gpu,
          batch_size,
          seed1,
          max_iterations,
          lr,
          n_save_iter,
          img_path,
          model_dir_root_path: str,
          note):
    """
    :param exp_name: experiment name
    :param fold_index: index of cross validation
    :param gpu: gpu id
    :param batch_size: batch size
    :param seed1: seed 1
    :param max_iterations: number of training iterations
    :param lr: learning rate
    :param n_save_iter: Determines how many epochs before saving model version
    :param img_path: dataset root path
    :param fold_root_path: fold root path
    :param model_dir_root_path: the model directory root path to save to
    :param note:
    :return:
    """

    """ setting """
    # gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(seed1)
    np.random.seed(seed1)
    torch.manual_seed(seed1)
    torch.cuda.manual_seed(seed1)

    # time
    now = time.localtime()
    now_format = time.strftime("%Y-%m-%d %H:%M:%S", now)  # time format
    date_now = now_format.split(' ')[0]
    time_now = now_format.split(' ')[1]

    # save model path
    save_path = os.path.join(model_dir_root_path, exp_name)  # ./work_dir/xxx
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # print setting
    print("----------------------------------setting-------------------------------------")
    print("lr:%f" % lr)
    print("path of saving model:%s" % save_path)
    print("data root path:%s" % img_path)
    print("----------------------------------setting-------------------------------------")

    # save parameters to TXT.
    parameter_dict = {"fold": fold_index,
                      "img_path": img_path,
                      "seed": seed1,
                      "batch size": batch_size,
                      "lr": lr,
                      "save_path": save_path,
                      'note': note}
    txt_name = 'parameter_log.txt'
    path = os.path.join(save_path, txt_name)
    with codecs.open(path, mode='a', encoding='utf-8') as file_txt:
        for key, value in parameter_dict.items():
            file_txt.write(str(key) + ':' + str(value) + '\n')

    # save this .py
    py_path_old = sys.argv[0]
    py_path_new = os.path.join(save_path, os.path.basename(py_path_old))
    shutil.copy(py_path_old, py_path_new)

    # logging
    logging.basicConfig(filename=os.path.join(model_dir_root_path, exp_name, 'log.txt'), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(parameter_dict)

    # tensorboardX
    writer = SummaryWriter(log_dir=save_path)

    # label_dict
    num_classes = 2
    label_list = [i for i in range(num_classes)]

    # patch size
    patch_size = (128, 128, 128)

    """ data generator """
    train_dataset = RegPro(img_path, transform=transforms.Compose([ToTensor()]))
    eval_dataset = RegPro(img_path.replace('train', 'val'))

    # dataloader
    def worker_init_fn(worker_id):
        random.seed(seed1 + worker_id)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True, worker_init_fn=worker_init_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=None, shuffle=False)

    """ model, optimizer, loss """
    model = VNet(n_channels=1, n_classes=num_classes, has_dropout=True, normalization='instancenorm').cuda()  # 'batchnorm'

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion1 = nn.CrossEntropyLoss()
    criterion2 = multi_class_dice_loss

    def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
        return initial_lr * (1 - epoch / max_epochs) ** exponent

    """ training loop """
    n_total_iter = 0
    max_epoch = max_iterations // len(train_dataloader)
    best_eval_dice = 0
    model.train()

    for epoch in range(max_epoch):
        lr_ = poly_lr(epoch, max_epoch, lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        for i_batch, sampled_batch in enumerate(train_dataloader):
            # start_time
            start = time.time()

            # generate moving data
            volume_batch = sampled_batch['volume'].to('cuda').float()
            seg_batch = sampled_batch['label'].to('cuda').float()

            # ------------------
            #    Train model
            # ------------------
            # zeros the parameter gradients
            optimizer.zero_grad()

            # run 3D U-Net model
            seg_output = model(volume_batch)

            # Calculate loss

            seg_batch_one_hot = util.onehot(seg_batch, label_list)
            seg_output_softmax = torch.nn.functional.softmax(seg_output, dim=1)
            seg_loss_dice = criterion2(seg_output_softmax, seg_batch_one_hot)

            loss = seg_loss_dice  # only dice loss(need to calculate class 0)

            # backwards and optimize
            loss.backward()
            optimizer.step()

            # ---------------------
            #     Print log
            # ---------------------
            n_total_iter += 1
            # Determine approximate time left
            end = time.time()
            iter_left = (max_epoch - epoch) * (len(train_dataloader) - i_batch)
            time_left = datetime.timedelta(seconds=iter_left * (end - start))
            used_time = datetime.timedelta(seconds=(end - start)).microseconds // 1000

            # print log
            logging.info("[Epoch: %4d/%d] [n_total_iter: %5d] [Train index: %2d/%d] "
                         "[loss: %f] [used time: %sms] [ETA: %s]"
                         % (epoch + 1, max_epoch, n_total_iter, i_batch + 1,
                            len(train_dataloader), loss.item(), used_time, time_left))

            # tensorboardX log writer
            writer.add_scalar("loss/Total", loss.item(), global_step=n_total_iter)
            # writer.add_scalar("loss/CE", seg_loss_ce.item(), global_step=n_total_iter)
            writer.add_scalar("loss/DICE", seg_loss_dice.item(), global_step=n_total_iter)
            writer.add_scalar("lr", lr_, global_step=n_total_iter)

            if n_total_iter % 100 == 0:
                image = seg_batch[0, 0:1, :, :, ::2].permute(3, 0, 2, 1).repeat(1, 3, 1, 1)  # [1, 0:1, :, :, ::2]
                grid_image = make_grid(image, 8, normalize=False)
                writer.add_image('Groundtruth', grid_image, n_total_iter)

                image = torch.argmax(seg_output, dim=1)[0:1, :, :, ::2].permute(3, 0, 2, 1).repeat(1, 3, 1, 1)  # [1:2, :, :, ::2] 1:2 to keep the dim
                grid_image = make_grid(image, 8, normalize=False)
                writer.add_image('Prediction', grid_image, n_total_iter)

                image = volume_batch[0, 0:1, :, :, ::2].permute(3, 0, 2, 1).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 8, normalize=True)
                writer.add_image('Image', grid_image, n_total_iter)

            # validate
            if n_total_iter % n_save_iter == 0:
                model.eval()
                logging.info('evaluating:')
                eval_dice_score = 0
                eval_pre_score = 0
                eval_sen_score = 0
                dice_scores = {'DICE ' + str(i): [] for i in range(1, num_classes)}
                for eval_sample in tqdm(eval_dataloader):
                    eval_input = eval_sample['volume'].cpu().detach().numpy()
                    eval_label = eval_sample['label'].cpu().detach().numpy()
                    pred, score_map = test_single_case(model, eval_input,
                                                       int(patch_size[0] // 1.5), int(patch_size[1] // 1.5),
                                                       int(patch_size[2] // 1.5), patch_size,
                                                       num_classes=num_classes)

                    for class_idx in range(1, num_classes):
                        pred_class = (pred == class_idx).astype(np.uint8)
                        eval_label_class = (eval_label == class_idx).astype(np.uint8)
                        if np.sum(eval_label_class) != 0:
                            dice_scores['DICE ' + str(class_idx)].append(dice_coefficient(pred_class, eval_label_class))

                    # single_case_dc = multi_class_dice(pred, eval_label, num_classes=num_classes)
                    single_case_dc = metric.binary.dc(pred, eval_label)
                    single_case_pre = metric.binary.precision(pred, eval_label)
                    single_case_sen = metric.binary.sensitivity(pred, eval_label)

                    eval_dice_score += single_case_dc
                    eval_pre_score += single_case_pre
                    eval_sen_score += single_case_sen

                eval_dice_score /= len(eval_dataloader)
                eval_pre_score /= len(eval_dataloader)
                eval_sen_score /= len(eval_dataloader)
                logging.info("evaluation result: Dice=%.4f Precision=%.4f Sensitivity=%.4f" %
                             (eval_dice_score, eval_pre_score, eval_sen_score))
                writer.add_scalar('eval_result/Dice', eval_dice_score, global_step=n_total_iter)
                writer.add_scalar('eval_result/Precision', eval_pre_score, global_step=n_total_iter)
                writer.add_scalar('eval_result/Sensitivity', eval_sen_score, global_step=n_total_iter)
                model.train()

                if eval_dice_score > best_eval_dice and n_total_iter > n_save_iter * 50:
                    best_eval_iter = n_total_iter
                    best_eval_dice = eval_dice_score
                    torch.save(model.state_dict(), "%s/model_%diter.pth" % (save_path, n_total_iter))
                    logging.info("saving best model -- iteration number:%d" % best_eval_iter)
                    writer.add_scalar('best_model', best_eval_dice, best_eval_iter)

            if n_total_iter >= max_iterations:
                break

        if n_total_iter >= max_iterations:
            break
    # save model
    torch.save(model.state_dict(), "%s/model_%d.pth" % (save_path, n_total_iter))
    logging.info("save model : %s/model_%d.pth" % (save_path, n_total_iter))
    writer.close()


if __name__ == "__main__":
    train(**vars(arg))

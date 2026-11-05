import os
import time
from datetime import datetime
import argparse
import json
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from spikingjelly.activation_based import functional
from spikingjelly.activation_based import surrogate
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from utils.cifar_dvs_util import CIFAR10DVS
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchtoolbox.transform import Cutout
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt

from pathlib import Path
from model import spiking_resnet, spiking_resnet19, spiking_vggnet, vgg_model
from spiking_neuron.CLIF import CLIFNeuron
from spiking_neuron.TLIF import TLIFNeuron, TLIFNeuron_S, TLIFNeuron_T
from spiking_neuron.ILIF import ILIFNeuron
from spiking_neuron.XLIF import XLIFNeuron
from spiking_neuron.LIF import LIFNeuron
from spiking_neuron.ReLU import ReLU
from utils.base_util import *
from utils import cifar_util
from utils import cifar_dvs_aug
from utils import augment
from utils.data_loaders import TinyImageNet


def train(args, model, train_data_loader, optimizer, epoch, num_classes, device, scaler=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_data_loader),
        [batch_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}/{args.epochs}] Batch: ")

    model.train()
    end = time.time()
    for i, (frame, label) in enumerate(train_data_loader):
        if args.dataset != 'cifar10_dvs':
            frame = frame.to(device)
            if args.dataset == 'dvs_gesture':
                frame = frame.transpose(0, 1)

        t_step = args.T
        label = label.to(device)
        label_real = torch.cat([label for _ in range(t_step)], 0)

        out_all = []
        total_fr = 0.0
        for t in range(t_step):
            if args.dataset == 'cifar10_dvs':
                input_frame = frame[t].to(device)
            elif args.dataset == 'dvs_gesture':
                input_frame = frame[t]
            else:
                input_frame = frame

            if args.amp:
                with amp.autocast():
                    out_fr = model(input_frame)
                    total_fr += out_fr.clone().detach()
                    out_all.append(out_fr)
            else:
                out_fr = model(input_frame)
                total_fr += out_fr.clone().detach()
                out_all.append(out_fr)

        out_all = torch.cat(out_all, 0)

        if args.amp:
            with amp.autocast():
                if args.loss_lambda > 0.0:  # the loss is a cross entropy term plus a mse term
                    if args.mse_n_reg:  # the mse term is not treated as a regularizer
                        label_one_hot = F.one_hot(label_real, num_classes).float()
                    else:
                        label_one_hot = torch.zeros_like(out_all).fill_(args.loss_means).to(out_all.device)
                    mse_loss = F.mse_loss(out_all, label_one_hot)
                    loss = ((1 - args.loss_lambda) * F.cross_entropy(out_all, label_real,
                                                                     label_smoothing=args.label_smoothing) + args.loss_lambda * mse_loss)
                else:  # the loss is just a cross entropy term
                    loss = F.cross_entropy(out_all, label_real, label_smoothing=args.label_smoothing)
        else:
            if args.loss_lambda > 0.0:  # the loss is a cross entropy term plus a mse term
                if args.mse_n_reg:  # the mse term is not treated as a regularizer
                    label_one_hot = F.one_hot(label_real, num_classes).float()
                else:
                    label_one_hot = torch.zeros_like(out_all).fill_(args.loss_means).to(out_all.device)
                mse_loss = F.mse_loss(out_all, label_one_hot)
                loss = ((1 - args.loss_lambda) * F.cross_entropy(out_all, label_real,
                                                                 label_smoothing=args.label_smoothing) + args.loss_lambda * mse_loss)
            else:  # the loss is just a cross entropy term
                loss = F.cross_entropy(out_all, label_real, label_smoothing=args.label_smoothing)

        optimizer.zero_grad()

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # measure accuracy and record loss
        if args.cutupmix_auto and len(label.shape) > 1:
            label = label.argmax(dim=-1)

        prec1, prec5 = accuracy(total_fr.data, label.data, topk=(1, 5))
        losses.update(loss.item(), input_frame.size(0))
        top1.update(prec1.item(), input_frame.size(0))
        top5.update(prec5.item(), input_frame.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i+1)

        functional.reset_net(model)

    logging.info('Train Epoch: [{}/{}], lr: {:.6f}, top1: {:.4f}'.format(epoch, args.epochs,
                                                                         optimizer.param_groups[0]['lr'], top1.avg))

    return top1.avg, losses.avg


def test(args, model, test_data_loader, num_classes, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (frame, label) in enumerate(test_data_loader):
            if args.dataset != 'cifar10_dvs':
                frame = frame.to(device)
                if args.dataset == 'dvs_gesture':
                    frame = frame.transpose(0, 1)

            t_step = args.T
            label = label.to(device)
            label_real = torch.cat([label for _ in range(t_step)], 0)

            out_all = []
            total_fr = 0.0
            for t in range(t_step):
                if args.dataset == 'cifar10_dvs':
                    input_frame = frame[t].to(device)
                elif args.dataset == 'dvs_gesture':
                    input_frame = frame[t]
                else:
                    input_frame = frame

                out_fr = model(input_frame)
                total_fr += out_fr.clone().detach()
                out_all.append(out_fr)

            out_all = torch.cat(out_all, 0)

            if args.loss_lambda > 0.0:  # the loss is a cross entropy term plus a mse term
                if args.mse_n_reg:  # the mse term is not treated as a regularizer
                    label_one_hot = F.one_hot(label_real, num_classes).float()
                else:
                    label_one_hot = torch.zeros_like(out_all).fill_(args.loss_means).to(out_all.device)
                mse_loss = F.mse_loss(out_all, label_one_hot)
                loss = ((1 - args.loss_lambda) * F.cross_entropy(out_all, label_real,
                                                                 label_smoothing=args.label_smoothing) + args.loss_lambda * mse_loss)
            else:  # the loss is just a cross entropy term
                loss = F.cross_entropy(out_all, label_real, label_smoothing=args.label_smoothing)

            prec1, prec5 = accuracy(total_fr.data, label.data, topk=(1, 5))

            losses.update(loss.item(), input_frame.size(0))
            top1.update(prec1.item(), input_frame.size(0))
            top5.update(prec5.item(), input_frame.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            functional.reset_net(model)

    return top1.avg, losses.avg


def plot_results(train_df, test_df, out_dir):
    """
    Args:
        train_df: pd.DataFrame
        test_df: pd.DataFrame
        out_dir: str
    """
    epochs = sorted([int(col) for col in train_df.columns])
    train_df = train_df[[str(e) for e in epochs]]
    test_df = test_df[[str(e) for e in epochs]]

    train_acc = train_df.loc['Acc@1'].values
    train_loss = train_df.loc['Loss'].values
    test_acc = test_df.loc['Acc@1'].values
    test_loss = test_df.loc['Loss'].values

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, test_acc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_plot_path = os.path.join(out_dir, 'accuracy_plot.png')
    plt.savefig(acc_plot_path)
    plt.close()
    logging.info(f"Accuracy plot saved to {acc_plot_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(out_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    plt.close()
    logging.info(f"Loss plot saved to {loss_plot_path}")


def main():
    parser = argparse.ArgumentParser(description='SNN training')
    parser.add_argument('-seed', default=1000, type=int)
    parser.add_argument('-cuda', default='cuda:0', type=str, help='cuda device to use')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')
    parser.add_argument('-data_dir', type=str, default='./data', help='directory of the used dataset')
    parser.add_argument('-dataset', default='cifar10', type=str, help='should be cifar10, cifar100, cifar10_dvs, dvs_gesture, or tiny_imagenet')
    parser.add_argument('-cutupmix_auto', action='store_true', help='cutupmix autoaugmentation for cifar and tinyimagenet')
    parser.add_argument('-out_dir', default='', type=str, help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-print-freq', default=1, type=int, metavar='N', help='print frequency (default: 1)')
    # options for optimizer
    parser.add_argument('-opt', default='SGD', type=str, help='use which optimizer. SGD or AdamW or Adam')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('-step_size', default=100, type=float, help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('-T_max', default=200, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('-weight_decay', type=float, default=5e-5)
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    # options for loss function
    parser.add_argument('-loss_lambda', type=float, default=0.05, help='the scaling factor for the MSE term in the loss')
    parser.add_argument('-mse_n_reg', action='store_true', help='loss function setting')
    parser.add_argument('-loss_means', type=float, default=1.0, help='used in the loss function when mse_n_reg=False')
    parser.add_argument('-label_smoothing', type=float, default=0.0, help='label_smoothing for cross entropy')
    # options for snns
    parser.add_argument('-T', default=4, type=int, help='simulating time-steps')
    parser.add_argument('-v_threshold', default=1.0, type=float, help='')
    parser.add_argument('-detach_reset', action='store_true', default=False, help='')
    parser.add_argument('-surrogate', default='rectangle', type=str,
                        help='used surrogate function. should be gaussian, sigmoid, atan, rectangle, or triangle')
    parser.add_argument('-neuron_node', type=str, default='TLIF', help='save the initialization of parameters')
    parser.add_argument('-model', type=str, default='spiking_resnet18', help='use which SNN model：spiking_vggnet-, spiking_resnet-')
    parser.add_argument('-drop_rate', type=float, default=0.0, help='dropout rate')

    args = parser.parse_args()

    seed_setting(args.seed)

    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    print(device)

    # =============== output setting ===============

    if args.out_dir == '':
        args.out_dir = './cs-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  # ./cs-YYYY-MM-DD-HH-MM-SS

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    setup_logging(os.path.join(args.out_dir, "log-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".txt"))

    with open(os.path.join(args.out_dir, 'args.json'), 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    logging.info(str(args))

    # =============== data loading ===============

    in_dim = None
    c_in = None

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        c_in = 3
        if args.dataset == 'cifar10':
            dataloader = torchvision.datasets.CIFAR10
            num_classes = 10
            normalization_mean = (0.4914, 0.4822, 0.4465)
            normalization_std = (0.2023, 0.1994, 0.2010)
        elif args.dataset == 'cifar100':
            dataloader = torchvision.datasets.CIFAR100
            num_classes = 100
            normalization_mean = (0.5071, 0.4867, 0.4408)
            normalization_std = (0.2675, 0.2565, 0.2761)
        else:
            raise NotImplementedError

        if args.cutupmix_auto:
            mixup_transforms = []
            mixup_transforms.append(cifar_util.RandomMixup(num_classes, p=1., alpha=0.2))
            mixup_transforms.append(cifar_util.RandomCutmix(num_classes, p=1., alpha=1.))
            mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
            collate_fn = lambda batch: mixupcutmix(*default_collate(batch))

            transform_train = cifar_util.ClassificationPresetTrain(mean=normalization_mean,
                                                                   std=normalization_std,
                                                                   interpolation=InterpolationMode('bilinear'),
                                                                   auto_augment_policy='ra',
                                                                   random_erase_prob=0.1)
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(normalization_mean, normalization_std),
            ])

            train_set = dataloader(root=args.data_dir, train=True, download=True, transform=transform_train)
            test_set = dataloader(root=args.data_dir, train=False, download=True, transform=transform_test)
            train_data_loader = DataLoader(dataset=train_set, batch_size=args.b, collate_fn=collate_fn,
                                           shuffle=True, drop_last=True, num_workers=args.j, pin_memory=True)
            test_data_loader = DataLoader(dataset=test_set, batch_size=args.b, shuffle=False, drop_last=False,
                                          num_workers=args.j, pin_memory=True)

        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                Cutout(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(normalization_mean, normalization_std),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(normalization_mean, normalization_std),
            ])

            train_set = dataloader(root=args.data_dir, train=True, download=True, transform=transform_train)
            test_set = dataloader(root=args.data_dir, train=False, download=True, transform=transform_test)
            train_data_loader = DataLoader(dataset=train_set, batch_size=args.b, shuffle=True, drop_last=True,
                                           num_workers=args.j, pin_memory=True)
            test_data_loader = DataLoader(dataset=test_set, batch_size=args.b, shuffle=False, drop_last=False,
                                          num_workers=args.j, pin_memory=True)

    elif args.dataset == 'tiny_imagenet':
        c_in = 3
        num_classes = 200
        normalize = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])  #  std=[0.2770, 0.2691, 0.2821]

        transoform_list = [
            transforms.RandomCrop(64, padding=8, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
        ]

        if args.cutupmix_auto:
            transoform_list.append(autoaugment.TrivialAugmentWide())

        transoform_list += [transforms.ToTensor(), normalize]

        transforms_train = transforms.Compose(transoform_list)
        transforms_test = transforms.Compose([transforms.ToTensor(), normalize,])

        train_data = TinyImageNet(os.path.join(args.data_dir, 'tiny-imagenet-200'), train=True, transform=transforms_train)
        test_data = TinyImageNet(os.path.join(args.data_dir, 'tiny-imagenet-200'), train=False, transform=transforms_test)

        train_data_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.b, shuffle=True,
            num_workers=args.j, pin_memory=True)

        test_data_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.b, shuffle=False,
            num_workers=args.j, pin_memory=True)

    elif args.dataset == 'cifar10_dvs':
        c_in = 2
        num_classes = 10

        transform_train = transforms.Compose([
            cifar_dvs_aug.ToPILImage(),
            cifar_dvs_aug.Resize(48),
            cifar_dvs_aug.RandomCrop(48, 4),
            cifar_dvs_aug.ToTensor(),
        ])

        transform_test = transforms.Compose([
            cifar_dvs_aug.ToPILImage(),
            cifar_dvs_aug.Resize(48),
            cifar_dvs_aug.ToTensor(),
        ])

        train_set = CIFAR10DVS(os.path.join(args.data_dir, 'cifar10_dvs'), train=True, use_frame=True, frames_num=args.T, split_by='number',
                              normalization=None, transform=transform_train)
        test_set = CIFAR10DVS(os.path.join(args.data_dir, 'cifar10_dvs'), train=False, use_frame=True, frames_num=args.T, split_by='number',
                             normalization=None, transform=transform_test)

        train_data_loader = DataLoader(train_set, batch_size=args.b, shuffle=True, num_workers=args.j)
        test_data_loader = DataLoader(test_set, batch_size=args.b, shuffle=False, num_workers=args.j)

    elif args.dataset == 'dvs_gesture':
        c_in = 2
        num_classes = 11
        in_dim = 128

        train_set = DVS128Gesture(os.path.join(args.data_dir, 'dvs_gesture'), train=True, data_type='frame', frames_number=args.T, split_by='number')
        train_data_loader = DataLoader(train_set, batch_size=args.b, shuffle=True, num_workers=args.j, drop_last=True, pin_memory=True)

        test_set = DVS128Gesture(os.path.join(args.data_dir, 'dvs_gesture'), train=False, data_type='frame', frames_number=args.T, split_by='number')
        test_data_loader = DataLoader(test_set, batch_size=args.b, shuffle=False, num_workers=args.j, drop_last=False, pin_memory=True)

    else:
        raise NotImplementedError

    # =============== model preparing ===============

    if args.surrogate == 'triangle':
        from surrogate import Triangle as surrogate_function
    elif args.surrogate == 'rectangle':
        from surrogate import Rectangle as surrogate_function
    elif args.surrogate == 'sigmoid':
        from surrogate import Sigmoid as surrogate_function
    elif args.surrogate =='atan':
        from surrogate import Atan as surrogate_function
    elif args.surrogate == 'gaussian':
        from surrogate import Gaussian as surrogate_function
    else:
        raise NotImplementedError

    if args.neuron_node == 'TLIF':
        neuron_node = TLIFNeuron
    elif args.neuron_node == 'TLIF-S':
        neuron_node = TLIFNeuron_S
    elif args.neuron_node == 'TLIF-T':
        neuron_node = TLIFNeuron_T
    elif args.neuron_node == 'CLIF':
        neuron_node = CLIFNeuron
    elif args.neuron_node == 'ILIF':
        neuron_node = ILIFNeuron
    elif args.neuron_node == 'XLIF':
        neuron_node = XLIFNeuron
    elif args.neuron_node == 'LIF':
        neuron_node = LIFNeuron
    elif args.neuron_node == 'ReLU':
        neuron_node = ReLU
    else:
        raise NotImplementedError

    if args.model in ['spiking_resnet18', 'spiking_resnet34', 'spiking_resnet50', 'spiking_resnet101', 'spiking_resnet152']:
        net = spiking_resnet.__dict__[args.model](neuron=neuron_node, num_classes=num_classes,
                                                  neuron_dropout=args.drop_rate, c_in=c_in,
                                                  v_threshold=args.v_threshold,
                                                  surrogate_function=surrogate_function.apply,
                                                  detach_reset=args.detach_reset)
    elif args.model in ['spiking_resnet19']:
        net = spiking_resnet19.__dict__[args.model](neuron=neuron_node, num_classes=num_classes,
                                             neuron_dropout=args.drop_rate, c_in=c_in,
                                             v_threshold=args.v_threshold,
                                             surrogate_function=surrogate_function.apply,
                                             detach_reset=args.detach_reset)
    elif args.model in ['spiking_vggnet11', 'spiking_vggnet13', 'spiking_vggnet16', 'spiking_vggnet19']:
        net = spiking_vggnet.__dict__[args.model](neuron=neuron_node, num_classes=num_classes,
                                                  neuron_dropout=args.drop_rate, c_in=c_in,
                                                  dataset=args.dataset,
                                                  v_threshold=args.v_threshold,
                                                  surrogate_function=surrogate_function.apply,
                                                  detach_reset=args.detach_reset)
    elif args.model in ['vggsnn']:
        net = vgg_model.__dict__[args.model](neuron=neuron_node, num_classes=num_classes,
                                             neuron_dropout=args.drop_rate, c_in=c_in,
                                             v_threshold=args.v_threshold,
                                             surrogate_function=surrogate_function.apply,
                                             detach_reset=args.detach_reset)

    else:
        raise NotImplementedError

    logging.info(str(net))
    _ = count_parameters(net)
    net.to(device)
    print(args.model)

    # =============== optimizer preparing ===============

    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'CosALR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
    else:
        raise NotImplementedError(args.lr_scheduler)

    # =============== loading models from checkpoint ===============

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    best_test_acc = 0.0
    if args.resume:
        print('resuming...')
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']
        print('start epoch: ', start_epoch, ', best acc: ', best_test_acc)


    # =============== training and testing ===============

    train_res = pd.DataFrame(index=['Acc@1', 'Loss'])
    test_res = pd.DataFrame(index=['Acc@1', 'Loss'])
    for epoch in range(start_epoch, args.epochs):
        epoch += 1
        flag = False

        train_acc, train_loss = train(args, net, train_data_loader, optimizer, epoch, num_classes, device, scaler=scaler)
        test_acc, test_loss = test(args, net, test_data_loader, num_classes, device)
        if optimizer.param_groups[0]['lr'] > 5e-5:
            lr_scheduler.step()
        # lr_scheduler.step()
        train_res[str(epoch)] = [round(train_acc, 4), round(train_loss, 4)]
        test_res[str(epoch)] = [round(test_acc, 4), round(test_loss, 4)]

        if test_acc > best_test_acc:
            flag = True
            best_test_acc = test_acc

        logging.info('Test Epoch: [{}/{}], test acc: {:.4f}, current best acc: {:.4f}'.format(epoch, args.epochs, test_acc, best_test_acc))

        train_res.to_csv(os.path.join(args.out_dir, 'train_res.csv'), index=True)
        test_res.to_csv(os.path.join(args.out_dir, 'test_res.csv'), index=True)

        save_checkpoint({
            'epoch': epoch + 1,
            'best_test_acc': best_test_acc,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }, is_best=flag, dirname=args.out_dir, filename='checkpoint.pth')

    plot_results(train_res, test_res, args.out_dir)

if __name__ == '__main__':
    main()

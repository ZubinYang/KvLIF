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

from pathlib import Path
from model import spiking_resnet, spiking_vggnet
from spiking_neuron.CLIF import CLIFNeuron
from spiking_neuron.TLIF import TLIFNeuron
from spiking_neuron.ILIF import ILIFNeuron
from spiking_neuron.XLIF import XLIFNeuron
from spiking_neuron.LIF import LIFNeuron
from spiking_neuron.ReLU import ReLU
from utils.base_util import *
from utils import cifar_util
from utils import cifar_dvs_aug
from utils.data_loaders import TinyImageNet

import csv

spike_counters = {}
def get_spike_hook(name):
    def hook(module, input, output):
        spikes = output
        if isinstance(output, tuple):
            spikes = output[0]

        num_spikes = torch.sum(spikes > 0).item()
        num_neurons = spikes.numel()

        if name not in spike_counters:
            spike_counters[name] = {'spikes': 0, 'total_neurons': 0}

        spike_counters[name]['spikes'] += num_spikes
        spike_counters[name]['total_neurons'] += num_neurons

    return hook


def inference(args, model, test_data_loader, device):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')

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

            prec1 = accuracy(total_fr.data, label.data)
            top1.update(prec1[0].item(), input_frame.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            functional.reset_net(model)

    return top1.avg


def main():
    parser = argparse.ArgumentParser(description='SNN inference')
    parser.add_argument('-seed', default=1000, type=int)
    parser.add_argument('-cuda', default='cuda:0', type=str, help='cuda device to use')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-j', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')
    parser.add_argument('-data_dir', type=str, default='./data', help='directory of the used dataset')
    parser.add_argument('-dataset', default='cifar10', type=str, help='should be cifar10, cifar100, cifar10_dvs, dvs_gesture, or tiny_imagenet')
    parser.add_argument('-out_dir', default='', type=str, help='root dir for saving logs and results')
    parser.add_argument('-weights', type=str, help='path to the trained model weights')
    parser.add_argument('-print-freq', default=1, type=int, metavar='N', help='print frequency (default: 1)')
    # options for snns
    parser.add_argument('-T', default=6, type=int, help='simulating time-steps')
    parser.add_argument('-v_threshold', default=1.0, type=float, help='')
    parser.add_argument('-detach_reset', action='store_true', default=False, help='')
    parser.add_argument('-surrogate', default='rectangle', type=str,
                        help='used surrogate function. should be gaussian, sigmoid, atan, rectangle, or triangle')
    parser.add_argument('-neuron_node', type=str, default='LIF', help='neuron type')
    parser.add_argument('-model', type=str, default='spiking_resnet18', help='use which SNN model：spiking_vggnet-, spiking_resnet-')
    parser.add_argument('-drop_rate', type=float, default=0.0, help='dropout rate')

    args = parser.parse_args()

    seed_setting(args.seed)

    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # =============== output setting ===============

    if args.out_dir == '':
        args.out_dir = './infer-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    setup_logging(os.path.join(args.out_dir, "log-infer-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".txt"))

    with open(os.path.join(args.out_dir, 'infer_args.json'), 'w') as fid:
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

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalization_mean, normalization_std),
        ])

        test_set = dataloader(root=args.data_dir, train=False, download=True, transform=transform_test)
        test_data_loader = DataLoader(dataset=test_set, batch_size=args.b, shuffle=False, drop_last=False,
                                      num_workers=args.j, pin_memory=True)

    elif args.dataset == 'tiny_imagenet':
        c_in = 3
        num_classes = 200
        normalize = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])

        transforms_test = transforms.Compose([transforms.ToTensor(), normalize,])

        test_data = TinyImageNet(os.path.join(args.data_dir, 'tiny-imagenet-200'), train=False, transform=transforms_test)
        test_data_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.b, shuffle=False,
            num_workers=args.j, pin_memory=True)

    elif args.dataset == 'cifar10_dvs':
        c_in = 2
        num_classes = 10

        transform_test = transforms.Compose([
            cifar_dvs_aug.ToPILImage(),
            cifar_dvs_aug.Resize(48),
            cifar_dvs_aug.ToTensor(),
        ])

        test_set = CIFAR10DVS(os.path.join(args.data_dir, 'cifar10_dvs'), train=False, use_frame=True, frames_num=args.T, split_by='number',
                              normalization=None, transform=transform_test)
        test_data_loader = DataLoader(test_set, batch_size=args.b, shuffle=False, num_workers=args.j)

    elif args.dataset == 'dvs_gesture':
        c_in = 2
        num_classes = 11

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
    elif args.surrogate == 'atan':
        from surrogate import Atan as surrogate_function
    elif args.surrogate == 'gaussian':
        from surrogate import Gaussian as surrogate_function
    else:
        raise NotImplementedError

    if args.neuron_node == 'TLIF':
        neuron_node = TLIFNeuron
    elif args.neuron_node == 'CLIF':
        neuron_node = CLIFNeuron
    elif args.neuron_node == 'XLIF':
        neuron_node = XLIFNeuron
    elif args.neuron_node == 'ILIF':
        neuron_node = ILIFNeuron
    elif args.neuron_node == 'LIF':
        neuron_node = LIFNeuron
    elif args.neuron_node == 'ReLU':
        neuron_node = ReLU
    else:
        raise NotImplementedError

    if args.model in ['spiking_resnet18', 'spiking_resnet34', 'spiking_resnet50', 'spiking_resnet101',
                      'spiking_resnet152']:
        net = spiking_resnet.__dict__[args.model](neuron=neuron_node, num_classes=num_classes,
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

    else:
        raise NotImplementedError

    logging.info(str(net))
    _ = count_parameters(net)
    net.to(device)
    print(args.model)

    # =============== loading model weights ===============

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights file not found: {args.weights}")

    checkpoint = torch.load(args.weights, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    net.load_state_dict(state_dict)
    logging.info("Model weights loaded successfully!")

    best_acc = checkpoint.get('best_test_acc', 0.0)
    logging.info(f"Trained model best accuracy: {best_acc:.4f}")

    # =============== register hooks ===============

    hook_handles = []
    for name, module in net.named_modules():
        if isinstance(module, neuron_node):
            handle = module.register_forward_hook(get_spike_hook(name))
            hook_handles.append(handle)
    print(f"Registered {len(hook_handles)} hooks on {neuron_node.__name__} layers.")

    # =============== inference ===============
    logging.info("Starting inference...")

    start_time = time.time()

    top1_acc = inference(args, net, test_data_loader, device)

    inference_time = time.time() - start_time

    for handle in hook_handles:
        handle.remove()
    print("Removed all hooks.")

    firing_rates = {}
    print("\n--- Firing Rate Calculation (over entire dataset) ---")

    for name, data in spike_counters.items():
        rate = data['spikes'] / data['total_neurons'] if data['total_neurons'] > 0 else 0
        firing_rates[name] = rate
        print(round(rate, 4))
    print("Average firing rate:", round(sum(firing_rates.values()) / len(firing_rates), 4))

    csv_file_path = os.path.join(args.out_dir, 'firing_rates.csv')
    try:
        with open(csv_file_path, 'w', newline='') as csvfile:
            fieldnames = ['layer_name', 'firing_rate']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for layer_name, rate in firing_rates.items():
                writer.writerow({'layer_name': layer_name, 'firing_rate': f"{rate:.6f}"})

        logging.info(f"Firing rates saved to: {csv_file_path}")
        print(f"Firing rates saved to CSV: {csv_file_path}")
    except Exception as e:
        logging.error(f"Failed to save firing rates to CSV: {e}")
        print(f"Error saving firing rates to CSV: {e}")

    # =============== save results ===============

    results = {
        'top1_accuracy': top1_acc,
        'inference_time_seconds': inference_time,
        'dataset': args.dataset,
        'model': args.model,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'firing_rates': firing_rates,
        'average_firing_rate': round(sum(firing_rates.values()) / len(firing_rates), 4)
    }

    with open(os.path.join(args.out_dir, 'inference_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    logging.info("\n" + "=" * 50)
    logging.info("INFERENCE RESULTS SUMMARY")
    logging.info("=" * 50)
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Top-1 Accuracy: {top1_acc:.4f}%")
    logging.info(f"Total inference time: {inference_time:.2f} seconds")


if __name__ == '__main__':
    main()
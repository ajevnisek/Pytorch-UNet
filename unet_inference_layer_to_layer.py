import argparse
import json
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate_layer_to_layer import evaluate
from unet import UNetLayer2Layer, LightweightUNetLayer2Layer, SuperLightweightUNetLayer2Layer, SuperDuperLightweightUNetLayer2Layer
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from datasets import get_dataset, DATASETS, get_num_classes
from architectures_unstructured import ARCHITECTURES, get_architecture
from classifier_utils import split_layername_to_triplet, LAYERNAMES, LAYERNAMES_TO_CHANNEL_DIM, LAYERNAMES_TO_SPATIAL_DIM, get_cifar100_resnet18_args, model_inference as classifier_inference
from train_layer_to_layer import get_activation_input, get_activation_drelu
from evaluate_layer_to_layer import evaluate_acc_metrics_layer_to_layer
FEATURES_CACHE_DICT = {}


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--unet-type', type=str, default='UNetLayer2Layer',
                        choices=['UNetLayer2Layer', 'LightweightUNetLayer2Layer',
                                 'SuperLightweightUNetLayer2Layer',
                                 'SuperDuperLightweightUNetLayer2Layer'],
                        help='unet type')
    # classifier related data:
    parser.add_argument('--classifier_name', type=str, default='resnet18_in', help='classifier name')
    parser.add_argument('--dataset_name', type=str, default='cifar100', help='dataset name')
    parser.add_argument('--classifier_checkpoint', type=str, default='checkpoints/resnet18_cifar100.pth',
                        help='classifier checkpoint path')
    # layers
    parser.add_argument('--layername_in', '-lni', type=str, default='layer1[0].alpha1',
                        choices=LAYERNAMES + ['images'],
                        help='layername input')
    parser.add_argument('--layername_out', '-lno', type=str, default='layer1[0].alpha2',
                        choices=LAYERNAMES + ['images'],
                        help='layername output')
    # artifacts
    parser.add_argument('--output-file', type=str, default='out.json')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # load unet model + checkpoint:
    n_channels = LAYERNAMES_TO_CHANNEL_DIM[args.layername_in]
    n_channels_out = LAYERNAMES_TO_CHANNEL_DIM[args.layername_out]
    out_spatial_H = out_spatial_W = LAYERNAMES_TO_SPATIAL_DIM[args.layername_out]
    if args.unet_type == 'UNetLayer2Layer':
        model = UNetLayer2Layer(n_channels=n_channels, n_classes=args.classes, n_features_out=n_channels_out,
                                out_spatial_H=out_spatial_H, out_spatial_W=out_spatial_W,
                                bilinear=args.bilinear)
    elif args.unet_type == 'LightweightUNetLayer2Layer':
        model = LightweightUNetLayer2Layer(n_channels=n_channels, n_features_out=n_channels_out,
                                           out_spatial_H=out_spatial_H, out_spatial_W=out_spatial_W, )
    elif args.unet_type == 'SuperLightweightUNetLayer2Layer':
        model = SuperLightweightUNetLayer2Layer(n_channels=n_channels, n_features_out=n_channels_out,
                                                out_spatial_H=out_spatial_H, out_spatial_W=out_spatial_W, )
    elif args.unet_type == 'SuperDuperLightweightUNetLayer2Layer':
        model = SuperDuperLightweightUNetLayer2Layer(n_channels=n_channels, n_features_out=n_channels_out,
                                                     out_spatial_H=out_spatial_H, out_spatial_W=out_spatial_W, )
    else:
        raise KeyError
    model = model.to(device)
    state_dict = torch.load(args.load, map_location=device)
    model.load_state_dict(state_dict)
    logging.info(f'Model loaded from {args.load}')
    model.to(device=device)
    # load dataset
    dataset_name = args.dataset_name
    batch_size = 128
    train_dataset = get_dataset(dataset_name, 'train')
    n_train = len(train_dataset)
    pin_memory = (dataset_name == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                              num_workers=max(1, os.cpu_count() - 2), pin_memory=pin_memory)
    test_dataset = get_dataset(dataset_name, 'test')
    n_val = len(test_dataset)
    pin_memory = (dataset_name == "imagenet")
    val_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
                            num_workers=max(1, os.cpu_count() - 2), pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=256,
                            num_workers=max(1, os.cpu_count() - 2), pin_memory=pin_memory)

    # initialize classifier
    classifier_name = args.classifier_name
    dataset_name = args.dataset_name
    classifier_checkpoint = args.classifier_checkpoint
    layername_in = args.layername_in
    layername_out = args.layername_out

    base_classifier = get_architecture(classifier_name, dataset_name, device, get_cifar100_resnet18_args())
    checkpoint = torch.load(classifier_checkpoint, map_location=device)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    base_classifier.eval()
    logging.info("Loaded the Base Classifier!")
    # Calculating the loaded model's test accuracy.
    original_acc = classifier_inference(base_classifier, test_loader, device, display=True, print_freq=1000)
    logging.info(f"Base classifier accuracy: {original_acc:.2f} [%]")
    # setting hooks:
    layer_name, block, relu_idx = split_layername_to_triplet(layername_in)
    hook_in = base_classifier.get_submodule(f"layer{layer_name}")[block].get_submodule(
        f'alpha{relu_idx}').register_forward_hook(
        get_activation_input('layername_in', FEATURES_CACHE_DICT))
    layer_name, block, relu_idx = split_layername_to_triplet(layername_out)
    hook_out = base_classifier.get_submodule(f"layer{layer_name}")[block].get_submodule(
        f'alpha{relu_idx}').register_forward_hook(
        get_activation_drelu('layername_out', FEATURES_CACHE_DICT))

    metrics = evaluate_acc_metrics_layer_to_layer(model, base_classifier, val_loader, device, args.amp, FEATURES_CACHE_DICT)
    print(metrics)
    metrics['layername_in'] = args.layername_in
    metrics['layername_out'] = args.layername_out
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

import argparse
import json
import logging
import os

import torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from unet import UNetLayer2Layer
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from datasets import get_dataset, DATASETS, get_num_classes
from architectures_unstructured import ARCHITECTURES, get_architecture
from classifier_utils import split_layername_to_triplet, LAYERNAMES, LAYERNAMES_TO_CHANNEL_DIM, LAYERNAMES_TO_SPATIAL_DIM, get_cifar100_resnet18_args, model_inference as classifier_inference
from train_layer_to_layer import get_activation_input, get_activation_drelu
from evaluate_layer_to_layer import evaluate_acc_metrics_layer_to_layer
from classifier_utils import train, test


FEATURES_CACHE_DICT = {}


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
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
    model = UNetLayer2Layer(n_channels=n_channels, n_classes=args.classes, n_features_out=n_channels_out,
                            out_spatial_H=out_spatial_H, out_spatial_W=out_spatial_W,
                            bilinear=args.bilinear)
    import ipdb; ipdb.set_trace()
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
    basic_classifier_args = get_cifar100_resnet18_args()
    base_classifier = get_architecture(classifier_name, dataset_name, device, basic_classifier_args)
    checkpoint = torch.load(classifier_checkpoint, map_location=device, )
    base_classifier.load_state_dict(checkpoint['state_dict'], strict=False)
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

    # replace drelu with unet:
    out_channel = base_classifier.layer1[0].alpha1.alphas.shape[1]
    feature_size = base_classifier.layer1[0].alpha1.alphas.shape[-1]
    from archs_unstructured.cifar_resnet import LearnableAlphaInputCacher, LearnableAlphaDReLUPrediction
    base_classifier.layer1[0].alpha1 = LearnableAlphaInputCacher(out_channel, feature_size).to(device)
    model.eval()
    # base_classifier.layer1[0].alpha2 = LearnableAlphaDReLUPrediction(
    #     out_channel, feature_size, unet=model, prev_module=base_classifier.layer1[0].alpha1).to(device)
    exec(f'base_classifier.{args.layername_out} = LearnableAlphaDReLUPrediction(out_channel, feature_size, unet=model, prev_module=base_classifier.layer1[0].alpha1).to(device)')

    base_classifier.eval()
    # Calculating the loaded model's test accuracy.
    original_acc = classifier_inference(base_classifier, test_loader, device, display=True, print_freq=1000)
    logging.info(f"Base classifier accuracy: {original_acc:.2f} [%]")
    metrics['base_classifier_original_accuracy'] = original_acc
    base_classifier.train()
    model.eval()
    for name, param in base_classifier.named_parameters():
        if 'alphas' in name:
            param.requires_grad = False
        if 'unet' in name:
            param.requires_grad = False
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = SGD(base_classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[args.epochs // 2, 3 * args.epochs // 4], last_epoch=-1)
    for epoch in range(args.epochs):
        # training
        train(train_loader, base_classifier, criterion, optimizer, epoch, device)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        _, top1, _ = test(test_loader, base_classifier, criterion, device, cur_step)
        scheduler.step()
        break
    metrics['base_classifier_after_finetuning_accuracy'] = top1
    with open(f'out/replacing_{args.layername_out}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)


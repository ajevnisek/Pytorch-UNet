import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from architectures_unstructured import ARCHITECTURES, get_architecture
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer, Adam
from torch.optim.lr_scheduler import StepLR
import datetime
import time
import numpy as np
import copy
import types
from math import ceil
from train_utils import AverageMeter, accuracy, accuracy_list, init_logfile, log
from utils import *
import sys
import matplotlib.pyplot as plt


class MyNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def split_layername_to_triplet(layername):
    layer_index = int(layername.split('layer')[-1].split('[')[0])
    block_index = int(layername.split('[')[-1].split(']')[0])
    subblock_relu_index = int(layername.split('alpha')[-1])
    return layer_index, block_index, subblock_relu_index


def get_activation_drelu(name, cache_dict):
    def hook(model, input, output):
        if name not in cache_dict:
            cache_dict[name] = [(input[0] > 0).detach().cpu().byte()]
        else:
            cache_dict[name].append((input[0] > 0).detach().cpu().byte())

    return hook


def get_drelus_for_layer(model, data_loader, layer_name, block, relu_idx):
    cache_dict = {}
    outputs_list = []

    layername = f'{layer_name}_block{block}_relu{relu_idx}'
    hook_in = model.get_submodule(layer_name)[block].get_submodule(f'alpha{relu_idx}').register_forward_hook(
        get_activation_drelu(layername, cache_dict))

    # The return value here does not matter, this inference run is just for caching purposes.
    model_inference(model, data_loader,
                                   device, display=True, outputs_list=outputs_list)
    drelus = torch.cat(cache_dict[layername], 0)
    outputs = torch.cat(outputs_list, 0)

    hook_in.remove()
    return drelus, outputs



device = torch.device("cuda")

my_args = MyNamespace(alpha=1e-05, arch='resnet18_in', batch=128, block_type='LearnableAlpha',
                      budegt_type='absolute', dataset='cifar100', epochs=2000, finetune_epochs=100,
                      gamma=0.1, gpu=0, logname='resnet18_in_unstructured_.txt', lr=0.001, lr_step_size=30,
                      momentum=0.9, num_of_neighbors=4,
                      outdir='./drelu_entropy_per_layer/cifar100/original/resnet18_in/',
                      print_freq=100, relu_budget=15000,
                      savedir='./checkpoints/resnet18_cifar100.pth', stride=1, threshold=0.01,
                      weight_decay=0.0005, workers=4)
train_dataset = get_dataset('cifar100', 'train')
test_dataset = get_dataset('cifar100', 'test')
pin_memory = (my_args.dataset == "imagenet")
train_loader = DataLoader(train_dataset, shuffle=False, batch_size=256,
                          num_workers=4, pin_memory=pin_memory)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=256,
                         num_workers=4, pin_memory=pin_memory)

# Loading the base_classifier
base_classifier = get_architecture('resnet18_in', 'cifar100', 'cuda', my_args)
checkpoint = torch.load(my_args.savedir, map_location=device)
base_classifier.load_state_dict(checkpoint['state_dict'], strict=False)

original_acc = model_inference(base_classifier, test_loader,
                               device, display=False)
print(f"Original model accuracy: {original_acc} [%]")

checkpoint = torch.load('checkpoints/resnet18_cifar100_30000_v2.pth', map_location=device)
reference_model = get_architecture('resnet18_in', 'cifar100', 'cuda', my_args)
reference_model.load_state_dict(checkpoint['state_dict'], strict=False)

reference_model_acc = model_inference(reference_model, test_loader,
                                      device, display=False)
print(f"Reference model accuracy: {reference_model_acc} [%]")

checkpoint = torch.load('checkpoints/resnet18_cifar100_onr_15000.pth', map_location=device)
onr_model = get_architecture('resnet18_in', 'cifar100', 'cuda', my_args)
onr_model.load_state_dict(checkpoint['state_dict'], strict=False)

onr_acc = model_inference(onr_model, test_loader,
                               device, display=False)
print(f"ONR model accuracy: {onr_acc} [%]")

checkpoint = torch.load('checkpoints/resnet18_cifar100_15000_snl.pth', map_location=device)
snl_at_target_model = get_architecture('resnet18_in', 'cifar100', 'cuda', my_args)
snl_at_target_model.load_state_dict(checkpoint['state_dict'], strict=False)

snl_at_target_model_acc = model_inference(snl_at_target_model, test_loader,
                                          device, display=False)
print(f"SNL at target budget accuracy: {snl_at_target_model_acc} [%]")


layer_names = ['layer1[0].alpha1', 'layer1[0].alpha2',
               'layer1[1].alpha1', 'layer1[1].alpha2',
               'layer2[0].alpha1', 'layer2[0].alpha2',
               'layer2[1].alpha1', 'layer2[1].alpha2',
               'layer3[0].alpha1', 'layer3[0].alpha2',
               'layer3[1].alpha1', 'layer3[1].alpha2',
               'layer4[0].alpha1', 'layer4[0].alpha2',
               'layer4[1].alpha1', 'layer4[1].alpha2']
reference = []
onr = []
for layername in layer_names:
    layer_index, block_index, subblock_relu_index = split_layername_to_triplet(layername)
    exec(f'prev = reference_model.get_submodule("layer{layer_index}")[{block_index}].alpha{subblock_relu_index}.alphas.sum().item()')
    exec(
        f'curr = onr_model.get_submodule("layer{layer_index}")[{block_index}].alpha{subblock_relu_index}.alphas.sum().item()')
    reference.append(prev)
    onr.append(curr)
    print(layername, int(prev), int(curr))


plt.close('all')
plt.bar([x-0.15 for x in range(len(layer_names))], reference, width=0.3, label='SNL-30K')
plt.bar([x+0.15 for x in range(len(layer_names))], onr, width=0.3, label='CD-15K')
plt.legend()
plt.title('ReLU Count in every layer: ResNet18, CIFAR-100')
plt.xticks(range(len(layer_names)), layer_names)
plt.xticks(range(len(layer_names)), layer_names, rotation='vertical')
plt.ylabel('ReLU count [#]')
plt.grid(True)
plt.tight_layout()
plt.savefig('drelu_entropy_relu_count.png')

layer_index, block_index, subblock_relu_index = 2, 0, 2
for layer in layer_names:
    layer_index, block_index, subblock_relu_index = split_layername_to_triplet(layer)
    exec(f'reference_alphas = reference_model.get_submodule("layer{layer_index}")[{block_index}].alpha{subblock_relu_index}.alphas')
    exec(f'onr_alphas = onr_model.get_submodule("layer{layer_index}")[{block_index}].alpha{subblock_relu_index}.alphas')

    original_drelus, original_outputs = get_drelus_for_layer(base_classifier, test_loader,
                                                             f"layer{layer_index}",
                                                             block_index,
                                                             subblock_relu_index)
    reference_drelus, reference_outputs = get_drelus_for_layer(reference_model, test_loader,
                                                               f"layer{layer_index}",
                                                               block_index,
                                                               subblock_relu_index)
    onr_drelus, onr_outputs = get_drelus_for_layer(onr_model, test_loader,
                                                   f"layer{layer_index}",
                                                   block_index,
                                                   subblock_relu_index)
    snl_at_target_budget_drelus, snl_at_target_budget_outputs = get_drelus_for_layer(snl_at_target_model, test_loader,
                                                                                     f"layer{layer_index}",
                                                                                     block_index,
                                                                                     subblock_relu_index)

    print(f"number of overlapping relus: {int((onr_alphas * reference_alphas).sum().item())}")
    print(f"number of relus in onr: {int((onr_alphas).sum().item())}")
    def indices_of_ones(binary_tensor):
        indices = torch.nonzero(binary_tensor == 1)
        return indices
    reference_relu_locations = indices_of_ones(reference_alphas)
    onr_relu_locations = indices_of_ones(onr_alphas)
    overlap_relu_locations = indices_of_ones(onr_alphas * reference_alphas)
    original_prob_eq_1 = []
    for location in reference_relu_locations:
        prob = original_drelus[..., location[1], location[2], location[3]].float().mean().item()
        original_prob_eq_1.append(prob)

    reference_prob_eq_1 = []
    for location in reference_relu_locations:
        prob = reference_drelus[..., location[1], location[2], location[3]].float().mean().item()
        reference_prob_eq_1.append(prob)

    selected_from_reference_eq_1 = []
    for location in overlap_relu_locations:
        prob = reference_drelus[..., location[1], location[2], location[3]].float().mean().item()
        selected_from_reference_eq_1.append(prob)

    snl_at_target_budget_prob_eq_1 = []
    for location in overlap_relu_locations:
        prob = snl_at_target_budget_drelus[..., location[1], location[2], location[3]].float().mean().item()
        snl_at_target_budget_prob_eq_1.append(prob)

    from math import sqrt, log2

    plt.close('all')
    #original_prob_eq_1
    plt.hist(original_prob_eq_1, bins=int(sqrt(len(original_prob_eq_1))), label='original', alpha=0.8)
    plt.hist(reference_prob_eq_1, bins=int(sqrt(len(reference_prob_eq_1))), label='ref', alpha=0.8)
    plt.hist(selected_from_reference_eq_1, bins=int(sqrt(len(selected_from_reference_eq_1))), label='selected by CD', alpha=0.8)
    # snl_at_target_budget_prob_eq_1
    plt.hist(snl_at_target_budget_prob_eq_1, bins=int(sqrt(len(snl_at_target_budget_prob_eq_1))), label='snl-15K', alpha=0.8)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Pr(DReLU=1)')
    plt.ylabel('count')
    plt.title(f'Histogram of Pr(DReLU=1) \n'
              f'layer{layer_index}[{block_index}].alpha{subblock_relu_index}')
    plt.tight_layout()
    plt.savefig(f'drelu_entropy_entropies/drelu_entropy_probs_layer{layer_index}[{block_index}].alpha{subblock_relu_index}.png')

    def binary_entropy(prob):
        if prob == 0 or prob == 1:
            return 0
        else:
            return -prob * log2(prob) - (1 - prob) * log2(1 - prob)

    plt.close('all')
    plt.subplot(2, 1, 1)
    plt.hist([binary_entropy(p) for p in original_prob_eq_1], color='b', bins=int(sqrt(len(original_prob_eq_1))), label='original', alpha=0.8)
    plt.hist([binary_entropy(p) for p in reference_prob_eq_1], color='g',bins=int(sqrt(len(reference_prob_eq_1))), label='ref', alpha=0.8)
    plt.hist([binary_entropy(p) for p in selected_from_reference_eq_1], color='r', bins=int(sqrt(len(selected_from_reference_eq_1))), label='selected by CD', alpha=0.8)
    # snl_at_target_budget_prob_eq_1
    plt.hist([binary_entropy(p) for p in snl_at_target_budget_prob_eq_1], color='m', bins=int(sqrt(len(snl_at_target_budget_prob_eq_1))), label='snl-15k', alpha=0.8)

    plt.legend()
    plt.grid(True)
    plt.xlabel('entropy')
    plt.ylabel('count')
    plt.suptitle('Histogram of entropy')
    plt.subplot(2, 4, 5)
    plt.hist([binary_entropy(p) for p in original_prob_eq_1], color='b', bins=int(sqrt(len(original_prob_eq_1))), label='original', alpha=0.8)
    plt.legend()
    plt.grid(True)
    plt.xlabel('entropy')
    plt.ylabel('count')
    plt.suptitle('Histogram of entropy')
    plt.subplot(2, 4, 6)
    plt.hist([binary_entropy(p) for p in reference_prob_eq_1], color='g',bins=int(sqrt(len(reference_prob_eq_1))), label='ref', alpha=0.8)
    plt.legend()
    plt.grid(True)
    plt.xlabel('entropy')
    plt.ylabel('count')
    plt.suptitle('Histogram of entropy')
    plt.subplot(2, 4, 7)
    plt.hist([binary_entropy(p) for p in selected_from_reference_eq_1], color='r', bins=int(sqrt(len(selected_from_reference_eq_1))), label='selected by CD', alpha=0.8)
    plt.legend()
    plt.grid(True)
    plt.xlabel('entropy')
    plt.ylabel('count')
    plt.suptitle('Histogram of entropy')
    plt.subplot(2, 4, 8)
    plt.hist([binary_entropy(p) for p in snl_at_target_budget_prob_eq_1], color='m', bins=int(sqrt(len(snl_at_target_budget_prob_eq_1))), label='snl-15k', alpha=0.8)
    plt.legend()
    plt.grid(True)
    plt.xlabel('entropy')
    plt.ylabel('count')
    plt.suptitle(f'Histogram of entropy \n'
                 f'layer{layer_index}[{block_index}].alpha{subblock_relu_index}')
    fig = plt.gcf()
    fig.set_size_inches((12, 6))
    plt.tight_layout()
    plt.savefig(f'drelu_entropy_entropies/drelu_entropy_entropies_layer{layer_index}[{block_index}].alpha{subblock_relu_index}.png')
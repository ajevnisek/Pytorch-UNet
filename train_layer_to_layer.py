import argparse
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
from unet import UNetLayer2Layer
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from datasets import get_dataset, DATASETS, get_num_classes
from architectures_unstructured import ARCHITECTURES, get_architecture
from classifier_utils import split_layername_to_triplet, LAYERNAMES, LAYERNAMES_TO_CHANNEL_DIM, LAYERNAMES_TO_SPATIAL_DIM, get_cifar100_resnet18_args, model_inference as classifier_inference
from evaluate_layer_to_layer import evaluate_acc_metrics_layer_to_layer


FEATURES_CACHE_DICT = {}
SUPPORTED_CRITERIONS = ['ce-loss', 'l1-loss', 'l2-loss', 'ce-loss+dice-loss']
# args. = Path('./checkpoints/UNet/')


def get_criterion(args):
    if args.criterion not in SUPPORTED_CRITERIONS:
        raise NotImplementedError()

    if args.criterion == 'ce-loss':
        return nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    if args.criterion == 'l1-loss':
        return nn.L1Loss()
    if args.criterion == 'l2-loss':
        return nn.MSELoss()


def get_activation_input(name, FEATURES_CACHE_DICT):
    def hook(model, input, output):
        FEATURES_CACHE_DICT[name] = input[0]
    return hook


def get_activation_drelu(name, FEATURES_CACHE_DICT):
    def hook(model, input, output):
        FEATURES_CACHE_DICT[name] = (input[0] > 0).byte()
    return hook


def train_model(
        model,
        device,
        classifier_name: str = 'resnet18_in',
        dataset_name: str = 'cifar100',
        classifier_checkpoint: str = 'checkpoints/resnet18_cifar100.pth',
        layername_in: str = 'layer1[0].alpha1',
        layername_out: str = 'layer1[0].alpha2',
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1 - 3. Create datasets + Create data loaders
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
    rand_val_loader = DataLoader(test_dataset, shuffle=True, batch_size=256,
                                 num_workers=max(1, os.cpu_count() - 2), pin_memory=pin_memory)


    # initialize classifier
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
    hook_in = base_classifier.get_submodule(f"layer{layer_name}")[block].get_submodule(f'alpha{relu_idx}').register_forward_hook(
        get_activation_input('layername_in', FEATURES_CACHE_DICT))
    layer_name, block, relu_idx = split_layername_to_triplet(layername_out)
    hook_out = base_classifier.get_submodule(f"layer{layer_name}")[block].get_submodule(f'alpha{relu_idx}').register_forward_hook(
        get_activation_drelu('layername_out', FEATURES_CACHE_DICT))

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must',
                            name=f"layer2layer-{args.layername_in}-to-{args.layername_out}-loss-{args.criterion}-epochs-{args.epochs}-batchsize-{args.batch_size}-lr-{args.lr}")
    experiment.config.update(vars(args))
    os.makedirs(str(Path(args.dir_checkpoint) / experiment.id), exist_ok=True)
    arguments_string = 'Experiment with:\n'
    largest_key = 0
    for k in vars(args):
        largest_key = max(len(str(k)), largest_key)
    for k, v in vars(args).items():
        arguments_string += f'{k}:' + ' ' * (largest_key - len(k)) + f' {v} \n'
    logging.info(arguments_string)
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    criterion = get_criterion(args)
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                classifier_input_images, class_labels = batch[0], batch[1]
                classifier_input_images = classifier_input_images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

                with torch.no_grad():
                    classifier_predictions = base_classifier(classifier_input_images)

                images = FEATURES_CACHE_DICT['layername_in']
                true_masks = FEATURES_CACHE_DICT['layername_out'].unsqueeze(1)

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if args.criterion == 'ce-loss+dice-loss':
                        criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
                        if model.n_classes == 1:
                            ce_loss = criterion(masks_pred, true_masks.float())
                            loss = ce_loss
                            loss += dice_loss(F.sigmoid(masks_pred), true_masks.float(), multiclass=False)
                        else:
                            ce_loss = criterion(masks_pred, true_masks)
                            loss = ce_loss
                            loss += dice_loss(
                                F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )
                    else:

                        ce_loss_criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
                        if model.n_classes == 1:
                            ce_loss = ce_loss_criterion(masks_pred, true_masks.float())
                            loss = get_criterion(args)(masks_pred, true_masks.float())
                        else:
                            ce_loss = ce_loss_criterion(masks_pred, true_masks)
                            loss = get_criterion(args)(masks_pred, true_masks.float())
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'ce loss': ce_loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        val_metrics = evaluate_acc_metrics_layer_to_layer(model, base_classifier, val_loader, device,
                                                                          amp, FEATURES_CACHE_DICT)
                        val_accuracy = val_metrics['accuracy']
                        val_score = val_metrics['dice']
                        scheduler.step(val_accuracy)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        logging.info('Validation Accuracy: {}'.format(val_accuracy))
                        try:
                            images, labels = next(iter(rand_val_loader))
                            # move images and labels to correct device and type
                            image = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                            mask_true = true_masks.to(device=device, dtype=torch.long)

                            # predict the mask
                            model.eval()
                            base_classifier.eval()
                            with torch.no_grad():
                                classifier_predictions = base_classifier(image)
                            images_to_unet = FEATURES_CACHE_DICT['layername_in']
                            true_masks = FEATURES_CACHE_DICT['layername_out'].unsqueeze(1)

                            # move images and labels to correct device and type
                            images_to_unet = images_to_unet.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                            mask_true = true_masks.to(device=device, dtype=torch.long)

                            # predict the mask
                            with torch.no_grad():
                                mask_pred = model(images_to_unet)
                            model.train()

                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice Score': val_score,
                                'validation Accuracy': val_accuracy,
                                'images': wandb.Image(image[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(mask_true[0].unsqueeze(0).float().cpu()),
                                    'pred': wandb.Image(mask_pred[0,:, :].float().cpu()),
                                    # 'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                # **histograms
                            })
                        except:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice Score': val_score,
                                'validation Accuracy': val_accuracy,
                                'step': global_step,
                                'epoch': epoch,
                            })

        if save_checkpoint:
            Path(args.dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            os.makedirs(str(Path(args.dir_checkpoint) / experiment.name ), exist_ok=True)
            torch.save(state_dict, str(Path(args.dir_checkpoint) / experiment.name / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--config-file', type=str, required=False, help='Number of epochs')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--criterion', help='training loss criterion',
                        default='l1-loss', choices=SUPPORTED_CRITERIONS)
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
    parser.add_argument('--dir_checkpoint', type=str, default='./checkpoints/UNet/',
                        help='checkpoints dir')
    # layers
    parser.add_argument('--layername_in', '-lni', type=str, default='layer1[0].alpha1',
                        choices=LAYERNAMES + ['images'],
                        help='layername input')
    parser.add_argument('--layername_out', '-lno', type=str, default='layer1[0].alpha2',
                        choices=LAYERNAMES + ['images'],
                        help='layername output')
    parser.add_argument('--layername_out_specific_channel', '-lnoc', type=int, default=0,
                        help='layername output specific channel to predict.')
    args = parser.parse_args()
    if args.config_file:
        import yaml
        args_from_file = yaml.load(open(args.config_file), Loader=yaml.FullLoader)
        for key, value in args_from_file.items():
            setattr(args, key, value)
    return args


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    n_channels = LAYERNAMES_TO_CHANNEL_DIM[args.layername_in]
    n_channels_out = LAYERNAMES_TO_CHANNEL_DIM[args.layername_out]
    out_spatial_H = out_spatial_W = LAYERNAMES_TO_SPATIAL_DIM[args.layername_out]
    model = UNetLayer2Layer(n_channels=n_channels, n_classes=args.classes, n_features_out=n_channels_out,
                            out_spatial_H=out_spatial_H, out_spatial_W=out_spatial_W,
                            bilinear=args.bilinear)
    model = model.to(device)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            classifier_name=args.classifier_name,
            dataset_name=args.dataset_name,
            classifier_checkpoint=args.classifier_checkpoint,
            layername_in=args.layername_in,
            layername_out=args.layername_out,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            classifier_name=args.classifier_name,
            dataset_name=args.dataset_name,
            classifier_checkpoint=args.classifier_checkpoint,
            layername_in=args.layername_in,
            layername_out=args.layername_out,
        )

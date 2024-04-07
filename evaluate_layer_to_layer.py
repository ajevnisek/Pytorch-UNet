import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from sklearn.metrics import precision_score, recall_score


def mul(x):
    if len(x) == 1:
        return x[0]
    return x[0] * mul(x[1:])


@torch.inference_mode()
def evaluate(net, classifier, dataloader, device, amp, FEATURES_CACHE_DICT, layername_out_specific_channel):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            classifier_input_images, class_labels = batch[0], batch[1]
            classifier_input_images = classifier_input_images.to(device=device, dtype=torch.float32,
                                                                 memory_format=torch.channels_last)

            with torch.no_grad():
                classifier_predictions = classifier(classifier_input_images)

            images = FEATURES_CACHE_DICT['layername_in']
            true_masks = FEATURES_CACHE_DICT['layername_out'][:, layername_out_specific_channel, :, :]

            # move images and labels to correct device and type
            image = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = true_masks.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred.squeeze(1), mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)



@torch.inference_mode()
def evaluate_acc_metrics(net, classifier, dataloader, device, amp, FEATURES_CACHE_DICT, layername_out_specific_channel):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = torch.tensor(0.0).to(device)
    correct = torch.tensor(0.0).to(device)
    total = torch.tensor(0.0).to(device)

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            classifier_input_images, class_labels = batch[0], batch[1]
            classifier_input_images = classifier_input_images.to(device=device, dtype=torch.float32,
                                                                 memory_format=torch.channels_last)

            with torch.no_grad():
                classifier_predictions = classifier(classifier_input_images)

            images = FEATURES_CACHE_DICT['layername_in']
            true_masks = FEATURES_CACHE_DICT['layername_out'][:, layername_out_specific_channel, :, :]

            # move images and labels to correct device and type
            image = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = true_masks.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                curr_correct = (mask_pred.squeeze(1) == mask_true).sum()
                correct += curr_correct
                total += mul(mask_true.shape)
                # compute the Dice score
                dice_score += dice_coeff(mask_pred.squeeze(1), mask_true, reduce_batch_first=False)

            else:
                assert False, 'not supported yet'
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()

    return {'dice': (dice_score / max(num_val_batches, 1)).item(), 'accuracy': (correct / total).item(),}


@torch.inference_mode()
def evaluate_confusion_metrics_layer_to_layer(net, classifier, dataloader, device, amp, FEATURES_CACHE_DICT):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = torch.tensor(0.0).to(device)
    correct = torch.tensor(0.0).to(device)
    total = torch.tensor(0.0).to(device)

    true_positives = torch.tensor(0.0).to(device)
    false_positives = torch.tensor(0.0).to(device)
    false_negatives = torch.tensor(0.0).to(device)
    true_negatives = torch.tensor(0.0).to(device)

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            classifier_input_images, class_labels = batch[0], batch[1]
            classifier_input_images = classifier_input_images.to(device=device, dtype=torch.float32,
                                                                 memory_format=torch.channels_last)

            with torch.no_grad():
                classifier_predictions = classifier(classifier_input_images)

            images = FEATURES_CACHE_DICT['layername_in']
            true_masks = FEATURES_CACHE_DICT['layername_out'].unsqueeze(1)

            # move images and labels to correct device and type
            image = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = true_masks.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                curr_correct = (mask_pred == mask_true).sum()
                correct += curr_correct
                total += mul(mask_true.shape)
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                # compute the confusion matrix
                predictions = mask_pred.flatten()
                labels = mask_true.flatten()
                for p, l in zip(predictions, labels):
                    if p == 1 and l == 1:
                        true_positives += 1
                    elif p == 1 and l == 0:
                        false_positives += 1
                    elif p == 0 and l == 1:
                        false_negatives += 1
                    elif p == 0 and l == 0:
                        true_negatives += 1

            else:
                assert False, 'not supported yet'
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    if true_positives + false_positives == 0:
        precision = torch.tensor(0)
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives == 0:
        recall = torch.tensor(0)
    else:
        recall = true_positives / (true_positives + false_negatives)
    return {'dice': (dice_score / max(num_val_batches, 1)).item(),
            'accuracy': (correct / total).item(),
            'precision': precision.item(), 'recall': recall.item(),
            'true_negatives': true_negatives.item(), 'false_positives': false_positives.item(),
            'false_negatives': false_negatives.item(), 'true_positives': true_positives.item()
            }


@torch.inference_mode()
def evaluate_acc_metrics_layer_to_layer(net, classifier, dataloader, device, amp, FEATURES_CACHE_DICT):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = torch.tensor(0.0).to(device)
    correct = torch.tensor(0.0).to(device)
    total = torch.tensor(0.0).to(device)
    l1_total = torch.tensor(0.0).to(device)
    mse_total = torch.tensor(0.0).to(device)

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            classifier_input_images, class_labels = batch[0], batch[1]
            classifier_input_images = classifier_input_images.to(device=device, dtype=torch.float32,
                                                                 memory_format=torch.channels_last)

            with torch.no_grad():
                classifier_predictions = classifier(classifier_input_images)

            images = FEATURES_CACHE_DICT['layername_in']
            true_masks = FEATURES_CACHE_DICT['layername_out'].unsqueeze(1)

            # move images and labels to correct device and type
            image = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = true_masks.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred_raw = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred_raw) > 0.5).float()
                curr_correct = (mask_pred == mask_true).sum()
                correct += curr_correct
                total += mul(mask_true.shape)
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                l1_total += nn.L1Loss()(F.sigmoid(mask_pred_raw), mask_true)
                mse_total += nn.MSELoss()(F.sigmoid(mask_pred_raw), mask_true)

            else:
                assert False, 'not supported yet'
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()

    return {'dice': (dice_score / max(num_val_batches, 1)).item(),
            'accuracy': (correct / total).item(),
            'l1-loss': (l1_total / max(num_val_batches, 1)).item(),
            'mse-loss': (mse_total / max(num_val_batches, 1)).item(),
            }


@torch.inference_mode()
def evaluate_confusion_metrics(net, classifier, dataloader, device, amp, FEATURES_CACHE_DICT, layername_out_specific_channel):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = torch.tensor(0.0).to(device)
    correct = torch.tensor(0.0).to(device)
    total = torch.tensor(0.0).to(device)

    true_positives = torch.tensor(0.0).to(device)
    false_positives = torch.tensor(0.0).to(device)
    false_negatives = torch.tensor(0.0).to(device)
    true_negatives = torch.tensor(0.0).to(device)


    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            classifier_input_images, class_labels = batch[0], batch[1]
            classifier_input_images = classifier_input_images.to(device=device, dtype=torch.float32,
                                                                 memory_format=torch.channels_last)

            with torch.no_grad():
                classifier_predictions = classifier(classifier_input_images)

            images = FEATURES_CACHE_DICT['layername_in']
            true_masks = FEATURES_CACHE_DICT['layername_out'][:, layername_out_specific_channel, :, :]

            # move images and labels to correct device and type
            image = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = true_masks.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                curr_correct = (mask_pred.squeeze(1) == mask_true).sum()
                correct += curr_correct
                total += mul(mask_true.shape)
                # compute the Dice score
                dice_score += dice_coeff(mask_pred.squeeze(1), mask_true, reduce_batch_first=False)
                # compute the confusion matrix
                predictions = mask_pred.flatten()
                labels = mask_true.flatten()
                for p, l in zip(predictions, labels):
                    if p == 1 and l == 1:
                        true_positives += 1
                    elif p == 1 and l == 0:
                        false_positives += 1
                    elif p == 0 and l == 1:
                        false_negatives += 1
                    elif p == 0 and l == 0:
                        true_negatives += 1
            else:
                assert False, 'not supported yet'
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    if true_positives + false_positives == 0:
        precision = torch.tensor(0)
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives == 0:
        recall = torch.tensor(0)
    else:
        recall = true_positives / (true_positives + false_negatives)

    return {'dice': (dice_score / max(num_val_batches, 1)).item(), 'accuracy': (correct / total).item(),
            'precision': precision.item(), 'recall': recall.item(),
            'true_negatives': true_negatives.item(), 'false_positives': false_positives.item(),
            'false_negatives': false_negatives.item(), 'true_positives': true_positives.item()}


import math
import sys
import utils
import torch
import torch.nn.functional as F
from epitope import *
from model import *

from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    f1_score, precision_score, recall_score, accuracy_score,
    brier_score_loss, log_loss
)
from typing import Iterable, Optional


def sequence_loss(pred, target, mask):
    """
    pred: [B, C, L]
    target: [B, L]
    mask: [B, L] (bool)
    """
    B, C, L = pred.shape
    pred = pred.transpose(1, 2).reshape(-1, C)      # [B*L, C]
    target = target.reshape(-1)                     # [B*L]
    mask = mask.reshape(-1)                         # [B*L], bool

    loss = F.cross_entropy(pred, target, reduction='none')  # [B*L]
    loss = loss[mask].mean()  # only valid positions
    return loss


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler=None, max_norm: float = 0,
                    model_ema: Optional[object] = None, mixup_fn=None,
                    set_training_mode=True):
    model.train(set_training_mode)
    if hasattr(criterion, 'train'):
        criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f"Epoch: [{epoch}]"
    print_freq = 10

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        samples = batch['embedding'].to(device)
        targets = batch['labels'].to(device)
        mask = batch['mask'].to(device).bool()
        samples = samples.transpose(1, 2)

        with torch.cuda.amp.autocast():
            output = model(samples)  # [B, num_classes, L]
            loss = sequence_loss(output, targets, mask)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            sys.exit(1)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, threshold=0.35):
    """
    return: results, sample_ious
    results: 13 metrics containing F1-L and F1-D
    sample_ious: AgIoU for each input sequence
    """
    model.eval()
    true_positives = 0
    union_positives = 0

    all_probs = []
    all_preds = []
    all_targets = []

    sample_ious = []

    for batch in data_loader:
        samples = batch['embedding'].to(device)       # [B, L, D]
        targets = batch['labels'].to(device)          # [B, L]
        mask = batch['mask'].to(device).bool()        # [B, L]
        samples = samples.transpose(1, 2)             # -> [B, D, L]

        with torch.cuda.amp.autocast():
            output = model(samples)                   # [B, 2, L]
            probs = torch.softmax(output, dim=1)[:, 1, :]  # [B, L]
            preds = (probs > threshold).long()
            preds = preds.masked_fill(~mask, 0)

            for i in range(samples.shape[0]):
                pred_i = preds[i]
                target_i = targets[i]
                mask_i = mask[i]

                tp_i = ((pred_i == 1) & (target_i == 1) & mask_i).sum().item()
                union_i = (((pred_i == 1) | (target_i == 1)) & mask_i).sum().item()
                iou_i = tp_i / union_i if union_i > 0 else 0.0
                sample_ious.append(iou_i)

            tp = ((preds == 1) & (targets == 1) & mask).sum().item()
            union = (((preds == 1) | (targets == 1)) & mask).sum().item()
            true_positives += tp
            union_positives += union

            all_probs.append(probs[mask].cpu())
            all_preds.append(preds[mask].cpu())
            all_targets.append(targets[mask].cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    agiou = true_positives / union_positives if union_positives > 0 else 0.0

    # ========== All metrics ==========
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = float('nan')

    try:
        pr_auc = average_precision_score(all_targets, all_probs)
    except:
        pr_auc = float('nan')

    try:
        pcc = np.corrcoef(all_probs, all_targets)[0, 1]
    except:
        pcc = float('nan')

    try:
        brier = brier_score_loss(all_targets, all_probs)
    except:
        brier = float('nan')

    try:
        bce = log_loss(all_targets, all_probs, labels=[0, 1])
    except:
        bce = float('nan')

    # ===== Linear / Discontinuous F1 =====

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    # Identify epitope positions
    is_epitope = all_targets == 1
    non_epitope_idx = all_targets == 0

    # True epitope classification
    true_linear_flags = np.array(classify_linear_epitopes(all_targets))  # full chain

    # Initialize masks
    linear_mask = np.zeros_like(all_targets, dtype=bool)
    nonlinear_mask = np.zeros_like(all_targets, dtype=bool)

    # Assign masks for true epitope positions
    linear_mask[is_epitope] = true_linear_flags[is_epitope]
    nonlinear_mask[is_epitope] = ~true_linear_flags[is_epitope]

    # Assign masks for non-epitope positions (predicted epitope)
    pred_linear_flags = np.array(classify_linear_epitopes(all_preds))
    linear_mask[non_epitope_idx] = pred_linear_flags[non_epitope_idx]
    nonlinear_mask[non_epitope_idx] = ~pred_linear_flags[non_epitope_idx]

    # Safe F1 computation
    def safe_f1(y_true, y_pred):
        if len(y_true) == 0:
            return float('nan')
        return f1_score(y_true, y_pred, zero_division=0)

    linear_f1 = safe_f1(all_targets[linear_mask], all_preds[linear_mask])
    nonlinear_f1 = safe_f1(all_targets[nonlinear_mask], all_preds[nonlinear_mask])

    # ======= Results =======
    results = {
        "AgIoU": round(agiou, 3),
        "Precision": round(precision_score(all_targets, all_preds, zero_division=0), 3),
        "Recall": round(recall_score(all_targets, all_preds, zero_division=0), 3),
        "F1": round(f1_score(all_targets, all_preds, zero_division=0), 3),
        "MCC": round(matthews_corrcoef(all_targets, all_preds), 3),
        "Accuracy": round(accuracy_score(all_targets, all_preds), 3),
        "AUC": round(auc, 3),
        "PR-AUC": round(pr_auc, 3),
        "PCC": round(pcc, 3),
        "Brier": round(brier, 3),
        "BCE": round(bce, 3),
        "Linear_F1": round(linear_f1, 3),
        "Nonlinear_F1": round(nonlinear_f1, 3)
    }

    return results, sample_ious

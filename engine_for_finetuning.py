"""
@time: 2021/12/11

@ author: ysx
"""
import math
import sys
from typing import Iterable, Optional
import torch
import utils
import numpy as np
from timm.utils import accuracy, ModelEma
from sklearn.metrics import roc_auc_score


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, beta_scheduler,
                    max_norm: float = 0, model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (batch, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the auc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        inputs1, inputs2 = batch
        inputs1 = inputs1.to(device, non_blocking=True)
        inputs2 = inputs2.to(device, non_blocking=True)

        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs, skl = model(inputs1, inputs2)
            ce_loss = criterion(outputs, targets)

        beta = beta_scheduler(it)
        loss = ce_loss + beta*skl

        ce_loss_value = ce_loss.item()
        skl_value = skl.item()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(data_iter_step + 1) % update_freq == 0)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        if args.data_path != '../data/ZheJiang/':
            outputs = torch.sigmoid(outputs)
        # _, outputs = torch.max(outputs, 1)
        metrics = utils.evaluate_metrics(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy(), threshold=0.5)

        metric_logger.update(ce_loss=ce_loss_value)
        metric_logger.update(skl=skl_value)
        metric_logger.update(beta=beta)

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)

        metric_logger.update(acc=metrics['acc'])
        metric_logger.update(f1=metrics['f1'])
        metric_logger.update(auc=metrics['auc'])

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(ce_loss=ce_loss_value, head="loss")
            log_writer.update(skl=skl_value, head="loss")
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(beta=beta, head="loss")

            log_writer.update(acc=metrics['acc'], head="metrics")
            log_writer.update(f1=metrics['f1'], head="metrics")
            log_writer.update(auc=metrics['auc'], head="metrics")

            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args, data_loader, model, device, header=None):
    if args.data_path != '../data/ZheJiang/':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = header + ':'

    # switch to evaluation mode
    model.eval()

    outputs, targets = [], []
    for batch, target in metric_logger.log_every(data_loader, 10, header):
        inputs1, inputs2 = batch
        inputs1 = inputs1.to(device, non_blocking=True)
        inputs2 = inputs2.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output, _ = model(inputs1, inputs2)
            loss = criterion(output, target)

        if args.data_path != '../data/ZheJiang/':
            output = torch.sigmoid(output)

        # _, output = torch.max(output, 1)
        for i in range(len(output)):
            outputs.append(output[i].cpu().detach().numpy())
            targets.append(target[i].cpu().detach().numpy())

    metrics = utils.evaluate_metrics(np.array(targets), np.array(outputs), threshold=0.5)

    metric_logger.update(loss=loss.item())
    metric_logger.meters['acc'].update(metrics['acc'])
    metric_logger.meters['f1'].update(metrics['f1'])
    metric_logger.meters['auc'].update(metrics['auc'])

    # gather the stats from all processes
    print('{} acc: {acc.global_avg:.4f}, f1: {f1.global_avg:.4f}, auc: {auc.global_avg:.4f}, loss: {losses.global_avg:.4f}'
          .format(header, acc=metric_logger.acc, f1=metric_logger.f1, auc=metric_logger.auc, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

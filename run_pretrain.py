"""
@time: 2021/12/10

@ author: ysx
"""
import torch
import time
import datetime
import os
import json
import utils
from pathlib import Path
from utils import NativeScalerWithGradNormCount as NativeScaler
from timm.models import create_model
from optim_factory import create_optimizer
from config_pretrain import config
from engine_for_pretraining import train_one_epoch
from schedulers import ExponentialScheduler, LinearScheduler, SmoothScheduler
import modeling_pretrain
from datasets import load_datasets_pretrain


def main(args=config):
    utils.setup_seed(args.seed)

    print('torch.cuda.is_available:', torch.cuda.is_available())

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # datasets
    dataset_train = load_datasets_pretrain(datafolder=args.data_path, window_size=args.window_size,
                                           mask_ratio=args.mask_ratio)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    num_training_steps_per_epoch = len(dataset_train) // args.batch_size
    # model
    model = create_model(model_name=args.model_name,
                         pretrained=False,
                         drop_path_rate=args.drop_path,
                         drop_block_rate=None,
                         is_using_low_dim=args.is_using_low_dim,
                         normlize_target=args.normlize_target,
                         is_img_view1=args.is_img_view1,
                         is_img_view2=args.is_img_view2,
                         )

    model.to(utils.device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    args.lr = args.lr * args.batch_size / 256
    print('number of params: {} M'.format(n_parameters / 1e6))
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % args.batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (args.batch_size * num_training_steps_per_epoch))

    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()

    beta_scheduler = ExponentialScheduler(start_value=args.beta_start_value, end_value=args.beta_end_value,
                                          n_iterations=num_training_steps_per_epoch*args.max_epochs,
                                          start_iteration=num_training_steps_per_epoch)

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(args.lr, args.min_lr, args.max_epochs, num_training_steps_per_epoch,
                                                warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps)

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end, args.max_epochs,
                                                num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f, Length = %d" % (max(wd_schedule_values), min(wd_schedule_values),
                                                         len(wd_schedule_values)))

    utils.auto_load_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.start_epoch} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.max_epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model=model, data_loader=data_loader_train,
            optimizer=optimizer, device=utils.device, epoch=epoch,
            loss_scaler=loss_scaler, beta_scheduler=beta_scheduler,
            max_norm=args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values)

        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.max_epochs:
                utils.save_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    main(config)


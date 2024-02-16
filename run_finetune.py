"""
@time: 2021/12/11

@ author: ysx
"""
import os
import time
import datetime
import torch
import utils
import json
from pathlib import Path
from collections import OrderedDict
from config_finetune import config
from timm.models import create_model
from utils import NativeScalerWithGradNormCount as NativeScaler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, LayerDecayValueAssigner
from engine_for_finetuning import train_one_epoch, evaluate
from schedulers import ExponentialScheduler, LinearScheduler, SmoothScheduler
from datasets import load_datasets_finetune
import modeling_finetune


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

    dataset_train, dataset_val, dataset_test, num_classes = load_datasets_finetune(datafolder=args.data_path,
                                                                                   experiment=args.experiment)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    model = create_model(
        model_name=args.model_name,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        is_img_view1=args.is_img_view1,
        is_img_view2=args.is_img_view2,
        is_using_low_dim=args.is_using_low_dim,
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None

        if args.model_key in checkpoint:
            checkpoint_model = checkpoint[args.model_key]
            print("Load state_dict by model_key = %s" % args.model_key)

        if checkpoint_model is None:
            checkpoint_model = checkpoint

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('mencoder.'):
                new_dict[key[9:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict
        # utils.load_pos_embed('view1_encoder.vit.pos_embed', checkpoint_model, model)
        # utils.load_pos_embed('view2_encoder.vit.pos_embed', checkpoint_model, model)
        print(list(model.state_dict().keys()))
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
        # model.load_state_dict(checkpoint_model, strict=False)
    model.to(utils.device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    args.lr = args.lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    # Note: to modify assigner.get_layer_id
    num_layers = model.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    optimizer = create_optimizer(
        args, model, skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = NativeScaler()

    beta_scheduler = ExponentialScheduler(start_value=args.beta_start_value, end_value=args.beta_end_value,
                                          n_iterations=num_training_steps_per_epoch * args.max_epochs,
                                          start_iteration=num_training_steps_per_epoch)

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.max_epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.max_epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        if args.data_path != '../data/ZheJiang/':
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.test:
        test_stats = evaluate(args, data_loader_test, model, utils.device, header='Test')
        print(f"Acc, F1 and Auc of the network on the {len(dataset_test)} test data: {test_stats['acc']:.4f}, {test_stats['f1']:.4f}, {test_stats['auc']:.4f}")
        exit(0)

    print(f"Start training for {args.start_epoch} epochs")
    start_time = time.time()

    best_epoch_auc, test_auc, max_auc = 0, 0., 0.
    best_epoch_acc, test_acc, max_acc = 0, 0., 0.
    best_epoch_f1, test_f1, max_f1 = 0, 0., 0.
    for epoch in range(args.start_epoch, args.max_epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            utils.device, epoch, loss_scaler, beta_scheduler, args.clip_grad, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq, args=args
        )

        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.max_epochs:
                utils.save_model(
                    args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler,
                    epoch=epoch, model_ema=model_ema)

        if data_loader_val is not None:
            # val
            val_stats = evaluate(args, data_loader_val, model, utils.device, header='Val')
            print(f"Acc, F1 and Auc of the network on the {len(dataset_val)} val dataset:: {val_stats['acc']:.4f}, {val_stats['f1']:.4f}, {val_stats['auc']:.4f}")
            # test
            test_stats = evaluate(args, data_loader_test, model, utils.device, header='Test')
            print(f"Acc, F1 and Auc of the network on the {len(dataset_test)} test dataset:: {test_stats['acc']:.4f}, {test_stats['f1']:.4f}, {test_stats['auc']:.4f}")
            if log_writer is not None:

                log_writer.update(val_acc=val_stats['acc'], head="perf", step=epoch)
                log_writer.update(val_f1=val_stats['f1'], head="perf", step=epoch)
                log_writer.update(val_auc=val_stats['auc'], head="perf", step=epoch)
                log_writer.update(val_loss=val_stats['loss'], head="perf", step=epoch)

                log_writer.update(test_acc=test_stats['acc'], head="perf", step=epoch)
                log_writer.update(test_f1=test_stats['f1'], head="perf", step=epoch)
                log_writer.update(test_auc=test_stats['auc'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            if max_acc < val_stats["acc"]:
                max_acc = val_stats["acc"]
                test_acc = test_stats['acc']
                best_epoch_acc = epoch
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler,
                        epoch="best_acc", model_ema=model_ema)

            if max_f1 < val_stats["f1"]:
                max_f1 = val_stats["f1"]
                test_f1 = test_stats['f1']
                best_epoch_f1 = epoch
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler,
                        epoch="best_f1", model_ema=model_ema)

            if max_auc < val_stats["auc"]:
                max_auc = val_stats["auc"]
                test_auc = test_stats['auc']
                best_epoch_auc = epoch
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler,
                        epoch="best_auc", model_ema=model_ema)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         # **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        print(f'*************best epoch Acc:{best_epoch_acc}, Max Val Acc: {max_acc:.4f}, Test Acc: {test_acc:.4f}')
        print(f'*************best epoch Auc:{best_epoch_auc}, Max Val Auc: {max_auc:.4f}, Test Auc: {test_auc:.4f}')
        print(f'*************best epoch F1:{best_epoch_f1}, Max Val F1: {max_f1:.4f}, Test F1: {test_f1:.4f}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time: {total_time_str}')


if __name__ == '__main__':
    main(config)


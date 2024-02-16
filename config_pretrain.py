"""
@time: 2021/12/9

@ author: ysx
"""


class Config:
    '''MIBVIT pre-training script'''
    # 'CPSC', experiment=None
    # 'ptbxl', experiment='exp0','exp1','exp1.1','exp1.1.1','exp2','exp3'
    # 'hf',  experiment=None
    # 'ZheJiang', experiment=None
    data_path = '../data/ALL/'
    #
    experiment = ''

    output_dir = './save_checkpoint/pretrain'

    log_dir = './log/pretrain'
    # random seeds (default: 42)
    seed = 42
    # (default: 1)
    num_workers = 8
    # resume from checkpoint
    resume = ''
    #
    auto_resume = True
    # (default: 0)
    start_epoch = 0
    # (default: 64)
    batch_size = 64
    # (default: 200)
    max_epochs = 200
    # save checkpoint frequency (default: 20)
    save_ckpt_freq = 20
    # model name
    model_name = 'pretrain_mib_vit'
    #
    window_size = (1, 100)
    # ratio of the visual tokens/patches need be masked (default: 0.75)
    mask_ratio = 0.75
    # Drop path rate (default: 0.1)
    drop_path = 0.0

    num_classes = 10
    # Start and end values of the hyperparameter beta
    beta_start_value = 1e-2
    beta_end_value = 1e-4
    # reconstructing the original data using low-dimensional vectors
    is_using_low_dim = False
    # normalized the target patch
    normlize_target = False
    # view 1 is the image?
    is_img_view1 = False
    # view 2 is the image?
    is_img_view2 = True

    # Optimizer parameters
    opt = 'Adamw'
    # Optimizer Epsilon (default: 1e-8)
    opt_eps = 1e-8
    # Optimizer Betas (default: None)
    opt_betas = None
    # Clip gradient norm (default: None)
    clip_grad = None
    # SGD momentum (default: 0.9)
    momentum = 0.9
    # weight decay(default: 0.05)
    weight_decay = 0.05
    # Final value of the weight decay. We use a cosine schedule for WD.
    # (Set the same value with args.weight_decay to keep weight decay no change)
    weight_decay_end = None
    # learning rate (default: 1.5e-4)
    lr = 1.5e-4
    # warmup learning rate (default: 1e-6)
    warmup_lr = 1e-6
    # lower lr bound for cyclic schedulers that hit 0 (1e-5)
    min_lr = 1e-5
    # epochs to warmup LR, if scheduler supports
    warmup_epochs = 40
    # epochs to warmup LR, if scheduler supports
    warmup_steps = -1


config = Config()




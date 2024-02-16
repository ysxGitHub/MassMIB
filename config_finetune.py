"""
@time: 2021/12/10

@ author: ysx
"""

class Config:
    '''MIBVIT fine-tuning and evaluation script for MTS classification'''
    # 'CPSC', experiment=None
    # 'ptbxl', experiment='exp0','exp1','exp1.1','exp1.1.1','exp2','exp3'
    # 'hf',  experiment=None
    # 'ZheJiang', experiment=None
    data_path = '../data/ZheJiang/'
    #
    experiment = 'exp3'
    # Perform test only
    test = False
    # Label smoothing (default: 0.1)
    smoothing = 0.
    #
    output_dir = './save_checkpoint/finetune'
    #
    log_dir = './log/finetune'
    # random seeds (default: 42)
    seed = 9
    # (default: 1)
    num_workers = 8
    # resume from checkpoint
    resume = ''
    #
    auto_resume = False
    # finetune from checkpoint
    # finetune = './save_checkpoint/pretrain/checkpoint-199.pth'
    finetune = None
    #
    model_key = 'model'
    # (default: 0)
    start_epoch = 0
    # (default: 64)
    batch_size = 64
    #
    update_freq = 1
    # (default: 50)
    max_epochs = 30
    # save checkpoint frequency (default: 20)
    save_ckpt_freq = 30
    #
    save_ckpt = True
    # model name
    model_name = 'mib_vit_net'
    # Drop path rate (default: 0.1)
    drop_path = 0.1
    # Dropout rate (default: 0.)
    drop = 0.
    # Attention dropout rate (default: 0.)
    attn_drop_rate = 0.
    #
    use_mean_pooling = True
    #
    model_prefix = ''
    #
    num_classes = 10
    # values of the hyperparameter beta
    beta_value = 0.5
    # Start and end values of the hyperparameter beta
    beta_start_value = 1e-2
    beta_end_value = 1e-4
    # reconstructing the original data using low-dimensional vectors
    is_using_low_dim = False
    # view 1 is the image?
    is_img_view1 = False
    # view 2 is the image?
    is_img_view2 = True
    #
    init_scale = 0.001
    #
    model_ema = True
    #
    model_ema_decay = 0.9999
    #
    model_ema_force_cpu = False

    # Optimizer parameters
    opt = 'adamw'
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
    # learning rate (default: 1e-3)
    lr = 1e-3
    # warmup learning rate (default: 1e-6)
    warmup_lr = 1e-6
    # lower lr bound for cyclic schedulers that hit 0 (1e-5)
    min_lr = 1e-6
    # epochs to warmup LR, if scheduler supports
    warmup_epochs = 5
    # epochs to warmup LR, if scheduler supports
    warmup_steps = -1
    # (default: 0.75)
    layer_decay = 0.75


config = Config()

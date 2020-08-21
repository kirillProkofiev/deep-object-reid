from yacs.config import CfgNode as CN


def get_default_config():
    cfg = CN()

    # model
    cfg.model = CN()
    cfg.model.name = 'resnet50'
    cfg.model.pretrained = False
    cfg.model.download_weights = True
    cfg.model.load_weights = ''  # path to model weights
    cfg.model.resume = ''  # path to checkpoint for resume training
    cfg.model.dropout = CN()
    cfg.model.dropout.p = 0.0
    cfg.model.dropout.mu = 0.1
    cfg.model.dropout.sigma = 0.03
    cfg.model.dropout.dist = 'none'
    cfg.model.feature_dim = 512  # embedding size
    cfg.model.bn_eval = False
    cfg.model.bn_frozen = False
    cfg.model.enable_attentions = False
    cfg.model.pooling_type = 'avg'
    cfg.model.IN_first = False
    cfg.model.extra_blocks = False
    cfg.model.lct_gate = False
    cfg.model.fpn = CN()
    cfg.model.fpn.enable = True
    cfg.model.fpn.dim = 256
    cfg.model.fpn.process = 'concatenation'
    cfg.model.rsc = CN()
    cfg.model.rsc.enable = False
    cfg.model.rsc.drop_percentage = 0.33
    cfg.model.rsc.mode = 'overall'

    # data
    cfg.data = CN()
    cfg.data.type = 'image'
    cfg.data.root = 'reid-data'
    cfg.data.sources = ['market1501']
    cfg.data.targets = ['market1501']
    cfg.data.workers = 4  # number of data loading workers
    cfg.data.split_id = 0  # Split index
    cfg.data.height = 256  # image height
    cfg.data.width = 128  # image width
    cfg.data.enable_masks = False
    cfg.data.combineall = False  # combine train, query and gallery for training
    cfg.data.norm_mean = [0.485, 0.456, 0.406]  # default is imagenet mean
    cfg.data.norm_std = [0.229, 0.224, 0.225]  # default is imagenet std
    cfg.data.save_dir = 'log'  # path to save log
    cfg.data.min_samples_per_id = 1
    # specific datasets
    cfg.market1501 = CN()
    cfg.market1501.use_500k_distractors = False  # add 500k distractors to the gallery set for market1501
    cfg.cuhk03 = CN()
    cfg.cuhk03.labeled_images = False  # use labeled images, if False, use detected images
    cfg.cuhk03.classic_split = False  # use classic split by Li et al. CVPR14
    cfg.cuhk03.use_metric_cuhk03 = False  # use cuhk03's metric for evaluation

    # sampler
    cfg.sampler = CN()
    cfg.sampler.train_sampler = 'RandomSampler'
    cfg.sampler.batch_num_instances = 4  # number of instances per identity for RandomIdentitySampler
    cfg.sampler.epoch_num_instances = -1
    cfg.sampler.fill_instances = False

    # video reid setting
    cfg.video = CN()
    cfg.video.seq_len = 15  # number of images to sample in a tracklet
    cfg.video.sample_method = 'evenly'  # how to sample images from a tracklet
    cfg.video.pooling_method = 'avg'  # how to pool features over a tracklet

    # train
    cfg.train = CN()
    cfg.train.optim = 'adam'
    cfg.train.lr = 0.0003
    cfg.train.weight_decay = 5e-4
    cfg.train.max_epoch = 60
    cfg.train.start_epoch = 0
    cfg.train.batch_size = 32
    cfg.train.fixbase_epoch = 0  # number of epochs to fix base layers
    cfg.train.open_layers = ['classifier']  # layers for training while keeping others frozen
    cfg.train.staged_lr = False  # set different lr to different layers
    cfg.train.new_layers = ['classifier']  # newly added layers with default lr
    cfg.train.base_lr_mult = 0.1  # learning rate multiplier for base layers
    cfg.train.lr_scheduler = 'single_step'
    cfg.train.stepsize = [20]  # stepsize to decay learning rate
    cfg.train.gamma = 0.1  # learning rate decay multiplier
    cfg.train.lr_scales = []
    cfg.train.print_freq = 20  # print frequency
    cfg.train.seed = 1  # random seed
    cfg.train.warmup = 1  # After fixbase_epoch
    cfg.train.warmup_factor_base = 0.1
    cfg.train.frozen_factor_base = 1.0

    # optimizer
    cfg.sgd = CN()
    cfg.sgd.momentum = 0.9  # momentum factor for sgd and rmsprop
    cfg.sgd.dampening = 0.  # dampening for momentum
    cfg.sgd.nesterov = False  # Nesterov momentum
    cfg.rmsprop = CN()
    cfg.rmsprop.alpha = 0.99  # smoothing constant
    cfg.adam = CN()
    cfg.adam.beta1 = 0.9  # exponential decay rate for first moment
    cfg.adam.beta2 = 0.999  # exponential decay rate for second moment

    # loss
    cfg.loss = CN()
    cfg.loss.name = 'softmax'
    cfg.loss.softmax = CN()
    cfg.loss.softmax.label_smooth = False  # use label smoothing regularizer
    cfg.loss.softmax.conf_penalty = 0.0
    cfg.loss.softmax.pr_product = False
    cfg.loss.softmax.m = 0.35
    cfg.loss.softmax.s = 30.0
    cfg.loss.softmax.end_s = -1.0
    cfg.loss.softmax.duration_s = -1
    cfg.loss.softmax.skip_steps_s = -1
    cfg.loss.softmax.adaptive_margins = False
    cfg.loss.softmax.base_num_classes = -1
    cfg.loss.triplet = CN()
    cfg.loss.triplet.margin = 0.3  # distance margin
    cfg.loss.triplet.weight_t = 1.  # weight to balance hard triplet loss
    cfg.loss.triplet.weight_x = 0.  # weight to balance cross entropy loss

    # metric_losses
    cfg.metric_losses = CN()
    cfg.metric_losses.enable = False
    cfg.metric_losses.center_coeff = 0.0
    cfg.metric_losses.triplet_coeff = 0.0
    cfg.metric_losses.local_push_coeff = 1.0

    # attribute loss
    cfg.attr_loss = CN()
    cfg.attr_loss.names = ['color', 'type']
    cfg.attr_loss.num_classes = [-1, -1]
    cfg.attr_loss.label_smooth = False
    cfg.attr_loss.conf_penalty = 0.0
    cfg.attr_loss.pr_product = False
    cfg.attr_loss.m = 0.35
    cfg.attr_loss.s = 30.0
    cfg.attr_loss.end_s = -1.0
    cfg.attr_loss.duration_s = -1
    cfg.attr_loss.skip_steps_s = -1

    # regularizers
    cfg.reg = CN()
    cfg.reg.ow = False
    cfg.reg.ow_beta = 1e-3
    cfg.reg.nw = False
    cfg.reg.nw_max_ratio = 10.0
    cfg.reg.nw_scale = 10.0
    cfg.reg.hd = False
    cfg.reg.hd_max_score = 0.5
    cfg.reg.hd_scale = 10.0
    cfg.reg.of = False
    cfg.reg.of_beta = 1e-6
    cfg.reg.of_start_epoch = 23

    # test
    cfg.test = CN()
    cfg.test.batch_size = 100
    cfg.test.dist_metric = 'euclidean'  # distance metric, ['euclidean', 'cosine']
    cfg.test.normalize_feature = False  # normalize feature vectors before computing distance
    cfg.test.ranks = [1, 5, 10, 20]  # cmc ranks
    cfg.test.evaluate = False  # test only
    cfg.test.eval_freq = -1  # evaluation frequency (-1 means to only test after training)
    cfg.test.start_eval = 0  # start to evaluate after a specific epoch
    cfg.test.rerank = False  # use person re-ranking
    cfg.test.visrank = False  # visualize ranked results (only available when cfg.test.evaluate=True)
    cfg.test.visrank_topk = 10  # top-k ranks to visualize
    cfg.test.visactmap = False  # visualize CNN activation maps
    cfg.test.apply_masks = False

    # Augmentations
    cfg.data.transforms = CN()

    cfg.data.transforms.random_flip = CN()
    cfg.data.transforms.random_flip.enable = True
    cfg.data.transforms.random_flip.p = 0.5

    cfg.data.transforms.random_crop = CN()
    cfg.data.transforms.random_crop.enable = False
    cfg.data.transforms.random_crop.p = 0.5
    cfg.data.transforms.random_crop.scale = 0.9

    cfg.data.transforms.random_gray_scale = CN()
    cfg.data.transforms.random_gray_scale.enable = False
    cfg.data.transforms.random_gray_scale.p = 0.5

    cfg.data.transforms.random_padding = CN()
    cfg.data.transforms.random_padding.enable = False
    cfg.data.transforms.random_padding.p = 0.5
    cfg.data.transforms.random_padding.padding = (0, 10)

    cfg.data.transforms.random_perspective = CN()
    cfg.data.transforms.random_perspective.enable = False
    cfg.data.transforms.random_perspective.p = 0.5
    cfg.data.transforms.random_perspective.distortion_scale = 0.5

    cfg.data.transforms.color_jitter = CN()
    cfg.data.transforms.color_jitter.enable = False
    cfg.data.transforms.color_jitter.p = 0.5
    cfg.data.transforms.color_jitter.brightness = 0.2
    cfg.data.transforms.color_jitter.contrast = 0.15
    cfg.data.transforms.color_jitter.saturation = 0.0
    cfg.data.transforms.color_jitter.hue = 0.0

    cfg.data.transforms.random_erase = CN()
    cfg.data.transforms.random_erase.enable = False
    cfg.data.transforms.random_erase.p = 0.5
    cfg.data.transforms.random_erase.sl = 0.2
    cfg.data.transforms.random_erase.sh = 0.4
    cfg.data.transforms.random_erase.rl = 0.3
    cfg.data.transforms.random_erase.rh = 3.3
    cfg.data.transforms.random_erase.fill_color = (125.307, 122.961, 113.8575)
    cfg.data.transforms.random_erase.norm_image = True

    cfg.data.transforms.random_rotate = CN()
    cfg.data.transforms.random_rotate.enable = False
    cfg.data.transforms.random_rotate.p = 0.5
    cfg.data.transforms.random_rotate.angle = (-5, 5)

    cfg.data.transforms.cut_out_with_prior = CN()
    cfg.data.transforms.cut_out_with_prior.enable = False
    cfg.data.transforms.cut_out_with_prior.p = 0.5
    cfg.data.transforms.cut_out_with_prior.max_area = 0.1

    cfg.data.transforms.random_blur = CN()
    cfg.data.transforms.random_blur.enable = False
    cfg.data.transforms.random_blur.p = 0.5
    cfg.data.transforms.random_blur.k = 5

    cfg.data.transforms.random_figures = CN()
    cfg.data.transforms.random_figures.enable = False
    cfg.data.transforms.random_figures.p = 0.5
    cfg.data.transforms.random_figures.random_color = True
    cfg.data.transforms.random_figures.always_single_figure = False
    cfg.data.transforms.random_figures.thicknesses = (1, 6)
    cfg.data.transforms.random_figures.circle_radiuses = (5, 64)
    cfg.data.transforms.random_figures.figure_prob = 0.5
    cfg.data.transforms.random_figures.before_resize = True

    cfg.data.transforms.random_patch = CN()
    cfg.data.transforms.random_patch.enable = False
    cfg.data.transforms.random_patch.p = 0.5
    cfg.data.transforms.random_patch.pool_capacity = 50000
    cfg.data.transforms.random_patch.min_sample_size = 100
    cfg.data.transforms.random_patch.patch_min_area = 0.01
    cfg.data.transforms.random_patch.patch_max_area = 0.5
    cfg.data.transforms.random_patch.patch_min_ratio = 0.1
    cfg.data.transforms.random_patch.prob_rotate = 0.5
    cfg.data.transforms.random_patch.prob_flip_leftright = 0.5

    cfg.data.transforms.random_grid = CN()
    cfg.data.transforms.random_grid.enable = False
    cfg.data.transforms.random_grid.p = 0.33
    cfg.data.transforms.random_grid.color = (-1, -1, -1)
    cfg.data.transforms.random_grid.grid_size = (24, 64)
    cfg.data.transforms.random_grid.thickness = (1, 1)
    cfg.data.transforms.random_grid.angle = (0, 180)
    cfg.data.transforms.random_grid.before_resize = True

    cfg.data.transforms.random_background_substitution = CN()
    cfg.data.transforms.random_background_substitution.enable = False
    cfg.data.transforms.random_background_substitution.p = 0.2
    cfg.data.transforms.random_background_substitution.images_root_dir = ''
    cfg.data.transforms.random_background_substitution.images_list_file = ''

    cfg.data.transforms.mixup = CN()
    cfg.data.transforms.mixup.enable = False
    cfg.data.transforms.mixup.p = 0.33
    cfg.data.transforms.mixup.alpha = 0.2
    cfg.data.transforms.mixup.images_root_dir = ''
    cfg.data.transforms.mixup.images_list_file = ''

    cfg.data.transforms.batch_transform = CN()
    cfg.data.transforms.batch_transform.enable = False
    cfg.data.transforms.batch_transform.type = 'Pairing'
    cfg.data.transforms.batch_transform.alpha = 1.
    cfg.data.transforms.batch_transform.anchor_bias = 0.8

    return cfg


def imagedata_kwargs(cfg):
    return {
        'root': cfg.data.root,
        'sources': cfg.data.sources,
        'targets': cfg.data.targets,
        'height': cfg.data.height,
        'width': cfg.data.width,
        'transforms': cfg.data.transforms,
        'norm_mean': cfg.data.norm_mean,
        'norm_std': cfg.data.norm_std,
        'use_gpu': cfg.use_gpu,
        'split_id': cfg.data.split_id,
        'combineall': cfg.data.combineall,
        'batch_size_train': cfg.train.batch_size,
        'batch_size_test': cfg.test.batch_size,
        'workers': cfg.data.workers,
        'batch_num_instances': cfg.sampler.batch_num_instances,
        'epoch_num_instances': cfg.sampler.epoch_num_instances,
        'fill_instances': cfg.sampler.fill_instances,
        'train_sampler': cfg.sampler.train_sampler,
        'enable_masks': cfg.data.enable_masks,
        # image
        'cuhk03_labeled': cfg.cuhk03.labeled_images,
        'cuhk03_classic_split': cfg.cuhk03.classic_split,
        'market1501_500k': cfg.market1501.use_500k_distractors,
        'apply_masks_to_test': cfg.test.apply_masks,
        'min_samples_per_id': cfg.data.min_samples_per_id,
    }


def videodata_kwargs(cfg):
    return {
        'root': cfg.data.root,
        'sources': cfg.data.sources,
        'targets': cfg.data.targets,
        'height': cfg.data.height,
        'width': cfg.data.width,
        'transforms': cfg.data.transforms,
        'norm_mean': cfg.data.norm_mean,
        'norm_std': cfg.data.norm_std,
        'use_gpu': cfg.use_gpu,
        'split_id': cfg.data.split_id,
        'combineall': cfg.data.combineall,
        'batch_size_train': cfg.train.batch_size,
        'batch_size_test': cfg.test.batch_size,
        'workers': cfg.data.workers,
        'batch_num_instances': cfg.sampler.batch_num_instances,
        'train_sampler': cfg.sampler.train_sampler,
        # video
        'seq_len': cfg.video.seq_len,
        'sample_method': cfg.video.sample_method
    }


def optimizer_kwargs(cfg):
    return {
        'optim': cfg.train.optim,
        'lr': cfg.train.lr,
        'weight_decay': cfg.train.weight_decay,
        'momentum': cfg.sgd.momentum,
        'sgd_dampening': cfg.sgd.dampening,
        'sgd_nesterov': cfg.sgd.nesterov,
        'rmsprop_alpha': cfg.rmsprop.alpha,
        'adam_beta1': cfg.adam.beta1,
        'adam_beta2': cfg.adam.beta2,
        'staged_lr': cfg.train.staged_lr,
        'new_layers': cfg.train.new_layers,
        'base_lr_mult': cfg.train.base_lr_mult
    }


def lr_scheduler_kwargs(cfg):
    return {
        'lr_scheduler': cfg.train.lr_scheduler,
        'stepsize': cfg.train.stepsize,
        'gamma': cfg.train.gamma,
        'lr_scales': cfg.train.lr_scales,
        'max_epoch': cfg.train.max_epoch,
        'warmup': cfg.train.warmup,
        'frozen': cfg.train.fixbase_epoch,
        'warmup_factor_base': cfg.train.warmup_factor_base,
        'frozen_factor_base': cfg.train.frozen_factor_base
    }


def model_kwargs(cfg, num_classes):
    if isinstance(num_classes, (tuple, list)) and len(num_classes) == 1:
        num_classes = num_classes[0]

    return {
        'name': cfg.model.name,
        'num_classes': num_classes,
        'loss': cfg.loss.name,
        'pretrained': cfg.model.pretrained,
        'download_weights': cfg.model.download_weights,
        'use_gpu': cfg.use_gpu,
        'dropout_cfg': cfg.model.dropout,
        'feature_dim': cfg.model.feature_dim,
        'fpn_cfg': cfg.model.fpn,
        'pooling_type': cfg.model.pooling_type,
        'input_size': (cfg.data.height, cfg.data.width),
        'IN_first': cfg.model.IN_first,
        'extra_blocks': cfg.model.extra_blocks,
        'lct_gate': cfg.model.lct_gate,
        'bn_eval': cfg.model.bn_eval,
        'bn_frozen': cfg.model.bn_frozen,
        'enable_attentions': cfg.model.enable_attentions and cfg.data.enable_masks,
        'attr_names': cfg.attr_loss.names,
        'attr_num_classes': cfg.attr_loss.num_classes,
        'rsc_cfg': cfg.model.rsc,
    }


def engine_run_kwargs(cfg):
    return {
        'save_dir': cfg.data.save_dir,
        'max_epoch': cfg.train.max_epoch,
        'start_epoch': cfg.train.start_epoch,
        'fixbase_epoch': cfg.train.fixbase_epoch,
        'open_layers': cfg.train.open_layers,
        'start_eval': cfg.test.start_eval,
        'eval_freq': cfg.test.eval_freq,
        'test_only': cfg.test.evaluate,
        'print_freq': cfg.train.print_freq,
        'dist_metric': cfg.test.dist_metric,
        'normalize_feature': cfg.test.normalize_feature,
        'visrank': cfg.test.visrank,
        'visrank_topk': cfg.test.visrank_topk,
        'use_metric_cuhk03': cfg.cuhk03.use_metric_cuhk03,
        'ranks': cfg.test.ranks,
        'rerank': cfg.test.rerank
    }


def transforms(cfg):
    return cfg.data.transforms


def augmentation_kwargs(cfg):
    return {
        'random_flip': cfg.data.transforms.random_flip,
        'random_crop': cfg.data.transforms.random_crop,
        'random_gray_scale': cfg.data.transforms.random_gray_scale,
        'random_padding': cfg.data.transforms.random_padding,
        'random_perspective': cfg.data.transforms.random_perspective,
        'color_jitter': cfg.data.transforms.color_jitter,
        'random_erase': cfg.data.transforms.random_erase,
        'random_rotate': cfg.data.transforms.random_rotate,
        'random_figures': cfg.data.transforms.random_figures,
        'random_grid': cfg.data.transforms.random_grid,
        'cut_out_with_prior': cfg.data.transforms.cut_out_with_prior
    }

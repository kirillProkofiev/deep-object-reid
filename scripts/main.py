import sys
import time
import os.path as osp
import argparse
import torch

import torchreid
from torchreid.engine import build_engine
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)
from torchreid.ops import DataParallel

from scripts.default_config import (
    imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs, model_kwargs
)


def build_datamanager(cfg):
    if cfg.data.type == 'image':
        return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets


def build_auxiliary_model(config_file, num_classes, device_ids=None, weights=None):
    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    cfg.merge_from_file(config_file)

    model = torchreid.models.build_model(**model_kwargs(cfg, num_classes))

    if check_isfile(weights):
        load_pretrained_weights(model, weights)

    if cfg.use_gpu:
        assert device_ids is not None

        model = DataParallel(model, device_ids=device_ids, output_device=0).cuda(device_ids[0])

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(cfg))

    return model, optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str, default='',
                        help='path to config file')
    parser.add_argument('-e', '--extra-config-files', type=str, nargs='*', default='',
                        help='path to extra config files')
    parser.add_argument('-w', '--extra-weights', type=str, nargs='*', default='',
                        help='path to extra model weights')
    parser.add_argument('--split-models', action='store_true',
                        help='whether to split models on own gpu')
    parser.add_argument('-s', '--sources', type=str, nargs='+',
                        help='source datasets (delimited by space)')
    parser.add_argument('-t', '--targets', type=str, nargs='+',
                        help='target datasets (delimited by space)')
    parser.add_argument('--root', type=str, default='',
                        help='path to data root')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Modify config options using the command-line')
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)

    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    enable_mutual_learning = len(args.extra_config_files) > 0

    datamanager = build_datamanager(cfg)
    num_train_classes = datamanager.num_train_pids

    print('Building main model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(**model_kwargs(cfg, num_train_classes))
    num_params, flops = compute_model_complexity(model, (1, 3, cfg.data.height, cfg.data.width))
    print('Main model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        if cfg.model.pretrained and not cfg.test.evaluate:
            state_dict = torch.load(cfg.model.load_weights)
            model.load_pretrained_weights(state_dict)
        else:
            load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        num_devices = 1
        if enable_mutual_learning and args.split_models:
            num_models = len(args.extra_config_files) + 1
            assert num_devices >= num_models
            assert num_devices % num_models == 0

            num_devices_per_model = num_devices // num_models
            device_splits = []
            for model_id in range(num_models):
                device_splits.append([
                    model_id * num_devices_per_model + i
                    for i in range(num_devices_per_model)
                ])

            main_device_ids = device_splits[0]
            extra_device_ids = device_splits[1:]
        else:
            main_device_ids = list(range(num_devices))
            extra_device_ids = [main_device_ids for _ in range(len(args.extra_config_files))]

        model = DataParallel(model, device_ids=main_device_ids, output_device=0).cuda(main_device_ids[0])
    else:
        extra_device_ids = [None for _ in range(len(args.extra_config_files))]

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(cfg))

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )

    if enable_mutual_learning:
        print('Enabled mutual learning between {} models.'.format(len(args.extra_config_files) + 1))

        if len(args.extra_weights) > 0:
            assert len(args.extra_weights) == len(args.extra_config_files)
            weights = args.extra_weights
        else:
            weights = [None] * len(args.extra_config_files)

        models, optimizers, schedulers = [model], [optimizer], [scheduler]
        for config_file, model_weights, device_ids in zip(args.extra_config_files, weights, extra_device_ids):
            aux_model, aux_optimizer, aux_scheduler = build_auxiliary_model(
                config_file, num_train_classes, device_ids, model_weights
            )

            models.append(aux_model)
            optimizers.append(aux_optimizer)
            schedulers.append(aux_scheduler)
    else:
        models, optimizers, schedulers = model, optimizer, scheduler

    print('Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type))
    engine = build_engine(cfg, datamanager, models, optimizers, schedulers)
    engine.run(**engine_run_kwargs(cfg))


if __name__ == '__main__':
    main()

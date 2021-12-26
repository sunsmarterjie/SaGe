import random
import re
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, Runner, obj_from_dict
from .gssl_runner import GSSL_Runner

from openselfsup.datasets import build_dataloader
from openselfsup.hooks import build_hook, DistOptimizerHook, DistOptimizerDecoderHook, TextLoggerHook, CheckpointHook, DecoderLrUpdaterHook, CosineAnnealingLrUpdaterHook
from openselfsup.utils import get_root_logger, optimizers, print_log, vis_match

try:
    import apex
except:
    pass
    # print('apex is not installed')


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if 'loss' not in loss_name:
            continue
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
    #loss = losses['loss']
    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars


# i = 0
def batch_processor(model, data, train_mode):
    """Process a data batch.

    This method is required as an argument of Runner, which defines how to
    process a data batch and obtain proper outputs. The first 3 arguments of
    batch_processor are fixed.

    Args:
        model (nn.Module): A PyTorch model.
        data (dict): The data batch in a dict.
        train_mode (bool): Training mode or not. It may be useless for some
            models.

    Returns:
        dict: A dict containing losses and log vars.
    """
    global i
    #     i += 1
    # import pdb;pdb.set_trace()
    #     if i % 50 == 0:
    #         import pdb;pdb.set_trace()
    losses = model(**data)
    if hasattr(model.module, 'vis_match') and model.module.vis_match:
        img_v1 = data['img'][:, 0, ...]
        img_v2 = data['img'][:, 1, ...]
        vis_match(img_v1, img_v2, losses['cos_sim_o2t'], losses['cos_sim_t2o'])

    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                timestamp=None,
                meta=None):
    logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(
            model, dataset, cfg, logger=logger, timestamp=timestamp, meta=meta)
    else:
        _non_dist_train(
            model, dataset, cfg, logger=logger, timestamp=timestamp, meta=meta)


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with regular expression as keys
                  to match parameter names and a dict containing options as
                  values. Options include 6 fields: lr, lr_mult, momentum,
                  momentum_mult, weight_decay, weight_decay_mult.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Example:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> paramwise_options = {
        >>>     '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay_mult=0.1),
        >>>     '\Ahead.': dict(lr_mult=10, momentum=0)}
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001,
        >>>                      paramwise_options=paramwise_options)
        >>> optimizer = build_optimizer(model, optimizer_cfg)
    """
    
    if optimizer_cfg['type'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_cfg.pop('lr'))
        return optimizer
    
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        return obj_from_dict(optimizer_cfg, optimizers,
                             dict(params=model.parameters()))
    else:
        assert isinstance(paramwise_options, dict)
        params = []
        for name, param in model.named_parameters():
            param_group = {'params': [param]}
            if name.startswith('decoder'):
                continue
            if not param.requires_grad:
                params.append(param_group)
                continue

            for regexp, options in paramwise_options.items():
                if re.search(regexp, name):
                    for key, value in options.items():
                        if key.endswith('_mult'):  # is a multiplier
                            key = key[:-5]
                            assert key in optimizer_cfg, \
                                "{} not in optimizer_cfg".format(key)
                            value = optimizer_cfg[key] * value
                        param_group[key] = value
                        if not dist.is_initialized() or dist.get_rank() == 0:
                            print_log('paramwise_options -- {}: {}={}'.format(
                                name, key, value))

            # otherwise use the global settings
            params.append(param_group)

        optimizer_cls = getattr(optimizers, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)


def build_optimizer_decoder(model, optimizer_decoder_cfg):
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with regular expression as keys
                  to match parameter names and a dict containing options as
                  values. Options include 6 fields: lr, lr_mult, momentum,
                  momentum_mult, weight_decay, weight_decay_mult.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """

    print('optimizer decoder cfg:', optimizer_decoder_cfg)
    optimizer_decoder_cfg = optimizer_decoder_cfg.copy()
    paramwise_options = optimizer_decoder_cfg.pop('paramwise_options', None)
    if optimizer_decoder_cfg['type'].lower() == 'adam':
        params = []
        for name, param in model.named_parameters():
            if name.startswith('decoder'):
                print('decoder optimizer params name:', name)
                param_group = {'params': [param]}
                
                for regexp, options in paramwise_options.items():
                    if re.search(regexp, name):
                        for key, value in options.items():
                            if key.endswith('_mult'):  # is a multiplier
                                key = key[:-5]
                                assert key in optimizer_decoder_cfg, \
                                    "{} not in optimizer_cfg".format(key)
                                value = optimizer_decoder_cfg[key] * value
                            param_group[key] = value
                            if not dist.is_initialized() or dist.get_rank() == 0:
                                print_log('paramwise_options -- {}: {}={}'.format(
                                    name, key, value))
                
                params.append(param_group)
        optimizer = torch.optim.Adam(params, lr=optimizer_decoder_cfg.pop('lr'), betas=(0.9, 0.999))
        print('Optimizer Decoder Adam !!!')
        return optimizer

    if hasattr(model, 'module'):
        model = model.module

    optimizer_decoder_cfg = optimizer_decoder_cfg.copy()
    paramwise_options = optimizer_decoder_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        return obj_from_dict(optimizer_decoder_cfg, optimizers,
                             dict(params=model.parameters()))
    else:
        assert isinstance(paramwise_options, dict)
        params = []
        for name, param in model.named_parameters():
            if name.startswith('decoder'):
                param_group = {'params': [param]}

                for regexp, options in paramwise_options.items():
                    if re.search(regexp, name):
                        for key, value in options.items():
                            if key.endswith('_mult'):  # is a multiplier
                                key = key[:-5]
                                assert key in optimizer_decoder_cfg, \
                                    "{} not in optimizer_cfg".format(key)
                                value = optimizer_decoder_cfg[key] * value
                            param_group[key] = value
                            if not dist.is_initialized() or dist.get_rank() == 0:
                                print_log('paramwise_options -- {}: {}={}'.format(
                                    name, key, value))

                # otherwise use the global settings
                params.append(param_group)

        optimizer_cls = getattr(optimizers, optimizer_decoder_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_decoder_cfg)


def _dist_train(model, dataset, cfg, logger=None, timestamp=None, meta=None):
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True,
            shuffle=True,
            replace=getattr(cfg.data, 'sampling_replace', False),
            seed=cfg.seed,
            drop_last=getattr(cfg.data, 'drop_last', False),
            prefetch=cfg.prefetch,
            img_norm_cfg=cfg.img_norm_cfg) for ds in dataset
    ]
    optimizer = build_optimizer(model, cfg.optimizer)
    optimizer_decoder = build_optimizer_decoder(model, cfg.optimizer_decoder)
    if 'use_fp16' in cfg and cfg.use_fp16:
        model, optimizer = apex.amp.initialize(model.cuda(), optimizer, opt_level="O1")
        model, optimizer_decoder = apex.amp.initialize(model.cuda(), optimizer_decoder, opt_level="O1")
        print_log('**** Initializing mixed precision done. ****')

    # put model on gpus
    model = MMDistributedDataParallel(
        model if next(model.parameters()).is_cuda else model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False, find_unused_parameters=True)
    
    #print('cfg.work_dir:', cfg.work_dir)
    # build runner
    runner = GSSL_Runner(
        model,
        batch_processor,
        optimizer,
        optimizer_decoder,
        cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    optimizer_config = DistOptimizerHook(**cfg.optimizer_config)  # already a hook but config
    optimizer_decoder_config = DistOptimizerDecoderHook(**cfg.optimizer_decoder_config)
    log_config = TextLoggerHook()
    checkpoint_config = CheckpointHook(**cfg.checkpoint_config)

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config, optimizer_decoder_config,
                                   checkpoint_config, log_config)
    runner.register_hook(DistSamplerSeedHook())
    print(runner.hooks)
    if optimizer_decoder_config is None:
        runner.hooks[2].args['create_symlink'] = False
    else:
        runner.hooks[3].args['create_symlink'] = False

    # register custom hooks
    for hook in cfg.get('custom_hooks', ()):
        if hook.type == 'DeepClusterHook':
            common_params = dict(dist_mode=True, data_loaders=data_loaders)
        else:
            common_params = dict(dist_mode=True)
        runner.register_hook(build_hook(hook, common_params))
    
    if optimizer_decoder_config is not None:
        #runner.register_hook(DecoderLrUpdaterHook(**cfg.decoder_lr_updater_hooks))
        runner.register_hook(CosineAnnealingLrUpdaterHook(**cfg.decoder_lr_updater_hooks))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model,
                    dataset,
                    cfg,
                    validate=False,
                    logger=None,
                    timestamp=None,
                    meta=None):
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False,
            shuffle=True,
            replace=getattr(cfg.data, 'sampling_replace', False),
            seed=cfg.seed,
            drop_last=getattr(cfg.data, 'drop_last', False),
            prefetch=cfg.prefetch,
            img_norm_cfg=cfg.img_norm_cfg) for ds in dataset
    ]

    if 'use_fp16' in cfg and cfg.use_fp16 == True:
        raise NotImplementedError('apex do not support non_dist_train!')
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(
        model,
        batch_processor,
        optimizer,
        cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp
    optimizer_config = cfg.optimizer_config
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    # register custom hooks
    for hook in cfg.get('custom_hooks', ()):
        if hook.type == 'DeepClusterHook':
            common_params = dict(dist_mode=False, data_loaders=data_loaders)
        else:
            common_params = dict(dist_mode=False)
        runner.register_hook(build_hook(hook, common_params))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)

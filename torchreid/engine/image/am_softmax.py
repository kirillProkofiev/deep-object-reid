"""
 MIT License

 Copyright (c) 2018 Kaiyang Zhou

 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchreid.engine import Engine
from torchreid.losses import get_regularizer, AMSoftmaxLoss, CrossEntropyLoss, MetricLosses
from torchreid import metrics

class ImageAMSoftmaxEngine(Engine):
    r"""AM-Softmax-loss engine for image-reid.
    """

    def __init__(self, datamanager, model, optimizer, reg_cfg, metric_cfg, batch_transform_cfg,
                 scheduler=None, use_gpu=False, softmax_type='stock', label_smooth=False,
                 conf_penalty=False, pr_product=False, m=0.35, s=10, end_s=None,
                 duration_s=None, skip_steps_s=None, enable_masks=False,
                 adaptive_margins=False, attr_cfg=None, base_num_classes=-1, rsc_cfg=None):
        super(ImageAMSoftmaxEngine, self).__init__(datamanager, use_gpu)

        self.rsc_cfg = rsc_cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        assert softmax_type in ['stock', 'am']
        assert s > 0.0
        if softmax_type == 'am':
            assert m >= 0.0

        self.regularizer = get_regularizer(reg_cfg)
        self.enable_metric_losses = metric_cfg.enable
        self.enable_masks = enable_masks

        num_batches = len(self.train_loader)
        num_classes = self.datamanager.num_train_pids
        if not isinstance(num_classes, (list, tuple)):
            num_classes = [num_classes]
        self.num_classes = num_classes
        self.num_targets = len(self.num_classes)

        self.main_losses = nn.ModuleList()
        self.ml_losses = list()
        for trg_id, trg_num_classes in enumerate(self.num_classes):
            scale_base_size = trg_num_classes if base_num_classes <= 0 else base_num_classes
            scale_factor = np.log(trg_num_classes - 1) / np.log(scale_base_size - 1)

            if softmax_type == 'stock':
                self.main_losses.append(CrossEntropyLoss(
                    use_gpu=self.use_gpu,
                    label_smooth=label_smooth,
                    conf_penalty=conf_penalty,
                    scale=scale_factor * s
                ))
            elif softmax_type == 'am':
                if adaptive_margins:
                    trg_class_counts = datamanager.data_counts[trg_id]
                    assert len(trg_class_counts) == trg_num_classes
                else:
                    trg_class_counts = None

                self.main_losses.append(AMSoftmaxLoss(
                    use_gpu=self.use_gpu,
                    label_smooth=label_smooth,
                    conf_penalty=conf_penalty,
                    m=m,
                    s=scale_factor * s,
                    end_s=scale_factor * end_s if self._valid(end_s) else None,
                    duration_s=duration_s * num_batches if self._valid(duration_s) else None,
                    skip_steps_s=skip_steps_s * num_batches if self._valid(skip_steps_s) else None,
                    pr_product=pr_product,
                    class_counts=trg_class_counts
                ))

            if self.enable_metric_losses:
                feature_dim = self.model.module.feature_dim
                if hasattr(self.model.module, 'out_feature_dims'):
                    feature_dim = self.model.module.out_feature_dims[trg_id]

                self.ml_losses.append(MetricLosses(
                    trg_num_classes,
                    feature_dim,
                    metric_cfg.center_coeff,
                    metric_cfg.triplet_coeff,
                    metric_cfg.local_push_coeff,
                    name='ml_{}'.format(trg_id)
                ))

        self.enable_attr = attr_cfg is not None
        self.attr_losses = {}
        if self.enable_attr:
            self.attr_losses = nn.ModuleDict()
            for attr_name, attr_size in zip(attr_cfg.names, attr_cfg.num_classes):
                if attr_size is None or attr_size <= 0:
                    continue

                self.attr_losses[attr_name] = AMSoftmaxLoss(
                    use_gpu=self.use_gpu,
                    label_smooth=attr_cfg.label_smooth,
                    conf_penalty=attr_cfg.conf_penalty,
                    m=attr_cfg.m,
                    s=attr_cfg.s,
                    end_s=attr_cfg.end_s if self._valid(attr_cfg.end_s) else None,
                    duration_s=attr_cfg.duration_s * num_batches if self._valid(attr_cfg.duration_s) else None,
                    skip_steps_s=attr_cfg.skip_steps_s * num_batches if self._valid(attr_cfg.skip_steps_s) else None,
                    pr_product=attr_cfg.pr_product
                )

            if len(self.attr_losses) == 0:
                self.enable_attr = False
            else:
                self.attr_name_map = {attr_name: attr_id for attr_id, attr_name in enumerate(attr_cfg.names)}

        self.batch_transform_cfg = batch_transform_cfg
        self.lambd_distr = torch.distributions.beta.Beta(self.batch_transform_cfg.alpha,
                                                         self.batch_transform_cfg.alpha)

    @staticmethod
    def _valid(value):
        return value is not None and value > 0

    def forward_backward(self, data):
        n_iter = self.epoch * self.num_batches + self.batch_idx
        self.optimizer.zero_grad()

        train_records = self.parse_data_for_train(data, True, self.enable_masks, self.use_gpu)
        imgs = train_records['img']
        obj_ids = train_records['obj_id']
        imgs, obj_ids = self._apply_batch_transform(imgs, obj_ids)

        run_kwargs = self._prepare_run_kwargs(obj_ids)
        model_output = self.model(imgs, **run_kwargs)
        all_logits, all_embeddings, extra_data = self._parse_model_output(model_output)

        total_loss = torch.zeros([], dtype=imgs.dtype, device=imgs.device)
        loss_summary = dict()

        num_trg_losses = 0
        avg_acc = 0
        for trg_id in range(self.num_targets):
            trg_mask = train_records['dataset_id'] == trg_id

            trg_obj_ids = obj_ids[trg_mask]
            trg_num_samples = trg_obj_ids.numel()
            if trg_num_samples == 0:
                continue

            trg_logits = all_logits[trg_id][trg_mask]
            main_loss = self.main_losses[trg_id](trg_logits, trg_obj_ids, iteration=n_iter)
            avg_acc += metrics.accuracy(trg_logits, trg_obj_ids)[0].item()
            loss_summary['main_{}'.format(trg_id)] = main_loss.item()

            trg_loss = main_loss
            if self.enable_metric_losses:
                ml_loss_module = self.ml_losses[trg_id]
                embd = all_embeddings[trg_id][trg_mask]

                ml_loss_module.init_iteration()
                ml_loss, ml_loss_summary = ml_loss_module(embd, trg_logits, trg_obj_ids, n_iter)
                ml_loss_module.end_iteration()

                loss_summary['ml_{}'.format(trg_id)] = ml_loss.item()
                loss_summary.update(ml_loss_summary)
                trg_loss += ml_loss

            total_loss += trg_loss
            num_trg_losses += 1
        total_loss /= float(num_trg_losses)
        avg_acc /= float(num_trg_losses)

        if self.enable_attr and train_records['attr'] is not None:
            attributes = train_records['attr']
            all_attr_logits = extra_data['attr_logits']

            num_attr_losses = 0
            total_attr_loss = 0
            for attr_name, attr_loss_module in self.attr_losses.items():
                attr_labels = attributes[self.attr_name_map[attr_name]]
                valid_attr_mask = attr_labels >= 0

                attr_labels = attr_labels[valid_attr_mask]
                if attr_labels.numel() == 0:
                    continue

                attr_logits = all_attr_logits[attr_name][valid_attr_mask]

                attr_loss = attr_loss_module(attr_logits, attr_labels, iteration=n_iter)
                loss_summary[attr_name] = attr_loss.item()

                total_attr_loss += attr_loss
                num_attr_losses += 1

            total_loss += total_attr_loss / float(max(1, num_attr_losses))

        if self.enable_masks and train_records['mask'] is not None:
            att_loss_val = 0.0
            for att_map in extra_data['att_maps']:
                if att_map is not None:
                    with torch.no_grad():
                        att_map_size = att_map.size()[2:]
                        pos_float_mask = F.interpolate(train_records['mask'], size=att_map_size, mode='nearest')
                        pos_mask = pos_float_mask > 0.0
                        neg_mask = ~pos_mask

                        trg_mask_values = torch.where(pos_mask,
                                                      torch.ones_like(pos_float_mask),
                                                      torch.zeros_like(pos_float_mask))
                        num_positives = trg_mask_values.sum(dim=(1, 2, 3), keepdim=True)
                        num_negatives = float(att_map_size[0] * att_map_size[1]) - num_positives

                        batch_factor = 1.0 / float(att_map.size(0))
                        pos_weights = batch_factor / num_positives.clamp_min(1.0)
                        neg_weights = batch_factor / num_negatives.clamp_min(1.0)

                    att_errors = torch.abs(att_map - trg_mask_values)
                    att_pos_errors = (pos_weights * att_errors)[pos_mask].sum()
                    att_neg_errors = (neg_weights * att_errors)[neg_mask].sum()

                    att_loss_val += 0.5 * (att_pos_errors + att_neg_errors)

            if att_loss_val > 0.0:
                loss_summary['att'] = att_loss_val.item()
                total_loss += att_loss_val

        if self.regularizer is not None and (self.epoch + 1) > self.fixbase_epoch:
            reg_loss = self.regularizer(self.model)

            loss_summary['reg'] = reg_loss.item()
            total_loss += reg_loss

        loss_summary['loss'] = total_loss.item()

        total_loss.backward()
        self.optimizer.step()

        return loss_summary, avg_acc

    def _prepare_run_kwargs(self, trgt_ids=None):
        run_kwargs = dict()
        if self.enable_metric_losses:
            run_kwargs['get_embeddings'] = True
        if self.enable_attr or self.enable_masks:
            run_kwargs['get_extra_data'] = True
        if trgt_ids is not None and self.rsc_cfg:
            run_kwargs['trg_labels'] = trgt_ids

        return run_kwargs

    def _parse_model_output(self, model_output):
        if self.enable_metric_losses:
            all_logits, all_embeddings = model_output[:2]
            all_embeddings = all_embeddings if isinstance(all_embeddings, (tuple, list)) else [all_embeddings]
        else:
            all_logits = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
            all_embeddings = None

        all_logits = all_logits if isinstance(all_logits, (tuple, list)) else [all_logits]

        if self.enable_attr or self.enable_masks:
            extra_data = model_output[-1]
        else:
            extra_data = None

        return all_logits, all_embeddings, extra_data

    def _apply_batch_transform(self, imgs, obj_ids):
        if self.batch_transform_cfg.enable:
            lambd = self.batch_transform_cfg.anchor_bias \
                    + (1 - self.batch_transform_cfg.anchor_bias) \
                    * self.lambd_distr.sample((imgs.shape[0],))
            lambd = lambd.view(-1, 1, 1, 1)

            permuted_idx = torch.randperm(imgs.shape[0])
            imgs = lambd * imgs + (1 - lambd) * imgs[permuted_idx]

        return imgs, obj_ids

import math
import torch.optim as optim
import torch.nn as nn
import torch
from torch.nn.utils import clip_grad_norm_
import models.modules

import logging

logger = logging.getLogger(__name__)

class Optim(object):
    def set_parameters(self, params, model_para):
        self.params = list(model_para)  # careful: params may be a generator
        if self.method == 'sgd':
            self.optimizer = optim.SGD(params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(params, lr=self.lr)  # 修改打印lr的策略，查看para——groups
            logger.info("len(self.optimizer.param_groups) {}".format(len(self.optimizer.param_groups)))
            logger.info([p['lr'] for p in self.optimizer.param_groups])
            # self.optimizer = optim.Adam(self.params, betas=(0.9, 0.98), eps=1e-09)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)
        # lr_scheduler
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=64, eta_min=4e-08,
        #                                                             last_epoch=-1)

    def __init__(self, method, lr, max_grad_norm, min_lr, max_weight_value=None, lr_decay=1, start_decay_at=None,
                 decay_bad_count=6):
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.max_weight_value = max_weight_value
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self.decay_bad_count = decay_bad_count
        self.best_metric = [0,0]  # rouge-1, rouge-2
        self.bad_count = 0
        self.decay_indicator= 2
        self.min_lr = min_lr
        self.start_decay = False
        self.last_score = None


    def step(self):
        # Compute gradients norm.
        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()
        if self.max_weight_value:
            for p in self.params:
                p.data.clamp_(0 - self.max_weight_value, self.max_weight_value)
        # self.scheduler.step()

    def updateLearningRate_epoch_decay(self, score, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)

        self.last_score = score
        for p in self.optimizer.param_groups:
            p['lr'] = self.lr

    def updateLearningRate(self, score, epoch):
        if score[self.decay_indicator-1] > self.best_metric[self.decay_indicator-1]:
            self.best_metric = score
            self.bad_count = 0
        else:
            self.bad_count += 1

        # if self.bad_count >= self.decay_bad_count and self.lr * self.lr_decay >= self.min_lr:
        if self.bad_count >= self.decay_bad_count and self.lr >= 1e-6:
            self.lr = self.lr * self.lr_decay
            for p in self.optimizer.param_groups:
                if p['lr'] >= 1e-6:
                    p['lr'] *= self.lr_decay
                logger.info("Decaying learning rate to %g" % p['lr'])
            self.bad_count = 0
        logger.info('Bad_count: {0}\tCurrent lr: {1} {2}'.format(self.bad_count,
                                                                 self.optimizer.param_groups[0]['lr'],
                                                                 self.optimizer.param_groups[1]['lr']))
        logger.info('Best metric: rouge-1 {0}, rouge-2 {1}'.format(self.best_metric[0], self.best_metric[1]))

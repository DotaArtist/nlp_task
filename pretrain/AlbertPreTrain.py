#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""albert 预训练"""

__author__ = 'yp'

# TODO

import os
import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from apex import amp
from torch.optim import Optimizer
from transformers.modeling_albert import AlbertModel, AlbertPreTrainedModel
from transformers.configuration_albert import AlbertConfig
from torch.nn import CrossEntropyLoss
from transformers.modeling_bert import ACT2FN
from tfrecord.torch.dataset import TFRecordDataset
from matplotlib import pyplot as plot


class AlbertSequenceOrderHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 2)
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        prediction_scores = hidden_states + self.bias

        return prediction_scores


class AlbertForPretrain(AlbertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.albert = AlbertModel(config)

        # For Masked LM
        # The original huggingface implementation, created new output weights via dense layer
        # However the original Albert
        self.predictions_dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.predictions_activation = ACT2FN[config.hidden_act]
        self.predictions_LayerNorm = nn.LayerNorm(config.embedding_size)
        self.predictions_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.predictions_decoder = nn.Linear(config.embedding_size, config.vocab_size)

        self.predictions_decoder.weight = self.albert.embeddings.word_embeddings.weight

        # For sequence order prediction
        self.seq_relationship = AlbertSequenceOrderHead(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            seq_relationship_labels=None,
    ):

        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        loss_fct = CrossEntropyLoss()

        sequence_output = outputs[0]

        sequence_output = self.predictions_dense(sequence_output)
        sequence_output = self.predictions_activation(sequence_output)
        sequence_output = self.predictions_LayerNorm(sequence_output)
        prediction_scores = self.predictions_decoder(sequence_output)

        if masked_lm_labels is not None:
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size)
                                      , masked_lm_labels.view(-1))

        pooled_output = outputs[1]
        seq_relationship_scores = self.seq_relationship(pooled_output)
        if seq_relationship_labels is not None:
            seq_relationship_loss = loss_fct(seq_relationship_scores.view(-1, 2), seq_relationship_labels.view(-1))

        loss = masked_lm_loss + seq_relationship_loss

        return loss


class Lamb(Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.
    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0, adam=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        self.adam = adam
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group['lr']  # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(group['weight_decay'], p.data)

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(-step_size * trust_ratio, adam_step)

        return loss


MAX_LENGTH = 512
LEARNING_RATE = 0.001
EPOCH_DELT = 40
BATCH_SIZE = 48
MAX_GRAD_NORM = 1.0

print(torch.cuda.is_available())
print(f"--- Resume/Start training ---")
feat_map = {"input_ids": "int",
            "input_mask": "int",
            "segment_ids": "int",
            "next_sentence_labels": "int",
            "masked_lm_positions": "int",
            "masked_lm_ids": "int"}

pretrain_file = './medical_data_tfrecord_00000'

# Create albert pretrain model
config = AlbertConfig.from_json_file("e:/model_file/my_albert/config.json")
albert_pretrain = AlbertForPretrain(config)

if torch.cuda.is_available():
    albert_pretrain.cuda()
    print(albert_pretrain.device)

# Create optimizer
optimizer = Lamb([{"params": [p for n, p in list(albert_pretrain.named_parameters())]}], lr=LEARNING_RATE)

# FP16
albert_pretrain, optimizer = amp.initialize(albert_pretrain, optimizer, opt_level="O2")

albert_pretrain.train()
dataset = TFRecordDataset(pretrain_file, index_path=None, description=feat_map)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True)

tmp_loss = 0
start_time = time.time()

if os.path.isfile('pretrain_checkpoint'):
    print(f"--- Load from checkpoint ---")
    checkpoint = torch.load("pretrain_checkpoint")
    albert_pretrain.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    losses = checkpoint['losses']

else:
    epoch = -1
    losses = []

EPOCH = EPOCH_DELT + epoch + 1

for e in range(epoch + 1, EPOCH):
    for batch in tqdm(loader):
        b_input_ids = batch['input_ids'].long()
        b_token_type_ids = batch['segment_ids'].long()
        b_seq_relationship_labels = batch['next_sentence_labels'].long()

        # Convert the dataformat from loaded decoded format into format
        # loaded format is created by google's Albert create_pretrain.py script
        # required by huggingfaces pytorch implementation of albert
        mask_rows = np.nonzero(batch['masked_lm_positions'].numpy())[0]
        mask_cols = batch['masked_lm_positions'].numpy()[batch['masked_lm_positions'].numpy() != 0]
        b_attention_mask = np.zeros((BATCH_SIZE, MAX_LENGTH), dtype=np.int64)
        b_attention_mask[mask_rows, mask_cols] = 1
        b_masked_lm_labels = np.zeros((BATCH_SIZE, MAX_LENGTH), dtype=np.int64) - 100
        b_masked_lm_labels[mask_rows, mask_cols] = batch['masked_lm_ids'].numpy()[batch['masked_lm_positions'].numpy() != 0]
        b_attention_mask = torch.tensor(b_attention_mask).long()
        b_masked_lm_labels = torch.tensor(b_masked_lm_labels).long()

        loss = albert_pretrain(input_ids=b_input_ids.cuda()
                               , attention_mask=b_attention_mask.cuda()
                               , token_type_ids=b_token_type_ids.cuda()
                               , masked_lm_labels=b_masked_lm_labels.cuda()
                               , seq_relationship_labels=b_seq_relationship_labels.cuda())

        # clears old gradients
        optimizer.zero_grad()
        # backward pass
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=albert_pretrain.parameters(), max_norm=MAX_GRAD_NORM)
        # update parameters
        optimizer.step()

        tmp_loss += loss.detach().item()

    # print metrics and save to checkpoint every epoch
    print(f"Epoch: {e}")
    print(f"Train loss: {(tmp_loss / 20)}")
    print(f"Train Time: {(time.time() - start_time) / 60} mins")
    losses.append(tmp_loss / 20)

    tmp_loss = 0
    start_time = time.time()

    torch.save({'model_state_dict': albert_pretrain.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'epoch': e, 'loss': loss, 'losses': losses}
               , 'pretrain_checkpoint')

plot.plot(losses)

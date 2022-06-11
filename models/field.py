# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from scipy.optimize import linear_sum_assignment

import torch
from torch import nn
import torch.nn.functional as F

from queue import Queue
from collections import OrderedDict
from itertools import zip_longest
import numpy as np
import math, os

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from matcher import build_matcher

class SMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SMHA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.in_proj_bias   = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        self.out_proj       = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)
        self._reset_parameters()
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)
    def in_projection(
        self, 
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        w: torch.Tensor,
        b = None,
    ):
        E = query.size(-1)
        if key is value:
            if query is key:
                # self-attention
                return torch._C._nn.linear(query, w, b).chunk(3, dim=-1)
            else:
                # encoder-decoder attention
                w_q, w_kv = w.split([E, E * 2])
                if b is None:
                    b_q = b_kv = None
                else:
                    b_q, b_kv = b.split([E, E * 2])
                return (torch._C._nn.linear(query, w_q, b_q),) + torch._C._nn.linear(key, w_kv, b_kv).chunk(2, dim=-1)
        else:
            w_q, w_k, w_v = w.chunk(3)
            if b is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = b.chunk(3)
            return torch._C._nn.linear(query, w_q, b_q), torch._C._nn.linear(key, w_k, b_k), torch._C._nn.linear(value, w_v, b_v)    
    def forward(self, query, key, value, attn_mask=None, need_weights=True):
        
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(self.num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // self.num_heads
        
        q, k, v = self.in_projection(query, key, value, self.in_proj_weight, self.in_proj_bias)
        
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        # update source sequence length after adjustments
        src_len = k.size(1)
        # adjust dropout probability
        dropout_p = self.dropout
        if not self.training:
            dropout_p = 0.0
        #
        # (deep breath) calculate attention and out projection
        #
        attn_output, attn_output_weights = self.attention(q, k, v, dropout_p, attn_mask)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None
        
    def attention(
        self, 
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.0,
        attn_mask=None
    ):
        B, Nt, E = q.shape
        q = q / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = torch.bmm(q, k.transpose(-2, -1))
        if attn_mask is not None:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn += new_attn_mask
        attn = torch.sigmoid(attn)
        if dropout_p > 0.0:
            attn = dropout(attn, p=dropout_p)
        output = torch.bmm(attn / (attn.sum(dim=-1)[..., None] + 1e-4), v)
        return output, attn

class binary_head(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, aux_loss=True, dropout=0.1):
        super(binary_head, self).__init__()
        self.num_heads = num_heads
        self.center = mean_farthest_assignment(hidden_dim)
        self.field  = SMHA(hidden_dim, num_heads)
        # self.energy = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.aux_loss = aux_loss
    
    def get_output(self, hs_pair, centers, layer=-1):
        L, N, Q, C  = hs_pair.shape
        base_idx     = (torch.arange(Q - 1)[None, ...].expand(Q, -1) + torch.arange(Q)[:, None].expand(-1, Q - 1)) % Q

        mean_field, assignment = self.field(centers[layer].permute(1, 0, 2), hs_pair[layer].permute(1, 0, 2), hs_pair[layer].permute(1, 0, 2)) # 2, N, C; N, 2, Q: output_1_score
        # assert torch.sum(torch.isnan(assignment)) < 1
        card = assignment.sum(-1) # N, 2
        s_idx, d_idx = card.argmin(-1), card.argmax(-1) # N; N: output_1_idx
        cd, cs = mean_field[d_idx, torch.arange(N), :], mean_field[s_idx, torch.arange(N), :] # N, C; N, C

        centers_rm   = self.center(hs_pair[layer, :, base_idx, :]).flatten(1, 2) # N, Q*2, C
        mask = torch.zeros(N * self.num_heads, Q * 2, Q, dtype=torch.bool, device=hs_pair.device) # N, Q, Q + 1
        mask[:, torch.arange(Q) * 2, range(Q)] = True
        mask[:, torch.arange(Q) * 2 + 1, range(Q)] = True
        mean_field_rm, assignment_rm = self.field(centers_rm.permute(1, 0, 2), hs_pair[layer].permute(1, 0, 2), hs_pair[layer].permute(1, 0, 2), attn_mask=mask) # Q*2, N, C; N, Q*2, Q
        mean_field_rm = mean_field_rm.view(Q, 2, N, C) # Q, 2, N, C
        assignment_rm = assignment_rm.view(N, Q, 2, Q) # N, Q, 2, Q
        card_rm = assignment_rm.sum(dim=-1) # N, Q, 2
        s_idx_rm, d_idx_rm = card_rm.argmin(-1), card_rm.argmax(-1) # N, Q; N, Q
        mean_field_rm = mean_field_rm.permute(1, 2, 0, 3) # 2, N, Q, C 
        mean_field_rm = mean_field_rm.flatten(1, 2) # 2, NQ, C
        s_idx_rm, d_idx_rm = s_idx_rm.flatten(), d_idx_rm.flatten() # NQ; NQ
        cd_rm, cs_rm = mean_field_rm[d_idx_rm, torch.arange(N * Q), :].view(N, Q, C), mean_field_rm[s_idx_rm, torch.arange(N * Q), :].view(N, Q, C) # N, Q, C; N, Q, C
        shift_rm = torch.cdist(cd_rm, cd[:, None, :], p=2).squeeze(-1) + torch.cdist(cs_rm, cs[:, None, :], p=2).squeeze(-1) # N, Q: output_2
        
        mean_pair = hs_pair[layer].mean(dim=1)[:, None, :] # N, 1, C
        centers_md = self.center(torch.cat([hs_pair[layer, :, base_idx, :], mean_pair[:, None, ...].expand(-1, Q, -1, -1)], dim=2)).flatten(1, 2) # N, Q, 2, C -> N, Q*2, C
        KV = torch.cat([hs_pair[layer], mean_pair], dim=1).permute(1, 0, 2) # N, Q+1, C
        mask = torch.zeros(N * self.num_heads, Q * 2, Q + 1, dtype=torch.bool, device=hs_pair.device) # N, Q, Q + 1
        mask[:, torch.arange(Q) * 2, range(Q)] = True
        mask[:, torch.arange(Q) * 2 + 1, range(Q)] = True
        mean_field_md, assignment_md = self.field(centers_md.permute(1, 0, 2), KV, KV, attn_mask=mask)
        mean_field_md = mean_field_md.view(Q, 2, N, C) # Q, 2, N, C
        assignment_md = assignment_md.view(N, Q, 2, Q + 1) # N, Q, 2, Q
        card_md = assignment_md.sum(dim=-1) # N, Q, 2
        s_idx_md, d_idx_md = card_md.argmin(-1), card_md.argmax(-1) # N, Q; N, Q
        mean_field_md = mean_field_md.permute(1, 2, 0, 3) # 2, N, Q, C 
        mean_field_md = mean_field_md.flatten(1, 2) # 2, NQ, C
        s_idx_md, d_idx_md = s_idx_md.flatten(), d_idx_md.flatten() # NQ; NQ
        cd_md, cs_md = mean_field_md[d_idx_md, torch.arange(N * Q), :].view(N, Q, C), mean_field_md[s_idx_md, torch.arange(N * Q), :].view(N, Q, C) # N, Q, C; N, Q, C
        shift_md = torch.cdist(cd_md, cd[:, None, :], p=2).squeeze(-1) + torch.cdist(cs_md, cs[:, None, :], p=2).squeeze(-1) # N, Q: output_2
        
        out  = {
            'field_logits': assignment, 'field_logits_sidx': s_idx, 'field_logits_didx': d_idx,
            'rm_logits': shift_rm, 'md_logits': shift_md,
        }
        return out
        
    def forward(self, hs_pair):

        L, N, Q, C  = hs_pair.shape
        centers     = self.center(hs_pair) # L, N, 2, C
        #####TODO
        out = {}
        out.update(self.get_output(hs_pair, centers, -1))
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(hs_pair, centers)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, hs_pair, centers):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [self.get_output(hs_pair, centers, i) for i in range(hs_pair.shape[0] - 1)]
    
class mini_class_head(nn.Module):
    def __init__(self, num_obj_classes=80, hidden_dim=256, binary=False, aux_loss=True):
        super(mini_class_head, self).__init__()
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        self.obj_bbox_embed.requires_grad = False
        self.sub_bbox_embed.requires_grad = False
        self.obj_class_embed.requires_grad = False
        self.binary = binary
        if self.binary:
            self.binary_class_embed = nn.Linear(hidden_dim, 2)
        self.aux_loss = aux_loss
    def forward(self, hs_pair):
        outputs_obj_class  = self.obj_class_embed(hs_pair) # (6, N, Q, 80)
        outputs_obj_coord  = self.obj_bbox_embed(hs_pair).sigmoid() # (6, N, Q, 4)
        outputs_sub_coord  = self.sub_bbox_embed(hs_pair).sigmoid() # (6, N, Q, 4)
        
        out = {'pred_obj_logits': outputs_obj_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1],} 
        outputs_binary_class = None
        if self.binary:
            outputs_binary_class = self.binary_class_embed(hs_pair)
            out['pred_binary_logits'] = outputs_binary_class[-1]
            
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_sub_coord, outputs_obj_coord, outputs_binary_class)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_sub_coord, outputs_obj_coord, outputs_binary_class=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if not self.binary:
            return [{'pred_obj_logits': a, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                    for a, c, d in zip_longest(outputs_obj_class[:-1], outputs_sub_coord[:-1], outputs_obj_coord[:-1])]
        else:
            return [{'pred_obj_logits': a, 'pred_sub_boxes': c, 'pred_obj_boxes': d, 'pred_binary_logits': e}
                    for a, c, d, e in zip_longest(outputs_obj_class[:-1], outputs_sub_coord[:-1], outputs_obj_coord[:-1], outputs_binary_class[:-1])]

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, verb_loss_type, dataset_file, alpha, obj_reweight, verb_reweight, use_static_weights, queue_size, p_obj, p_verb):
        super().__init__()

        assert verb_loss_type == 'bce' or verb_loss_type == 'focal'

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        # self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.verb_loss_type = verb_loss_type
        
        self.alpha = alpha

        if dataset_file == 'hico':
            self.obj_nums_init = [1811, 9462, 2415, 7249, 1665, 3587, 1396, 1086, 10369, 800, \
                                  287, 77, 332, 2352, 974, 470, 1386, 4889, 1675, 1131, \
                                  1642, 185, 92, 717, 2228, 4396, 275, 1236, 1447, 1207, \
                                  2949, 2622, 1689, 2345, 1863, 408, 5594, 1178, 562, 1479, \
                                  988, 1057, 419, 1451, 504, 177, 1358, 429, 448, 186, \
                                  121, 441, 735, 706, 868, 1238, 1838, 1224, 262, 517, \
                                  5787, 200, 529, 1337, 146, 272, 417, 1277, 31, 213, \
                                  7, 102, 102, 2424, 606, 215, 509, 529, 102, 572]
        else:
            self.obj_nums_init = [5397, 238, 332, 321, 5, 6, 45, 90, 59, 20, \
                                  13, 5, 6, 313, 28, 25, 46, 277, 20, 16, \
                                  154, 0, 7, 13, 356, 191, 458, 66, 337, 1364, \
                                  1382, 958, 1166, 68, 258, 221, 1317, 1428, 759, 201, \
                                  190, 444, 274, 587, 124, 107, 102, 37, 226, 16, \
                                  30, 22, 187, 320, 222, 465, 893, 213, 56, 322, \
                                  306, 13, 55, 834, 23, 104, 38, 861, 11, 27, \
                                  0, 16, 22, 405, 50, 14, 145, 63, 9, 11]
        self.obj_nums_init.append(3 * sum(self.obj_nums_init))  # 3 times fg for bg init

        if dataset_file == 'hico':
            self.verb_nums_init = [67, 43, 157, 321, 664, 50, 232, 28, 5342, 414, \
                                   49, 105, 26, 78, 157, 408, 358, 129, 121, 131, \
                                   275, 1309, 3, 799, 2338, 128, 633, 79, 435, 1, \
                                   905, 19, 319, 47, 816, 234, 17958, 52, 97, 648, \
                                   61, 1430, 13, 1862, 299, 123, 52, 328, 121, 752, \
                                   111, 30, 293, 6, 193, 32, 4, 15421, 795, 82, \
                                   30, 10, 149, 24, 59, 504, 57, 339, 62, 38, \
                                   472, 128, 672, 1506, 16, 275, 16092, 757, 530, 380, \
                                   132, 68, 20, 111, 2, 160, 3209, 12246, 5, 44, \
                                   18, 7, 5, 4815, 1302, 69, 37, 25, 5048, 424, \
                                   1, 235, 150, 131, 383, 72, 76, 139, 258, 464, \
                                   872, 360, 1917, 1, 3775, 1206, 1]
        else:
            self.verb_nums_init = [4001, 4598, 1989, 488, 656, 3825, 367, 367, 677, 677, \
                                   700, 471, 354, 498, 300, 313, 300, 300, 622, 458, \
                                   500, 498, 489, 1545, 133, 142, 38, 116, 388]
        self.verb_nums_init.append(3 * sum(self.verb_nums_init))
        
        self.obj_reweight       = obj_reweight
        self.verb_reweight      = verb_reweight
        self.use_static_weights = use_static_weights
        
        Maxsize = queue_size

        if self.obj_reweight:
            self.q_obj = Queue(maxsize=Maxsize)
            self.p_obj = p_obj
            self.obj_weights_init = self.cal_weights(self.obj_nums_init, p=self.p_obj)

        if self.verb_reweight:
            self.q_verb = Queue(maxsize=Maxsize)
            self.p_verb = p_verb
            self.verb_weights_init = self.cal_weights(self.verb_nums_init, p=self.p_verb)

    def cal_weights(self, label_nums, p=0.5):
        num_fgs = len(label_nums[:-1])
        weight = np.zeros(num_fgs + 1)
        num_all = sum(label_nums[:-1])
        
        bottom = np.array(label_nums[:num_fgs])
        idx    = np.where(bottom > 0)[0]
        weight[idx] = np.power(num_all / bottom[idx], p)
        weight = weight / np.mean(weight[weight > 0])

        weight[-1] = np.power(num_all / label_nums[-1], p) if label_nums[-1] != 0 else 0

        weight = torch.FloatTensor(weight).cuda()
        return weight
    
    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        src_logits = outputs['pred_obj_logits']
        if src_logits is None:
            return None

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        if not self.obj_reweight:
            obj_weights = self.empty_weight
        elif self.use_static_weights:
            obj_weights = self.obj_weights_init
        else:
            obj_label_nums_in_batch = [0] * (self.num_obj_classes + 1)
            for target_class in target_classes:
                for label in target_class:
                    obj_label_nums_in_batch[label] += 1

            if self.q_obj.full():
                self.q_obj.get()
            self.q_obj.put(np.array(obj_label_nums_in_batch))
            accumulated_obj_label_nums = np.sum(self.q_obj.queue, axis=0)
            obj_weights = self.cal_weights(accumulated_obj_label_nums, p=self.p_obj)

            aphal = min(math.pow(0.999, self.q_obj.qsize()), 0.9)
            obj_weights = aphal * self.obj_weights_init + (1 - aphal) * obj_weights
        
        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, obj_weights)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        if pred_logits is None:
            return None
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses
        
    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        src_logits = outputs['pred_verb_logits']
        if src_logits is None:
            return None

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        if not self.verb_reweight:
            verb_weights = None
        elif self.use_static_weights:
            verb_weights = self.verb_weights_init
        else:
            verb_label_nums_in_batch = [0] * (self.num_verb_classes + 1)
            for target_class in target_classes:
                for label in target_class:
                    label_classes = torch.where(label > 0)[0]
                    if len(label_classes) == 0:
                        verb_label_nums_in_batch[-1] += 1
                    else:
                        for label_class in label_classes:
                            verb_label_nums_in_batch[label_class] += 1

            if self.q_verb.full():
                self.q_verb.get()
            self.q_verb.put(np.array(verb_label_nums_in_batch))
            accumulated_verb_label_nums = np.sum(self.q_verb.queue, axis=0)
            verb_weights = self.cal_weights(accumulated_verb_label_nums, p=self.p_verb)

            aphal = min(math.pow(0.999, self.q_verb.qsize()),0.9)
            verb_weights = aphal * self.verb_weights_init + (1 - aphal) * verb_weights
        
        if self.verb_loss_type == 'bce':
            loss_verb_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes)
        elif self.verb_loss_type == 'focal':
            src_logits = src_logits.sigmoid()
            loss_verb_ce = self._neg_loss(src_logits, target_classes, weights=verb_weights, alpha=self.alpha)

        losses = {'loss_verb_ce': loss_verb_ce}
        return losses
        
    def loss_binary_labels(self, outputs, targets, indices, num_interactions):
        src_logits = outputs['pred_binary_logits']
        if src_logits is None:
            return None
        idx                 = self._get_src_permutation_idx(indices)
        target_classes_b    = torch.cat([t['binary_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes      = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)
        # print(target_classes.shape, src_logits.shape, idx)
        # assert 0
        target_classes[idx] = target_classes_b

        loss_binary_ce      = F.cross_entropy(src_logits.transpose(1, 2), target_classes)

        losses = {'loss_binary_ce': loss_binary_ce}
        return losses

    def loss_field_binary(self, outputs, targets, indices, num_interactions):
        N, _, Q = outputs['field_logits'].shape
        pos_logits = outputs['field_logits'][range(N), outputs['field_logits_sidx'], :]
        neg_logits = outputs['field_logits'][range(N), outputs['field_logits_didx'], :]
        src_logits = torch.cat([neg_logits[..., None], pos_logits[..., None]], dim=-1)
        idx                 = self._get_src_permutation_idx(indices)
        target_classes_b    = torch.cat([t['binary_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes      = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_b
        loss_binary_ce      = F.cross_entropy(src_logits.transpose(1, 2), target_classes)

        losses = {'loss_field_binary': loss_binary_ce}
        return losses
        
    def loss_cluster_binary(self, outputs, targets, indices, num_interactions):
        N, _, Q = outputs['field_logits'].shape
        idx                 = self._get_src_permutation_idx(indices)
        src_logits          = outputs['field_logits'].clamp(1e-3, 1-1e-3)
        src_matrix          = torch.bmm(src_logits.permute(0, 2, 1), src_logits).clamp(1e-7, 1-1e-7)
        target_classes_b    = torch.cat([t['binary_labels'][J] for t, (_, J) in zip(targets, indices)]).to(src_logits.dtype)
        target_classes      = torch.full((N, Q), 0, dtype=src_logits.dtype, device=src_logits.device)
        target_classes[idx] = target_classes_b
        sig_matrix          = torch.bmm(target_classes[..., None], target_classes[:, None, :])
        loss_matrix         = -(torch.log(src_matrix) * sig_matrix + torch.log(1 - src_matrix) * (1 - sig_matrix))
        losses = {'loss_cluster_binary': loss_matrix.mean()}
        return losses
        
    @torch.no_grad()
    def loss_bin_cardinality(self, outputs, targets, indices, num_interactions):
        N, _, Q = outputs['field_logits'].shape
        pos_logits = outputs['field_logits'][range(N), outputs['field_logits_sidx'], :]
        device = pos_logits.device
        tgt_lengths = torch.as_tensor([len(v['binary_labels']) for v in targets], device=device)
        card_pos = pos_logits.sum(1)
        card_err = F.l1_loss(card_pos.float(), tgt_lengths.float())
        losses = {'bin_cardinality_error': card_err}
        return losses
    
    def loss_shift_binary(self, outputs, targets, indices, num_interactions):
        N, Q = outputs['rm_logits'].shape
        idx                 = self._get_src_permutation_idx(indices)
        target_classes_b    = torch.cat([t['binary_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes      = torch.full(outputs['rm_logits'].shape[:2], 0, dtype=torch.int64, device=outputs['rm_logits'].device)
        target_classes[idx] = target_classes_b
        pos_logits = (outputs['rm_logits'] * target_classes).sum(dim=1)
        neg_logits = (outputs['rm_logits'] * (1 - target_classes)).sum(dim=1)
        pos_logits = pos_logits / (target_classes.sum(dim=1) + 1.)
        neg_logits = neg_logits / ((1 - target_classes).sum(dim=1) + 1.)

        losses = {'loss_shift_binary': (neg_logits - pos_logits).mean()}
        return losses
    
    def loss_comp_binary(self, outputs, targets, indices, num_interactions):
        N, Q = outputs['md_logits'].shape
        idx                 = self._get_src_permutation_idx(indices)
        target_classes_b    = torch.cat([t['binary_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes      = torch.full(outputs['md_logits'].shape[:2], 0, dtype=torch.int64, device=outputs['md_logits'].device)
        target_classes[idx] = target_classes_b
        pos_logits = (outputs['md_logits'] * target_classes).sum(dim=1)
        neg_logits = (outputs['md_logits'] * (1 - target_classes)).sum(dim=1)
        pos_logits = pos_logits / (target_classes.sum(dim=1) + 1.)
        neg_logits = neg_logits / ((1 - target_classes).sum(dim=1) + 1.)

        losses = {'loss_comp_binary': (neg_logits - pos_logits).mean()}
        return losses
        
    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        if outputs['pred_sub_boxes'] is None or outputs['pred_obj_boxes'] is None:
            return None
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses
        
    def loss_sub_boxes(self, outputs, targets, indices, num_interactions):
        if outputs['pred_sub_boxes'] is None:
            return None
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
        return losses
    
    def loss_obj_boxes(self, outputs, targets, indices, num_interactions):
        if outputs['pred_obj_boxes'] is None:
            return None
        idx = self._get_src_permutation_idx(indices)
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_obj_boxes.shape[0] == 0:
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def _neg_loss(self, pred, gt, weights=None, alpha=0.25):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        if weights is not None:
            pos_loss = pos_loss * weights[:-1]
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        num_pos  = pos_inds.float().sum()
        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'verb_labels': self.loss_verb_labels,
            'sub_obj_boxes': self.loss_sub_obj_boxes,
            'sub_boxes': self.loss_sub_boxes,
            'obj_boxes': self.loss_obj_boxes,
            'binary_labels': self.loss_binary_labels,
            'loss_field_binary': self.loss_field_binary,
            'bin_cardinality': self.loss_bin_cardinality,
            'loss_cluster_binary': self.loss_cluster_binary,
            'loss_shift_binary': self.loss_shift_binary,
            'loss_comp_binary': self.loss_comp_binary,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets, indices):
        for v in targets:
            v['obj_labels'] = v['obj_labels'][v['pair2obj']]
            v['obj_boxes']  = v['obj_boxes'][v['pair2obj']]
            v['sub_boxes']  = v['sub_boxes'][v['pair2sub']]

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        obj_indices_final, sub_indices_final, pair_indices_final = indices['indices'], indices['indices'], indices['indices']

        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if 'obj' in loss:
                losses.update(self.get_loss(loss, outputs, targets, obj_indices_final, num_interactions))
            elif 'sub' in loss:
                losses.update(self.get_loss(loss, outputs, targets, sub_indices_final, num_interactions))
            else:
                losses.update(self.get_loss(loss, outputs, targets, pair_indices_final, num_interactions))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                obj_indices, sub_indices, pair_indices = indices['indices_{}'.format(i)], indices['indices_{}'.format(i)], indices['indices_{}'.format(i)]
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'obj' in loss: 
                        l_dict = self.get_loss(loss, aux_outputs, targets, obj_indices, num_interactions, **kwargs)
                    elif 'sub' in loss:
                        l_dict = self.get_loss(loss, aux_outputs, targets, sub_indices, num_interactions, **kwargs)
                    else:
                        l_dict = self.get_loss(loss, aux_outputs, targets, pair_indices, num_interactions, **kwargs)
                    if l_dict is not None:
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

        return losses

class PostProcessHOI(nn.Module):

    def __init__(self, subject_category_id, binary=False, pnms=-1, aux_outputs=False):
        super().__init__()
        self.subject_category_id = subject_category_id
        self.binary = binary
        self.aux_outputs = aux_outputs

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['pred_obj_logits'], outputs['pred_verb_logits'], outputs['pred_sub_boxes'], outputs['pred_obj_boxes']
        N, Q = outputs['pred_obj_logits'].shape[:2]
        # out_field_logits, out_shift_logits, out_comp_logits = outputs['field_logits'][range(N), outputs['field_logits_sidx'], :], outputs['rm_logits'], outputs['md_logits']
        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        verb_scores      = out_verb_logits.sigmoid()
        # out_shift_logits = out_shift_logits
        # out_comp_logits  = out_comp_logits
        if self.binary:
            binary_scores = outputs['pred_binary_logits'].softmax(-1)[:, -1]
            

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for idx in range(len(obj_scores)):
            os, ol, vs, sb, ob = obj_scores[idx], obj_labels[idx], verb_scores[idx], sub_boxes[idx], obj_boxes[idx]
            # fs, ss, cs = out_field_logits[idx], out_shift_logits[idx], out_comp_logits[idx]
            if os.shape[0] != vs.shape[0]:
                os = os[:, None].expand(-1, self.num_hum).flatten(0, 1)
                ob = ob[:, None, ...].expand(-1, self.num_hum, -1).flatten(0, 1)
                ol = ol[:, None].expand(-1, self.num_hum).flatten(0, 1)
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            vs = vs * os.unsqueeze(1)

            ids = torch.arange(b.shape[0])
            if self.binary:
                results[-1].update({'bin_scores': binary_scores})
            results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2], 'obj_ids': ids[ids.shape[0] // 2:], 'obj_scores': os.to('cpu'),})
                                # 'field_scores': fs.to('cpu'), 'rm_scores': ss.to('cpu'), 'md_scores': cs.to('cpu')})

        if self.aux_outputs:
            for layer in range(len(outputs['aux_outputs'])):
                out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['aux_outputs'][layer]['pred_obj_logits'], outputs['aux_outputs'][layer]['pred_verb_logits'], outputs['aux_outputs'][layer]['pred_sub_boxes'], outputs['aux_outputs'][layer]['pred_obj_boxes']
                
                assert len(out_obj_logits) == len(target_sizes)
                assert target_sizes.shape[1] == 2

                obj_prob = F.softmax(out_obj_logits, -1)
                obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

                verb_scores = out_verb_logits.sigmoid()

                img_h, img_w = target_sizes.unbind(1)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
                sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
                sub_boxes = sub_boxes * scale_fct[:, None, :]
                obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
                obj_boxes = obj_boxes * scale_fct[:, None, :]
                for idx in range(len(obj_scores)):
                    os, ol, vs, sb, ob = obj_scores[idx], obj_labels[idx], verb_scores[idx], sub_boxes[idx], obj_boxes[idx]
                    sl = torch.full_like(ol, self.subject_category_id)
                    l = torch.cat((sl, ol))
                    b = torch.cat((sb, ob))
                    results[idx].update({'labels' + f'_{layer}': l.to('cpu'), 'boxes' + f'_{layer}': b.to('cpu')})
                    
                    vs = vs * os.unsqueeze(1)

                    results[idx].update({'verb_scores' + f'_{layer}': vs.to('cpu'), 'obj_scores' + f'_{layer}': os.to('cpu')})
                    if self.binary:
                        results[idx].update({'binary_scores' + f'_{layer}': bs.to('cpu')})

        return results

class hierarchical_cluster_assignment(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
    
    def forward(self, hs_pair):
        centers = hs_pair.clone()
        L, N, Q, C = centers.shape
        base_idx = torch.arange(L * N).view(L, N)
        base_x, base_y = base_idx // N, base_idx % N
        # Hierarchical Clustering
        for iter in range(Q - 2):
            L, N, Q, C = centers.shape
            distance = torch.cdist(centers, centers, 2)
            distance[:, :, range(Q), range(Q)] += torch.max(distance)
            index = distance.flatten(2).argmin(2)
            x, y = index // Q, index % Q
            centers[base_x, base_y, x] = (centers[base_x, base_y, x] + centers[base_x, base_y, y]) / 2
            centers[base_x, base_y, y] = centers[base_x, base_y, -1]
            centers = centers[:, :, :-1]
        return centers
        
class mean_farthest_assignment(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
    
    def forward(self, hs_pair):
        L, N, Q, C = hs_pair.shape
        c1 = torch.mean(hs_pair, dim=2)[:, :, None, :]
        d = torch.cdist(c1, hs_pair, p=2).squeeze().argmax(-1).flatten(0, 1)
        c2 = hs_pair.flatten(0, 1)[range(L * N), d, :].view(L, N, 1, -1)
        # print(c1.shape, c2.shape)
        centers = torch.cat([c1, c2], dim=2)
        return centers

class kmeans_cluster_assignment(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
    def forward(self, hs_pair, centers):
        # KMeans
        L, N, Q, C = hs_pair.shape
        assignment = -torch.cdist(hs_pair, centers, 2)
        for iter in range(Q):
            assignment = -torch.cdist(hs_pair, centers, 2)
            assignment = torch.softmax(assignment, -1)
            centers[:, :, 0] = (assignment[:, :, :, :1] * hs_pair).sum(-2) / (torch.sum(assignment[:, :, :, 0], dim=-1)[..., None] + 1e-7)
            centers[:, :, 1] = (assignment[:, :, :, 1:] * hs_pair).sum(-2) / (torch.sum(assignment[:, :, :, 1], dim=-1)[..., None] + 1e-7)
        return centers, assignment

class gaussian_divergence(nn.Module):
    def __init__(self, trans=None, hidden_dim=256):
        super().__init__()
        if trans == 'LIN':
            self.trans = nn.Linear(hidden_dim, hidden_dim)
        elif trans == 'MLP':
            self.trans = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        else:
            self.trans = nn.Identity()
    
    def forward(self, hs_pair):
        hs_pair   = self.trans(hs_pair)
        L, N, Q, C = hs_pair.shape
        var, mean = torch.var_mean(hs_pair, -2)
        base_idx  = (torch.arange(Q - 1)[None, ...].expand(Q, -1) + torch.arange(Q)[:, None].expand(-1, Q - 1)) % Q
        rm_var, rm_mean = torch.var_mean(hs_pair[:, :, base_idx, :], -2) # L, N, Q, C
        rm_dist = torch.distributions.normal.Normal(rm_mean, rm_var)
        dist    = torch.distributions.normal.Normal(mean[:, :, None, :], var[:, :, None, :])
        kl = torch.distributions.kl.kl_divergence(dist, rm_dist).mean(dim=-1)
        return kl

class pair_differential_module(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.class_head = mini_class_head(
            num_obj_classes=args.num_obj_classes,
            aux_loss=args.aux_loss,
            binary=args.binary,
        )
        # self.class_head.requires_grad = False
        # self.binary_head = nn.Linear(args.hidden_dim, 2)
        # self.binary_head = binary_head(hidden_dim=args.hidden_dim, num_heads=args.nheads, aux_loss=args.aux_loss, dropout=args.dropout)
    
    def forward(self, feat, pred_verb_logits):
        output = self.class_head(feat)
        output['pred_verb_logits'] = pred_verb_logits[-1]
        # output_bin = self.binary_head(feat)
        # for key in output_bin:
        #     if key not in ['aux_outputs']:
        #         output[key] = output_bin[key]
        #     else:
        for i in range(len(output['aux_outputs'])):
            output['aux_outputs'][i].update({'pred_verb_logits': pred_verb_logits[i]})
        return output

def build(args):
    device = torch.device(args.device)
    model  = pair_differential_module(args)
    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_obj_ce']   = args.obj_loss_coef
    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    weight_dict['loss_comp_binary']      = args.bin_loss_coef * 0
    weight_dict['loss_shift_binary']     = args.bin_loss_coef * 0
    weight_dict['bin_cardinality_error'] = args.bin_loss_coef
    weight_dict['loss_cluster_binary']   = args.bin_loss_coef
    weight_dict['loss_field_binary']     = args.bin_loss_coef
    weight_dict['loss_binary_ce']     = args.bin_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    losses = ['obj_labels', 'sub_boxes', 'obj_boxes', 'obj_cardinality', 'binary_labels']# 'loss_comp_binary', 'loss_shift_binary', 'bin_cardinality', 'loss_cluster_binary', 'loss_field_binary']
    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses, verb_loss_type=args.verb_loss_type, dataset_file=args.dataset_file, alpha=args.alpha, obj_reweight=args.obj_reweight, verb_reweight=args.verb_reweight, use_static_weights=args.use_static_weights, queue_size=args.queue_size, p_obj=args.p_obj, p_verb=args.p_verb)
    criterion.to(device)
    postprocessors = {'hoi': PostProcessHOI(args.subject_category_id, args.binary, args.pnms, args.aux_outputs)}

    return model, criterion, postprocessors

if __name__ == '__main__':
    q = torch.rand((64, 4, 256))
    k = torch.rand((64, 4, 256))
    v = torch.rand((64, 4, 256))
    
    attn = SMHA(256, 8)
    x, y = attn(q, k, v)
    print(x.shape, y.shape)
    
    head = mini_class_head(aux_loss=False)
    output = head(v)
    for key in output.keys():
        print(key, output[key].shape)
    
    hs_pair = torch.rand((6, 4, 64, 256))
    hier = mean_farthest_assignment()
    kmeans = kmeans_cluster_assignment()
    init_c = hier(hs_pair)
    print(init_c.shape)
    final_c, assign = kmeans(hs_pair, init_c)
    print(final_c.shape, assign.shape)
    
    binary = binary_head(aux_loss=False)
    out = binary(hs_pair)
    for key in out.keys():
        print(key, out[key].shape)
    
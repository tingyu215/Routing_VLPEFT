#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        # if lora_dropout > 0.:
        #     self.lora_dropout = nn.Dropout(p=lora_dropout)
        # else:
        #     self.lora_dropout = lambda x: x
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True
        
    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(
                    x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse
                )
                result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)
            

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    delta_w = F.conv1d(
                        self.lora_A.data.unsqueeze(0), 
                        self.lora_B.data.unsqueeze(-1), 
                        groups=sum(self.enable_lora)
                    ).squeeze(0)
                    self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    delta_w = F.conv1d(
                        self.lora_A.data.unsqueeze(0), 
                        self.lora_B.data.unsqueeze(-1), 
                        groups=sum(self.enable_lora)
                    ).squeeze(0)
                    self.weight.data += self.zero_pad(T(delta_w * self.scaling))
                self.merged = True        

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(
                    after_A.transpose(-2, -1), 
                    self.lora_B.unsqueeze(-1), 
                    groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result



class MergedLinearOnlyVL(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    # Apply routing function in the high-dim space (does not work)
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        add_relu: bool=False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
        
        self.add_relu = add_relu

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

            # nn.init.kaiming_uniform_(self.lora_A_vis, a=math.sqrt(5))
            # nn.init.zeros_(self.lora_B_vis)

    def zero_pad(self, x):
        # import ipdb
        # ipdb.set_trace()
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    delta_w = F.conv1d(
                        self.lora_A.data.unsqueeze(0), 
                        self.lora_B.data.unsqueeze(-1), 
                        groups=sum(self.enable_lora)
                    ).squeeze(0)
                    self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    delta_w = F.conv1d(
                        self.lora_A.data.unsqueeze(0), 
                        self.lora_B.data.unsqueeze(-1), 
                        groups=sum(self.enable_lora)
                    ).squeeze(0)
                    self.weight.data += self.zero_pad(T(delta_w * self.scaling))
                self.merged = True        

    def forward(self, x: torch.Tensor, vis_feat:torch.Tensor=None, multiply_ones=False):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(
                    after_A.transpose(-2, -1), 
                    self.lora_B.unsqueeze(-1), 
                    groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                # result += self.zero_pad(after_B) * self.scaling

                # vis and text share lora_A & lora_B
                after_A_vis = F.linear(self.lora_dropout(vis_feat), self.lora_A)
                after_B_vis = F.conv1d(
                    after_A_vis.transpose(-2, -1), 
                    self.lora_B.unsqueeze(-1), 
                    groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                bsz, vis_len, vis_dim  = after_B_vis.shape
                if multiply_ones:
                    ones_tensor = torch.ones((bsz, vis_len, vis_dim)).to(after_B_vis.device)
                    if not self.add_relu:
                        vis_mat = after_B @ after_B_vis.reshape(bsz, vis_dim, vis_len) @ ones_tensor
                    else:
                        vis_mat = torch.nn.functional.relu(after_B) @ torch.nn.functional.relu(after_B_vis.reshape(bsz, vis_dim, vis_len)) @ ones_tensor
                else:
                    if not self.add_relu:
                        vis_mat = after_B @ after_B_vis.reshape(bsz, vis_dim, vis_len) @ after_B_vis
                    else:
                        vis_mat = torch.nn.functional.relu(after_B) @ torch.nn.functional.relu(after_B_vis.reshape(bsz, vis_dim, vis_len)) @ after_B_vis

                result += self.zero_pad(vis_mat) * self.scaling

            return result




class MergedLinearOnlyVLBeforeB(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    # Routing functions applied in the low-rank bottleneck
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        add_relu: bool = False,
        element_mul = False,
        element_add = False,
        element_mul_expand = False,
        routing_xt = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
        self.add_relu = add_relu

        # choices of routing functions
        self.element_mul = element_mul
        self.element_add = element_add
        self.element_mul_expand = element_mul_expand
        self.routing_xt = routing_xt

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    delta_w = F.conv1d(
                        self.lora_A.data.unsqueeze(0), 
                        self.lora_B.data.unsqueeze(-1), 
                        groups=sum(self.enable_lora)
                    ).squeeze(0)
                    self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    delta_w = F.conv1d(
                        self.lora_A.data.unsqueeze(0), 
                        self.lora_B.data.unsqueeze(-1), 
                        groups=sum(self.enable_lora)
                    ).squeeze(0)
                    self.weight.data += self.zero_pad(T(delta_w * self.scaling))
                self.merged = True        

    def forward(self, x: torch.Tensor, vis_feat:torch.Tensor=None, multiply_ones=False):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)

                # vis and text share lora_A & lora_B
                after_A_vis = F.linear(self.lora_dropout(vis_feat), self.lora_A)
                bsz, vis_len, vis_dim  = after_A_vis.shape
                if multiply_ones:
                    # multiply with all-ones tensor, leads to bad results
                    ones_tensor = torch.ones((bsz, vis_len, vis_dim)).to(after_A_vis.device)
                    if not self.add_relu:
                        after_A_vis = after_A @ after_A_vis.reshape(bsz, vis_dim, vis_len) @ ones_tensor
                    else:
                        after_A_vis = torch.nn.functional.relu(after_A) @ torch.nn.functional.relu(after_A_vis.reshape(bsz, vis_dim, vis_len)) @ ones_tensor
                else:
                    if self.element_mul:
                        # routing function: x_t \circ x'_v
                        after_A_vis = after_A * after_A_vis
                    elif self.element_add:
                        # routing function: x_t + x'_v
                        after_A_vis = after_A + after_A_vis.expand_as(after_A)
                    elif self.element_mul_expand:
                        # routing function: x_t  x''_v
                        after_A_vis = after_A @ after_A_vis.expand(after_A_vis.shape[0], after_A_vis.shape[2], after_A_vis.shape[2])
                    elif self.routing_xt:
                        # routing function: x_t (x_t)^T  x_v, does not work very well, yet not fully explored
                        after_A_vis = after_A @ after_A.reshape(after_A.shape[0], after_A.shape[2], after_A.shape[1]) @ after_A_vis.expand_as(after_A)
                    elif not self.add_relu:
                        # routing function: x_t (x_v)^T  x_v
                        after_A_vis = after_A @ after_A_vis.reshape(bsz, vis_dim, vis_len) @ after_A_vis
                    else:
                        # routing function: x_t (x_v)^T  x_v with ReLU
                        after_A_vis = torch.nn.functional.relu(after_A) @ torch.nn.functional.relu(after_A_vis.reshape(bsz, vis_dim, vis_len)) @ after_A_vis

                after_B_vis = F.conv1d(
                    after_A_vis.transpose(-2, -1), 
                    self.lora_B.unsqueeze(-1), 
                    groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                
                result += self.zero_pad(after_B_vis) * self.scaling

            return result
        


class MergedLinearSplit(nn.Module):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        add_relu: bool = False,
        **kwargs
    ):
        super().__init__()
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        
        self.lora_alpha = lora_alpha
        self.r = r
        self.out_features = out_features
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A_text = nn.Linear(in_features, r//4 * sum(enable_lora), bias=False)
            self.lora_A_img = nn.Linear(in_features, r//4 * sum(enable_lora), bias=False)

            self.lora_A_text_up = nn.Linear(r//4* sum(enable_lora), r * sum(enable_lora), bias=False)
            self.lora_A_img_up = nn.Linear(r//4* sum(enable_lora), r * sum(enable_lora), bias=False)

            self.lora_B = nn.Parameter(torch.zeros(out_features // len(enable_lora) * sum(enable_lora), r))

            self.scaling = self.lora_alpha / self.r
            # Compute the indices
            self.lora_ind = torch.zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()

        self.add_relu = add_relu

    def reset_parameters(self):
        # nn.Linear.reset_parameters(self)
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_A_text.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_img.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_text_up.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_img_up.weight, a=math.sqrt(5))

        nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features))

    def forward(self, x: torch.Tensor, vis_feat:torch.Tensor=None, multiply_ones=False):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        # result = F.linear(x, T(self.weight), bias=self.bias)
        if self.r > 0:
            after_A_text = self.lora_A_text(self.lora_dropout(x))
            after_A_text = self.lora_A_text_up(after_A_text)

            after_A_img = self.lora_A_img(self.lora_dropout(vis_feat))
            after_A_img = self.lora_A_img_up(after_A_img)

            # after_A = torch.cat((after_A_img,  after_A_text), dim=1)
            if multiply_ones:
                ones_tensor = torch.ones((after_A_img.size())).to(after_A_img.device)
                if not self.add_relu:
                    after_A = after_A_text @ after_A_img.reshape(after_A_img.size()[0], after_A_img.size()[2], after_A_img.size()[1]) @ ones_tensor
                else:
                    after_A = torch.nn.functional.relu(after_A_text) @ torch.nn.functional.relu(after_A_img.reshape(after_A_img.size()[0], after_A_img.size()[2], after_A_img.size()[1])) @ ones_tensor
            else:
                if not self.add_relu:
                    after_A = after_A_text @ after_A_img.reshape(after_A_img.size()[0], after_A_img.size()[2], after_A_img.size()[1]) @ after_A_img
                else:
                    after_A = torch.nn.functional.relu(after_A_text) @ torch.nn.functional.relu(after_A_img.reshape(after_A_img.size()[0], after_A_img.size()[2], after_A_img.size()[1])) @ after_A_img
            
            after_B = F.conv1d(
                after_A.transpose(-2, -1), 
                self.lora_B.unsqueeze(-1), 
                groups=sum(self.enable_lora)
            ).transpose(-2, -1)
            
            result = self.zero_pad(after_B) * self.scaling

        return result


class MergedLinearSplitMerge(nn.Module):
    # routing function with separate W_down matrices for image and text
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        add_relu: bool=False,
        **kwargs
    ):
        super().__init__()
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        
        self.lora_alpha = lora_alpha
        self.r = r
        self.out_features = out_features
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A_text = nn.Linear(in_features, r//4 * sum(enable_lora), bias=False)
            self.lora_A_img = nn.Linear(in_features, r//4 * sum(enable_lora), bias=False)

            self.lora_A_up = nn.Linear(r//4* sum(enable_lora), r * sum(enable_lora), bias=False)

            self.lora_B = nn.Parameter(torch.zeros(out_features // len(enable_lora) * sum(enable_lora), r))

            self.scaling = self.lora_alpha / self.r
            # Compute the indices
            self.lora_ind = torch.zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        self.add_relu = add_relu

    def reset_parameters(self):
        # nn.Linear.reset_parameters(self)
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_A_text.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_img.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_up.weight, a=math.sqrt(5))

        nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features))

    def forward(self, x: torch.Tensor, vis_feat:torch.Tensor=None, multiply_ones=False):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        # result = F.linear(x, T(self.weight), bias=self.bias)
        if self.r > 0:
            after_A_text = self.lora_A_text(self.lora_dropout(x))
            after_A_img = self.lora_A_img(self.lora_dropout(vis_feat))

            # after_A = torch.cat((after_A_img,  after_A_text), dim=1)
            if multiply_ones:
                ones_tensor = torch.ones((after_A_img.size())).to(after_A_img.device)
                if not self.add_relu:
                    after_A = after_A_text @ after_A_img.reshape(after_A_img.size()[0], after_A_img.size()[2], after_A_img.size()[1]) @ ones_tensor
                else:
                    after_A = torch.nn.functional.relu(after_A_text) @ torch.nn.functional.relu(after_A_img.reshape(after_A_img.size()[0], after_A_img.size()[2], after_A_img.size()[1])) @ ones_tensor
            else:
                if not self.add_relu:
                    after_A = after_A_text @ after_A_img.reshape(after_A_img.size()[0], after_A_img.size()[2], after_A_img.size()[1]) @ after_A_img
                else:
                    after_A = torch.nn.functional.relu(after_A_text) @ torch.nn.functional.relu(after_A_img.reshape(after_A_img.size()[0], after_A_img.size()[2], after_A_img.size()[1])) @ after_A_img

            after_A = self.lora_A_up(after_A)
            
            after_B = F.conv1d(
                after_A.transpose(-2, -1), 
                self.lora_B.unsqueeze(-1), 
                groups=sum(self.enable_lora)
            ).transpose(-2, -1)
            
            result = self.zero_pad(after_B) * self.scaling

        return result


class MergedLinearSplitMergeRes(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        add_relu: bool=False,
        **kwargs
    ):
        super().__init__()
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        
        self.lora_alpha = lora_alpha
        self.r = r
        self.out_features = out_features
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A_text = nn.Linear(in_features, r//4 * sum(enable_lora), bias=False)
            self.lora_A_img = nn.Linear(in_features, r//4 * sum(enable_lora), bias=False)

            self.lora_A_up = nn.Linear(r//4* sum(enable_lora), r * sum(enable_lora), bias=False)

            self.lora_B = nn.Parameter(torch.zeros(out_features // len(enable_lora) * sum(enable_lora), r))

            self.scaling = self.lora_alpha / self.r
            # Compute the indices
            self.lora_ind = torch.zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)

            self.layer_norm = nn.LayerNorm(normalized_shape=r//4* sum(enable_lora), eps=1e-5, elementwise_affine=True)
        self.reset_parameters()
        self.add_relu = add_relu

    def reset_parameters(self):
        # nn.Linear.reset_parameters(self)
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_A_text.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_img.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_up.weight, a=math.sqrt(5))

        nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features))

    def forward(self, x: torch.Tensor, vis_feat:torch.Tensor=None):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        # result = F.linear(x, T(self.weight), bias=self.bias)
        if self.r > 0:
            after_A_text = self.lora_A_text(self.lora_dropout(x))
            after_A_img = self.lora_A_img(self.lora_dropout(vis_feat))

            # after_A = torch.cat((after_A_img,  after_A_text), dim=1)
            if not self.add_relu:
                after_A = after_A_text @ after_A_img.reshape(after_A_img.size()[0], after_A_img.size()[2], after_A_img.size()[1]) @ after_A_img
            else:
                after_A = torch.nn.functional.relu(after_A_text) @ torch.nn.functional.relu(after_A_img.reshape(after_A_img.size()[0], after_A_img.size()[2], after_A_img.size()[1])) @ after_A_img

            # residual connection
            after_A = after_A + after_A_text
            after_A = self.layer_norm(after_A)

            after_A = self.lora_A_up(after_A)
            
            after_B = F.conv1d(
                after_A.transpose(-2, -1), 
                self.lora_B.unsqueeze(-1), 
                groups=sum(self.enable_lora)
            ).transpose(-2, -1)
            
            result = self.zero_pad(after_B) * self.scaling

        return result
        


class MergedLinearSplitAttn(nn.Module):
    # LoRA with cross-attention
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        num_heads:int=4,
        **kwargs
    ):
        super().__init__()
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        
        self.lora_alpha = lora_alpha
        self.r = r
        self.out_features = out_features
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A_text = nn.Linear(in_features, r//4 * sum(enable_lora), bias=False)
            self.lora_A_img = nn.Linear(in_features, r//4 * sum(enable_lora), bias=False)


            self.lora_A_text_query = nn.Linear(r//4 * sum(enable_lora), r//4 * sum(enable_lora), bias=False)
            self.lora_A_img_key = nn.Linear(r//4 * sum(enable_lora), r//4 * sum(enable_lora), bias=False)
            self.lora_A_img_value = nn.Linear(r//4 * sum(enable_lora), r//4 * sum(enable_lora), bias=False)

            # self.lora_A_img_key = nn.Linear(r//4 * sum(enable_lora)*1, r//4 * sum(enable_lora)*10, bias=False)
            # self.lora_A_img_value = nn.Linear(r//4 * sum(enable_lora)*1, r//4 * sum(enable_lora)*10, bias=False)

            self.intermediate_dim = r//4 * sum(enable_lora)

            self.lora_A_query_up = nn.Linear(r//4* sum(enable_lora), r * sum(enable_lora), bias=False)
            
            self.lora_A_attn = CrossAttnNoOutProj(num_heads=num_heads, embed_dim=r//4)

            self.lora_A_attn_out = nn.Linear(r//4* sum(enable_lora), r//4* sum(enable_lora), bias=False)
            
            self.lora_B = nn.Parameter(torch.zeros(out_features // len(enable_lora) * sum(enable_lora), r))

            self.scaling = self.lora_alpha / self.r
            # Compute the indices
            self.lora_ind = torch.zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.Linear.reset_parameters(self)
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_A_text.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_img.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_text_query.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_img_key.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_img_value.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_query_up.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_attn_out.weight, a=math.sqrt(5))

        nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features))

    def forward(self, x: torch.Tensor, vis_feat:torch.Tensor=None, multiply_ones=False):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        if self.r > 0:
            after_A = self.lora_A_text(self.lora_dropout(x))
            after_A_img = self.lora_A_img(self.lora_dropout(vis_feat))

            after_A_img_key = self.lora_A_img_key(self.lora_dropout(after_A_img))
            
            after_A_img_value = self.lora_A_img_value(self.lora_dropout(after_A_img))
            after_A = self.lora_A_text_query(after_A)

            after_A_img_key = after_A_img_key.reshape(after_A_img_key.size()[0],-1, self.intermediate_dim)
            after_A_img_value = after_A_img_value.reshape(after_A_img_value.size()[0],-1, self.intermediate_dim)
                
            after_A = self.lora_A_attn(q=after_A, k=after_A_img_key, v=after_A_img_value)
            after_A = after_A.reshape((after_A.size()[0], -1, self.intermediate_dim))

            after_A = self.lora_A_attn_out(self.lora_dropout(after_A))    
            after_A = self.lora_A_query_up(after_A)
            
            after_B = F.conv1d(
                after_A.transpose(-2, -1), 
                self.lora_B.unsqueeze(-1), 
                groups=sum(self.enable_lora)
            ).transpose(-2, -1)
            
            result = self.zero_pad(after_B) * self.scaling

        return result



class CrossAttnNoOutProj(nn.Module):
    def __init__(self, num_heads, embed_dim, **kwargs):
        super(CrossAttnNoOutProj, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
    def forward(self, q, k, v):
        q = q.reshape(q.size()[0], -1, self.num_heads, q.size()[-1]//self.num_heads)
        q = q.permute(0,2,1,3)
        k = k.reshape(k.size()[0], -1, self.num_heads, k.size()[-1]//self.num_heads)
        k = k.permute(0,2,1,3)
        v = v.reshape(v.size()[0], -1, self.num_heads, v.size()[-1]//self.num_heads)
        v = v.permute(0,2,1,3)
        values, _ = scaled_dot_product(q, k, v)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(q.size()[0], -1, self.embed_dim)
        return values


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x, 
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)

class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)

class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)

# Can Extend to other ones like this

class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)
    






if __name__ == "__main__":
    lora_layer = MergedLinearOnlyVL(in_features=768, out_features=768, r=128, enable_lora=[True, False, True], fan_in_fan_out=True, merge_weights=False)

    lora_layer(torch.randn((4,5,768)), torch.randn((4,197,768)))
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


class VLLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        add_relu: bool = False,
        element_mul = False,
        element_add = False,
        element_mul_expand = False,
        routing_xt = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        # self.r = r
        # self.lora_alpha = lora_alpha
        # self.lora_dropout = lora_dropout
        # self.merge_weights = merge_weights
        self.fan_in_fan_out = fan_in_fan_out
        self.add_relu = add_relu
        self.element_mul = element_mul
        self.element_add = element_add
        self.element_mul_expand = element_mul_expand
        self.routing_xt = routing_xt
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

    def forward(self, x: torch.Tensor, vis_feat:torch.Tensor=None, multiply_ones=False):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)
        if self.r > 0:
            after_A = F.linear(self.lora_dropout(x), self.lora_A)

            # vis and text share lora_A & lora_B
            after_A_vis = F.linear(self.lora_dropout(vis_feat), self.lora_A)
            bsz, vis_len, vis_dim  = after_A_vis.shape
            if multiply_ones:
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

            after_B_vis = F.linear(after_A_vis, self.lora_B)
                
            result += after_B_vis * self.scaling
                
        return result



class VLLinearSplit(nn.Module):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        add_relu: bool = False,
        **kwargs
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.merge_weights = merge_weights
        self.fan_in_fan_out = fan_in_fan_out
        self.add_relu = add_relu
        # Actual trainable parameters
        if r > 0:
            self.lora_A_text = nn.Linear(in_features, r//4, bias=False)
            self.lora_A_img = nn.Linear(in_features, r//4, bias=False)

            self.lora_A_text_up = nn.Linear(r//4, r, bias=False)
            self.lora_A_img_up = nn.Linear(r//4, r, bias=False)

            self.lora_B = nn.Parameter(torch.zeros(out_features ,r))
            self.scaling = self.lora_alpha / self.r

        self.reset_parameters()

    def reset_parameters(self):
        
        nn.init.kaiming_uniform_(self.lora_A_text.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_img.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_text_up.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_img_up.weight, a=math.sqrt(5))

        nn.init.zeros_(self.lora_B)


    def forward(self, x: torch.Tensor, vis_feat:torch.Tensor=None, multiply_ones=False):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w


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
            
            after_B = F.linear(after_A, self.lora_B)
                
            result = after_B * self.scaling
                
        return result



class TextLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        add_relu: bool = False,
        do_pooling: bool=True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.add_relu = add_relu
        self.do_pooling = do_pooling
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

    def forward(self, x: torch.Tensor, attention_mask:torch.Tensor, multiply_ones=False):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        result = F.linear(x, T(self.weight), bias=self.bias)
        if self.r > 0:
            if self.do_pooling:
                vis_feat = pool(x, attention_mask)
                vis_feat = vis_feat.unsqueeze(1)
            else:
                vis_feat = x
            after_A = F.linear(self.lora_dropout(x), self.lora_A)

            # vis and text share lora_A & lora_B
            after_A_vis = F.linear(self.lora_dropout(vis_feat), self.lora_A)
            bsz, vis_len, vis_dim  = after_A_vis.shape
            if multiply_ones:
                ones_tensor = torch.ones((bsz, vis_len, vis_dim)).to(after_A_vis.device)
                if not self.add_relu:
                    after_A_vis = after_A @ after_A_vis.reshape(bsz, vis_dim, vis_len) @ ones_tensor
                else:
                    after_A_vis = torch.nn.functional.relu(after_A) @ torch.nn.functional.relu(after_A_vis.reshape(bsz, vis_dim, vis_len)) @ ones_tensor
            else:
                if not self.add_relu:
                    after_A_vis = after_A @ after_A_vis.reshape(bsz, vis_dim, vis_len) @ after_A_vis
                else:
                    after_A_vis = torch.nn.functional.relu(after_A) @ torch.nn.functional.relu(after_A_vis.reshape(bsz, vis_dim, vis_len)) @ after_A_vis

            after_B_vis = F.linear(after_A_vis, self.lora_B)
                
            result += after_B_vis * self.scaling
                
        return result


def pool_bart(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    # replace nan with 1. nan occurs when attention_mask is all zero, i.e. no ner postiion in article
    emb = torch.nan_to_num(emb, nan=1.0)
    return emb

def pool(last_hidden_states, attention_mask):
    # works for attention mask with values 0 and -inf
    pooling_mask = torch.clamp(attention_mask.squeeze(), max=0).add(1)
    pooling_mask[pooling_mask!=1] = 0

    # Expand mask to match embeddings shape and apply it
    expanded_mask = pooling_mask.unsqueeze(-1)  # Shape [32, 128, 1]
    masked_embeddings = last_hidden_states * expanded_mask  # Apply mask

    # Perform mean pooling while considering only unmasked positions
    summed = masked_embeddings.sum(dim=1)  # Sum along the sequence length
    mask_sum = expanded_mask.sum(dim=1)  # Count the number of unmasked positions
    pooled_embeddings = summed / mask_sum.clamp(min=1)  # Avoid division by zero
    
    return pooled_embeddings


class TextLinearCross(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        add_relu: bool = False,
        do_pooling: bool=True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.add_relu = add_relu
        self.do_pooling=do_pooling
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


    def forward(self, x: torch.Tensor, enc_hidden_states:torch.Tensor=None, enc_attention_mask=None, multiply_ones=False):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)
        if self.r > 0:
            if self.do_pooling:
                vis_feat = pool_bart(enc_hidden_states, enc_attention_mask)
                vis_feat = vis_feat.unsqueeze(1)
            else:
                vis_feat = enc_hidden_states
            after_A = F.linear(self.lora_dropout(x), self.lora_A)

            # vis and text share lora_A & lora_B
            after_A_vis = F.linear(self.lora_dropout(vis_feat), self.lora_A)
            bsz, vis_len, vis_dim  = after_A_vis.shape
            if multiply_ones:
                ones_tensor = torch.ones((bsz, vis_len, vis_dim)).to(after_A_vis.device)
                if not self.add_relu:
                    after_A_vis = after_A @ after_A_vis.reshape(bsz, vis_dim, vis_len) @ ones_tensor
                else:
                    after_A_vis = torch.nn.functional.relu(after_A) @ torch.nn.functional.relu(after_A_vis.reshape(bsz, vis_dim, vis_len)) @ ones_tensor
            else:
                if not self.add_relu:
                    after_A_vis = after_A @ after_A_vis.reshape(bsz, vis_dim, vis_len) @ after_A_vis
                else:
                    after_A_vis = torch.nn.functional.relu(after_A) @ torch.nn.functional.relu(after_A_vis.reshape(bsz, vis_dim, vis_len)) @ after_A_vis

            after_B_vis = F.linear(after_A_vis, self.lora_B)
                
            result += after_B_vis * self.scaling
                
        return result
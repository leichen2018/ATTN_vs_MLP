'''
    This is adjusted from https://github.com/albietz/transformer-birth/blob/main/ihead_basic_model.py
'''


from dataclasses import dataclass
import itertools
import logging
import random
import math
import numpy as np
import pickle
import time
import torch
import sys

from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Tuple

from copy import deepcopy

@dataclass
class ModelArgs:
    vocab_size: int = -1  # defined later
    dim: int = 256
    max_length: int = 256
    final_ffn: bool = False
    first_ffn: bool = False
    linear_final_ffn: bool = True
    linear_first_ffn: bool = True
    linear_relu_final_ffn: bool = False
    linear_relu_first_ffn: bool = False
    freeze_embeddings: bool = True
    freeze_output: bool = True
    tie_output: bool = False
    use_rope: bool = False
    sqrtd_embeddings: bool = False
    no_sqrtd: bool = False
    sin_cos: bool = False
    relu: bool = True
    parallel: bool = False
    mlp_multiplier: int = 4

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 use_rope: bool = False,
                 no_sqrtd: bool = False,
                 freeze_wk: bool = False,
                 freeze_wv: bool = False,
                 freeze_wo: bool = False,
                 identity_wv: bool = False,
                 diag_decay_wv: bool = False,
                 ):
        super().__init__()
        self.dim = dim
        self.use_rope = use_rope
        self.no_sqrtd = no_sqrtd

        self.wq = nn.Identity()

        self.wk = nn.Linear(dim, dim, bias=False)
        if freeze_wk:
            self.wk.weight.requires_grad_(False)

        self.wv = nn.Linear(dim, dim, bias=False)
        if diag_decay_wv:
            # self.wv.weight.data = torch.diag(1/torch.arange(start=1, end=dim+1))
            # self.wv.weight.data = torch.diag(1/torch.arange(start=1, end=dim+1)**0.5)
            # self.wv.weight.data = torch.diag(1/torch.arange(start=1, end=dim+1)**2)
            self.wv.weight.data = torch.diag(1/torch.arange(start=1, end=dim+1)**0.25)
        if identity_wv:
            self.wv.weight.data = torch.eye(dim)
        if freeze_wv:
            self.wv.weight.requires_grad_(False)

        self.wo = nn.Linear(dim, dim, bias=False)
        if freeze_wo:
            self.wo.weight.requires_grad_(False)

    def wo_low_rank(self, sparsity = 1):
        tmp = deepcopy(self.wo)

        with torch.no_grad():
            U, S, V = torch.svd(tmp.weight.data)
            S[int(sparsity * self.dim):] = 0
            tmp.weight.data = U @ torch.diag(S) @ V.T

            return tmp

    def wv_low_rank(self, sparsity = 1):
        tmp = deepcopy(self.wv)

        with torch.no_grad():
            U, S, V = torch.svd(tmp.weight.data)
            S[int(sparsity * self.dim):] = 0
            tmp.weight.data = U @ torch.diag(S) @ V.T

            return tmp
        
    def wo_wv_low_rank(self, sparsity = 1):
        tmp_wo = deepcopy(self.wo)
        tmp_wv = deepcopy(self.wv)

        with torch.no_grad():
            U, S, V = torch.svd(tmp_wo.weight.data @ tmp_wv.weight.data)
            S[int(sparsity * self.dim):] = 0

            tmp_wo.weight.data = U @ torch.diag(S) @ V.T

            return tmp_wo


    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                freqs_cis: Optional[torch.Tensor] = None,
                wo_low_rank: Optional[bool] = False,
                wo_sparsity: Optional[float] = 1.0,
                wv_low_rank: Optional[bool] = False,
                wv_sparsity: Optional[float] = 1.0,
                wo_wv_low_rank: Optional[bool] = False,
                wo_wv_sparsity: Optional[float] = 1.0,
                ):
        bs, slen, _ = x.shape
        assert mask is not None

        xq = self.wq(x).view(bs, slen, 1, self.dim)
        xk = self.wk(x).view(bs, slen, 1, self.dim)

        if wo_wv_low_rank:
            xv = x.view(bs, slen, 1, self.dim)
        elif not wv_low_rank:
            xv = self.wv(x).view(bs, slen, 1, self.dim)
        else:
            xv = self.wv_low_rank(wv_sparsity)(x).view(bs, slen, 1, self.dim)

        if self.use_rope:
            assert freqs_cis is not None
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # change to (bs, n_heads, slen, head_dim)
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        if self.no_sqrtd:
            scores = torch.matmul(xq, xk.transpose(2, 3))
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.dim)

        scores = scores + mask  # (bs, n_heads, slen, slen)
        pre_scores = scores.float().type_as(x).detach().cpu()
        scores = F.softmax(scores.float(), dim=-1).type_as(x)
        output = torch.matmul(scores, xv)  # (bs, n_heads, slen, head_dim)
        output = output.transpose(1, 2)  # (bs, slen, n_heads, head_dim)

        output = output.reshape(bs, slen, -1)
        
        if wo_wv_low_rank:
            return self.wo_wv_low_rank(wo_wv_sparsity)(output), scores, pre_scores
        elif not wo_low_rank:
            return self.wo(output), scores, pre_scores
        else:
            return self.wo_low_rank(wo_sparsity)(output), scores, pre_scores
        

class FeedForward(nn.Module):
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 relu: bool = True,
                 linear_relu: bool = False,
                 ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        
        if not linear_relu:
            self.w2 = nn.Linear(hidden_dim, dim, bias=False) 
        else:
            self.w2 = nn.Identity()
        
        self.dim = dim
        self.relu = relu
        self.linear_relu = linear_relu

    def w1_low_rank(self, sparsity = 1):
        tmp = deepcopy(self.w1)

        with torch.no_grad():
            U, S, V = torch.svd(tmp.weight.data)
            S[int(sparsity * self.dim):] = 0
            tmp.weight.data = U @ torch.diag(S) @ V.T

            return tmp
        
    def w2_low_rank(self, sparsity = 1):
        if self.linear_relu:
            return self.w2

        tmp = deepcopy(self.w2)

        with torch.no_grad():
            U, S, V = torch.svd(tmp.weight.data)
            S[int(sparsity * self.dim):] = 0
            tmp.weight.data = U @ torch.diag(S) @ V.T

            return tmp

    def forward(self, x,
                w1_low_rank: Optional[bool] = False,
                w1_sparsity: Optional[float] = 1.0,
                w2_low_rank: Optional[bool] = False,
                w2_sparsity: Optional[float] = 1.0,
                relu: Optional[bool] = None, # take care that the default is none to set the relu_flag as the model's initialization
                ):
        if not w1_low_rank:
            h = self.w1(x)
        else:
            h = self.w1_low_rank(sparsity=w1_sparsity)(x)
        
        if relu is None:
            relu_flag = self.relu 
        else:
            relu_flag = relu

        if relu_flag:
            h = F.relu(h.float()).type_as(x)
        else:
            h = h.float().type_as(x)

        if not w2_low_rank:
            return self.w2(h)
        else:
            return self.w2_low_rank(sparsity=w2_sparsity)(h)


class TransformerBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 use_rope: bool = False,
                 no_sqrtd: bool = False,
                 no_ffn: bool = False,
                 linear_ffn: bool = False,
                 parallel: bool = False,
                 freeze_wk: bool = False,
                 freeze_wv: bool = False,
                 freeze_wo: bool = False,
                 freeze_ffn: bool = False,
                 identity_wv: bool = False,
                 diag_decay_wv: bool = False,
                 relu: bool = True,
                 linear_relu_ffn: bool = False,
                 mlp_multiplier: int = 4,
                ):
        super().__init__()
        self.attention = Attention(
                dim=dim,
                use_rope=use_rope,
                no_sqrtd=no_sqrtd,
                freeze_wk=freeze_wk,
                freeze_wv=freeze_wv,
                freeze_wo=freeze_wo, 
                identity_wv=identity_wv,
                diag_decay_wv=diag_decay_wv,
                )
        self.no_ffn = no_ffn
        self.linear_ffn = linear_ffn
        self.parallel = parallel
        self.dim = dim
        self.hid_dim = mlp_multiplier * dim if not linear_relu_ffn else dim

        if not no_ffn:
            if linear_ffn:
                self.ff = nn.Linear(dim, dim, bias=False)
            else:
                self.ff = FeedForward(dim=dim, hidden_dim=self.hid_dim,
                                      relu = relu,
                                      linear_relu = linear_relu_ffn) # added by lei
            if freeze_ffn:
                for p in self.ff.parameters():
                    p.requires_grad_(False)

    def linear_low_rank(self, sparsity):
        tmp = deepcopy(self.ff)

        with torch.no_grad():
            U, S, V = torch.svd(tmp.weight.data)
            S[int(sparsity * self.dim):] = 0
            tmp.weight.data = U @ torch.diag(S) @ V.T

            return tmp

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                freqs_cis: Optional[torch.Tensor] = None,
                return_scores: bool = False,
                no_ffn: bool = False,
                wo_low_rank: Optional[bool] = False,
                wo_sparsity: Optional[float] = 1.0,
                wv_low_rank: Optional[bool] = False,
                wv_sparsity: Optional[float] = 1.0,
                wo_wv_low_rank: Optional[bool] = False,
                wo_wv_sparsity: Optional[float] = 1.0,
                w1_low_rank: Optional[bool] = False,
                w1_sparsity: Optional[float] = 1.0,
                w2_low_rank: Optional[bool] = False,
                w2_sparsity: Optional[float] = 1.0,
                wlin_low_rank: Optional[bool] = False,
                wlin_sparsity: Optional[float] = 1.0,
                no_residual: Optional[bool] = False,
                relu: Optional[bool] = None, # take care that the default is none to set the relu_flag as the model's initialization
                ):
        no_ffn = no_ffn or self.no_ffn

        h, scores, _ = self.attention(x, mask, freqs_cis=freqs_cis, wo_low_rank=wo_low_rank, wo_sparsity=wo_sparsity, wv_low_rank=wv_low_rank, wv_sparsity=wv_sparsity, wo_wv_low_rank=wo_wv_low_rank, wo_wv_sparsity=wo_wv_sparsity)

        if return_scores:
            return scores
        if no_ffn:
            if no_residual:
                return h
            return x + h
        else:
            if self.parallel:
                if self.linear_ffn:
                    return x + h + self.ff(x)
                else:
                    return x + h + self.ff(x, 
                                        w1_low_rank=w1_low_rank,
                                        w1_sparsity=w1_sparsity,
                                        w2_low_rank=w2_low_rank,
                                        w2_sparsity=w2_sparsity,
                                        relu=relu,
                                        )
            else:
                h = x + h

                if self.linear_ffn:
                    if not wlin_low_rank:
                        return h + self.ff(h)
                    else:
                        return h + self.linear_low_rank(sparsity=wlin_sparsity)(h)
                else:
                    return h + self.ff(h,
                                    w1_low_rank=w1_low_rank,
                                        w1_sparsity=w1_sparsity,
                                        w2_low_rank=w2_low_rank,
                                        w2_sparsity=w2_sparsity,
                                        relu=relu,
                                    )
                                    


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.tie_output = args.tie_output
        self.dim = args.dim
        self.use_rope = args.use_rope
        self.sin_cos = args.sin_cos

        # embeddings
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        if args.sqrtd_embeddings:
            self.tok_embeddings.weight.data.normal_(std=1./math.sqrt(args.dim))
        if args.freeze_embeddings:
            self.tok_embeddings.weight.requires_grad_(False)

        if self.sin_cos:
            # sin/cos position embeddings
            position = torch.arange(args.max_length).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, args.dim, 2) * (-math.log(10000.0) / args.dim))
            pe = torch.zeros(args.max_length, args.dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # random absolute positional embeddings
            pe = torch.randn(args.max_length, args.dim)
        if args.sqrtd_embeddings:
            pe *= 1. / math.sqrt(args.dim)

        self.register_buffer('pe', pe)

        freqs_cis = precompute_freqs_cis(
            self.dim // 1, args.max_length
        )
        self.register_buffer('freqs_cis', freqs_cis)

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=args.dim,
                use_rope=args.use_rope,
                no_sqrtd=args.no_sqrtd,
                no_ffn=not args.first_ffn,
                linear_ffn=args.linear_first_ffn,
                freeze_wk=False,
                freeze_wv=False, # it was frozen by default
                freeze_wo=False, # it was frozen by default
                identity_wv=False, # add by lei, need to set freeze_wv=True
                diag_decay_wv=False, # add by lei, need to set freeze_wv=True and can be overlapped by identity_wv=True
                relu=args.relu, # added by lei, for the nonlinearity in FFN
                freeze_ffn=False,
                linear_relu_ffn=args.linear_relu_first_ffn,
                mlp_multiplier=args.mlp_multiplier, 
                ),
            TransformerBlock( 
                dim=args.dim,
                use_rope=False,  # args.use_rope,
                no_sqrtd=args.no_sqrtd,
                no_ffn=not args.final_ffn,
                linear_ffn=args.linear_final_ffn,
                freeze_wk=False,
                freeze_wv=False, # it was frozen by default
                freeze_wo=False,
                freeze_ffn=False,
                linear_relu_ffn=args.linear_relu_final_ffn,
                parallel=args.parallel,
                mlp_multiplier = args.mlp_multiplier, 
                )
            ])

        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        if args.freeze_output:
            self.output.weight.requires_grad_(False)
        if args.tie_output:
            if args.freeze_output:
                self.output.weight.data = self.tok_embeddings.weight.data / math.sqrt(args.dim)
            else:
                self.output.weight = self.tok_embeddings.weight / math.sqrt(args.dim)

    def forward(self, tokens: torch.Tensor, return_layer: Optional[int] = None, before_ffn: bool = False,
                wo_low_rank: Optional[bool] = False,
                wo_sparsity: Optional[float] = 1.0,
                wv_low_rank: Optional[bool] = False,
                wv_sparsity: Optional[float] = 1.0,
                wo_wv_low_rank: Optional[bool] = False,
                wo_wv_sparsity: Optional[float] = 1.0,
                w1_low_rank: Optional[bool] = False,
                w1_sparsity: Optional[float] = 1.0,
                w2_low_rank: Optional[bool] = False,
                w2_sparsity: Optional[float] = 1.0,
                wlin_low_rank: Optional[bool] = False,
                wlin_sparsity: Optional[float] = 1.0,
                mlp2_w1_low_rank: Optional[bool] = False,
                mlp2_w1_sparsity: Optional[float] = 1.0,
                mlp2_w2_low_rank: Optional[bool] = False,
                mlp2_w2_sparsity: Optional[float] = 1.0,
                wlin2_low_rank: Optional[bool] = False,
                wlin2_sparsity: Optional[float] = 1.0,
                no_residual: Optional[bool] = False,
                relu: Optional[bool] = None, # take care that the default is none to set the relu_flag as the model's initialization
                ):
        B, N = tokens.shape

        # embedding layer
        h = self.tok_embeddings(tokens)
        if not self.use_rope:
            h = h + self.pe.unsqueeze(0)

        if return_layer == 0:
            return h

        # causal mask
        mask = torch.full((1, 1, N, N), float('-inf'), device=tokens.device)
        mask = torch.triu(mask, diagonal=1).type_as(h)

        # transformer blocks
        for i, layer in enumerate(self.layers):
            if return_layer == i + 1:
                return layer(h, mask, freqs_cis=self.freqs_cis, no_ffn=before_ffn)
            if i >= 1:
            # if i == 0:
                h = layer(h, mask, freqs_cis=self.freqs_cis, no_residual=no_residual,
                          w1_low_rank=mlp2_w1_low_rank,
                          w1_sparsity=mlp2_w1_sparsity,
                          w2_low_rank=mlp2_w2_low_rank,
                          w2_sparsity=mlp2_w2_sparsity,
                          wlin_low_rank=wlin2_low_rank,
                          wlin_sparsity=wlin2_sparsity,
                          relu=relu,
                          )
            else:
                h = layer(h, mask, freqs_cis=self.freqs_cis, wo_low_rank=wo_low_rank, wo_sparsity=wo_sparsity, wv_low_rank=wv_low_rank, wv_sparsity=wv_sparsity, wo_wv_low_rank=wo_wv_low_rank, wo_wv_sparsity=wo_wv_sparsity,
                          w1_low_rank=w1_low_rank,
                            w1_sparsity=w1_sparsity,
                            w2_low_rank=w2_low_rank,
                            w2_sparsity=w2_sparsity,
                            wlin_low_rank=wlin_low_rank,
                            wlin_sparsity=wlin_sparsity,
                            relu=relu,
                          )

        # output layer
        output = self.output(h)
        if self.tie_output:
            output /= math.sqrt(self.dim)
        return output.float()

    def forward_ff_only(self, tokens: torch.Tensor):
        B, N = tokens.shape

        # embedding layer
        h = self.tok_embeddings(tokens)
        if not self.use_rope:
            h = h + self.pe.unsqueeze(0)

        # transformer blocks
        for i, layer in enumerate(self.layers):
            h = h + layer.ff(h)

        # output layer
        output = self.output(h)
        if self.tie_output:
            output /= math.sqrt(self.dim)
        return output.float()

    def get_layer_scores(self, tokens: torch.Tensor, n: int = 0):
        assert n < len(self.layers)
        B, N = tokens.shape

        # embedding layer
        h = self.tok_embeddings(tokens)
        h = h + self.pe.unsqueeze(0)

        # causal mask
        mask = torch.full((1, 1, N, N), float('-inf'), device=tokens.device)
        mask = torch.triu(mask, diagonal=1).type_as(h)

        # transformer blocks
        for i, layer in enumerate(self.layers):
            if i == n:
                return layer(h, mask, freqs_cis=self.freqs_cis, return_scores=True)
            else:
                h = layer(h, mask, freqs_cis=self.freqs_cis)

    def get_top_preds(self, tokens: torch.Tensor, n: int = 4):
        squeeze = False
        if len(tokens.shape) == 1:
            squeeze = True
            tokens = tokens.unsqueeze(0)
        with torch.no_grad():
            preds = self(tokens).detach()
        vals, idxs = preds.sort(-1, descending=True)
        vals = vals[:,:,:n]
        idxs = idxs[:,:,:n]
        if squeeze:
            return vals.squeeze(0), idxs.squeeze(0)
        return vals, idxs


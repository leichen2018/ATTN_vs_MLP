'''
    This is adjusted from https://github.com/albietz/transformer-birth/blob/main/ihead_basic_main.py
'''

from collections import defaultdict
from dataclasses import dataclass
import itertools
import logging
import random
import json
import math
import numpy as np
import time
import torch
import sys

from omegaconf import OmegaConf
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Tuple
from pathlib import Path

from ihead_data import DataArgs, Dataset, iterate_batches
from ihead_basic_model import ModelArgs, Transformer

logging.getLogger().setLevel(logging.INFO)


@dataclass
class OptimArgs:
    learning_rate: float = 0.2  # for SGD
    weight_decay: float = 1e-4  # for SGD
    momentum: float = 0.9  # for SGD
    batch_size: int = 512
    use_sgd: bool = True  # otherwise use AdamW


@dataclass
class TrainerArgs:
    optim_args: OptimArgs
    data_args: DataArgs
    model_args: ModelArgs
    max_iters: Optional[int] = None
    eval_delta: int = 5
    log_norms: bool = False
    log_probes: bool = False
    freeze_until: str = ''
    loss_head_only: bool = True
    bigram_outs_train: bool = False
    bigram_outs_test: bool = False
    num_data_workers: int = 60
    seed: int = 42
    save_dir: Optional[str] = None
    root_dir: str = ''
    save_model: Optional[bool] = False
    save_model_dir: str = ''
    load_model: Optional[bool] = False
    load_model_dir: str = ''
    no_residual_train: bool = False
    no_residual: bool = True
    noise_prob: float = 0.3
    noise_token_id: int = 2
    freeze_L0: bool = False
    only_train_k2_ffn2: bool = False
    scale_output: float = 1.0

if __name__ == '__main__':
    args = TrainerArgs(
           optim_args=OptimArgs(),
           data_args=DataArgs(),
           model_args=ModelArgs()
        )
    cfg = OmegaConf.merge(OmegaConf.structured(args), OmegaConf.from_cli())

    print("The current noise probability is {}!".format(cfg.noise_prob))
    ds = Dataset(cfg.data_args, train_test=None, bigram_outs=cfg.bigram_outs_train, noise_prob=cfg.noise_prob, noise_token_id=cfg.noise_token_id)
    ds_test = Dataset(cfg.data_args, train_test=None, bigram_outs=cfg.bigram_outs_test, noise_prob=0.)
    ds_test.idxs = ds.idxs
    cfg.model_args.vocab_size = ds.num_tokens

    if cfg.save_dir is not None:
        outdir = Path(cfg.root_dir) / Path(cfg.save_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        # save params
        with open(outdir / 'params.json', 'w') as f:
                json.dump(dict(cfg), f, sort_keys=True, indent=4)
        outfile = open(outdir / 'res.jsonl', 'w')

    print(cfg.model_args)

    firstffn_w1w2_flag = cfg.model_args.first_ffn and (not cfg.model_args.linear_first_ffn) # True: if two-layer MLP in first transformer layer
    firstffn_wlin_flag = cfg.model_args.first_ffn and cfg.model_args.linear_first_ffn # True: if a linear layer in first transformer layer

    secondffn_w1w2_flag = cfg.model_args.final_ffn and (not cfg.model_args.linear_final_ffn)
    secondffn_wlin_flag = cfg.model_args.final_ffn and cfg.model_args.linear_final_ffn

    model = Transformer(cfg.model_args)
    if cfg.load_model:
        model.load_state_dict(torch.load('saved_models/' + cfg.load_model_dir))
        print("Successfully LOAD from {}".format('saved_models/' + cfg.load_model_dir))

    if cfg.freeze_L0:
        for name, p in model.layers[0].named_parameters():
            p.requires_grad = False

    if cfg.only_train_k2_ffn2:
        for name, p in model.named_parameters():
            if name in ['layers.1.attention.wk.weight', 'layers.1.ff.w1.weight', 'layers.1.ff.w2.weight']:
                p.requires_grad = True
            else:
                p.requires_grad = False

    if cfg.scale_output != 1:
        if secondffn_w1w2_flag:
            model.output.weight.data = model.output.weight.data * cfg.scale_output
        

    model.cuda()

    print(model)
    # attn probes
    attn_features = None
    attn_features2 = None
    attn_input_features = None
    attn_scores = None
    attn_scores2 = None
    def attn0_hook(_, inp, outp):
        global attn_features, attn_input_features, attn_scores
        attn_input_features = inp[0].detach()
        attn_features = outp[0].detach()
        attn_scores = outp[1].detach()
    model.layers[0].attention.register_forward_hook(attn0_hook)
    def attn1_hook(_, inp, outp):
        global attn_scores2, attn_features2
        attn_features2 = outp[0].detach()
        attn_scores2 = outp[1].detach()
    model.layers[1].attention.register_forward_hook(attn1_hook)

    # memory probes
    range_toks = torch.from_numpy(np.arange(ds.n_train_toks)).cuda()
    def test_wo1():
        toks = model.tok_embeddings(range_toks)
        toks = model.layers[1].attention.wv(toks)
        toks = model.layers[1].attention.wo(toks)
        toks = model.output(toks)
        return (toks.argmax(-1) == range_toks).float().mean().item()

    full_range_toks = torch.from_numpy(np.arange(ds.num_tokens)).cuda()
    conds = torch.from_numpy(np.array(ds.cond)).cuda()
    used_idxs = np.arange(ds.num_tokens)
    if cfg.data_args.fixed_special_toks:
        used_idxs = np.setdiff1d(used_idxs, ds.idxs)
    def test_ff1():
        toks = model.tok_embeddings(full_range_toks[used_idxs])
        toks = model.layers[1].ff(toks)
        toks = model.output(toks)
        return F.kl_div(F.log_softmax(toks, dim=1), conds[used_idxs], reduction='batchmean').item()

    range_pos_toks = torch.from_numpy(np.arange(cfg.model_args.max_length)).cuda()
    def test_wk0(cutoff=None):
        pe = model.pe[:cutoff,:]
        k = model.layers[0].attention.wk(pe[:-1])
        q = model.layers[0].attention.wq(pe[1:])
        return ((q @ k.t()).argmax(-1) == range_pos_toks[:pe.shape[0]-1]).float().mean().item()

    wk1_range_toks = full_range_toks.clone()
    if cfg.data_args.fixed_special_toks:
        wk1_range_toks = wk1_range_toks[ds.idxs]
    def test_wk1():
        toksk = model.tok_embeddings(wk1_range_toks)
        toksk = model.layers[0].attention.wv(toksk)
        toksk = model.layers[0].attention.wo(toksk)
        toksk = model.layers[1].attention.wk(toksk)

        toksq = model.tok_embeddings(wk1_range_toks)
        toksq = model.layers[1].attention.wq(toksq)
        return ((toksq @ toksk.t()).argmax(-1) == range_toks[:wk1_range_toks.shape[0]]).float().mean().item()
    
    def softrank(w):
        return torch.linalg.matrix_norm(w, ord='fro').detach().cpu().item()/torch.linalg.matrix_norm(w, ord=2).detach().cpu().item()

    def compute_wo_softrank():
        with torch.no_grad():
            wo1 = model.layers[0].attention.wo.weight.data
            wo2 = model.layers[1].attention.wo.weight.data

        return softrank(wo1), softrank(wo2)

    def compare_pred_diversity(pred, label, shared_noise = 2, eps=0.1):
        label_noise_index = (label==shared_noise)
        label_non_noise_index = (label!=shared_noise)
        pred_noise_index = (pred==shared_noise)
        pred_non_noise_index = (pred!=shared_noise)

        pred_noise_label_noise = (torch.sum(torch.logical_and(pred_noise_index, label_noise_index))/(eps + torch.numel(pred))).cpu().item()
        pred_noise_label_non_noise = (torch.sum(torch.logical_and(pred_noise_index, label_non_noise_index))/(eps + torch.numel(pred))).cpu().item()
        pred_non_noise_label_noise = (torch.sum(torch.logical_and(pred_non_noise_index, label_noise_index))/(eps + torch.numel(pred))).cpu().item()
        pred_non_noise_label_non_noise = (torch.sum(torch.logical_and(pred_non_noise_index, label_non_noise_index))/(eps + torch.numel(pred))).cpu().item()

        return pred_noise_label_noise, pred_noise_label_non_noise, pred_non_noise_label_noise, pred_non_noise_label_non_noise


    # initial param freezing
    freeze_until = defaultdict(list)
    to_freeze = []
    if cfg.freeze_until:
        for kv in cfg.freeze_until.split(','):
            k, v = kv.split(':')
            k = int(k)
            to_freeze.append(v)
            freeze_until[k].append(v)

        for name, p in model.named_parameters():
            if name in to_freeze:
                p.requires_grad_(False)

    # optim
    if cfg.optim_args.use_sgd:
        optimizer = torch.optim.SGD(model.parameters(),
                lr=cfg.optim_args.learning_rate,
                weight_decay=cfg.optim_args.weight_decay,
                momentum=cfg.optim_args.momentum)
    else:
        optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.optim_args.learning_rate,
                weight_decay=cfg.optim_args.weight_decay,
                betas=(0.9, 0.95),
                eps=1e-8)

    # a test batch for experimentation
    x_exp, out_exp = ds.gen_batch(np.random.default_rng(0), 128)
    x_exp = x_exp[:,:ds.seq_length]

    # OOD test data
    x_test, out_test = ds_test.gen_batch(np.random.default_rng(0), 512)
    x_t = torch.from_numpy(x_test[:,:ds.seq_length]).cuda()
    y_t = torch.from_numpy(x_test[:,1:ds.seq_length + 1]).cuda()
    outs_t = torch.from_numpy(out_test[:,:ds.seq_length]).cuda()

    import wandb
    wandb.init(project="icl-noise",)

    def mlp_trigger_margin_from_noise(model, trigger, noise_channel = 2):
        trigger_emb = model.tok_embeddings(torch.LongTensor([trigger]).cuda())
        trigger_emb = model.layers[0].ff(trigger_emb)
        output_logits = model.output(trigger_emb).squeeze()

        no_noise_mask = torch.ones(size=(65,), dtype=torch.bool).cuda()
        no_noise_mask[noise_channel] = False
        logits_no_noise_max = output_logits[no_noise_mask].max()

        return (output_logits[noise_channel] - logits_no_noise_max).cpu().detach().item()
    
    def mlp2_trigger_margin_from_noise(model, trigger, noise_channel = 2):
        trigger_emb = model.tok_embeddings(torch.LongTensor([trigger]).cuda())
        trigger_emb = model.layers[1].ff(trigger_emb)
        output_logits = model.output(trigger_emb).squeeze()

        no_noise_mask = torch.ones(size=(65,), dtype=torch.bool).cuda()
        no_noise_mask[noise_channel] = False
        logits_no_noise_max = output_logits[no_noise_mask].max()

        return (output_logits[noise_channel] - logits_no_noise_max).cpu().detach().item()
    
    def mlp1_mlp2_trigger_margin_from_noise(model, trigger, noise_channel = 2):
        trigger_emb = model.tok_embeddings(torch.LongTensor([trigger]).cuda())
        trigger_emb = model.layers[0].ff(trigger_emb)
        trigger_emb = model.layers[1].ff(trigger_emb)
        output_logits = model.output(trigger_emb).squeeze()

        no_noise_mask = torch.ones(size=(65,), dtype=torch.bool).cuda()
        no_noise_mask[noise_channel] = False
        logits_no_noise_max = output_logits[no_noise_mask].max()

        return (output_logits[noise_channel] - logits_no_noise_max).cpu().detach().item()

    def average_prob_noise(output, noise_channel=2):
        prob = torch.softmax(output, dim=-1)

        return torch.mean(prob[...,noise_channel].mean()).detach().cpu().item()

    def average_prob_correct(output, label):
        prob = torch.softmax(output, dim=-1)

        res = prob[torch.arange(output.size()[0]).to('cuda'), label]


        return res.mean().detach().cpu().item()
    
    t = time.time()
    t0 = t
    res = []
    for i, (x, y, outs) in enumerate(iterate_batches(ds, batch_size=cfg.optim_args.batch_size,
                                     num_workers=cfg.num_data_workers, seed=cfg.seed)):
        dt_data = time.time() - t

        t0_track = time.time()

        if cfg.max_iters is not None and i >= cfg.max_iters:
            if cfg.save_model:
                torch.save(model.state_dict(), 'saved_models/' + cfg.save_model_dir)
                print("Successfully SAVED at {}".format('saved_models/' + cfg.save_model_dir))

            if cfg.save_dir is not None:
                outfile.close()
            sys.exit(0)

        if cfg.model_args.first_ffn:
            for idxs in ds_test.idxs:
                mlp_margin = mlp_trigger_margin_from_noise(model, idxs, cfg.noise_token_id)


                wandb.log({
                    "mlp_trigger_margin/" + str(idxs) : mlp_margin,
                }, step=i
                )

        if cfg.model_args.final_ffn:
            for idxs in ds_test.idxs:
                mlp_margin = mlp2_trigger_margin_from_noise(model, idxs, cfg.noise_token_id)


                wandb.log({
                    "mlp2_trigger_margin/" + str(idxs) : mlp_margin,
                }, step=i
                )

        if cfg.model_args.first_ffn and cfg.model_args.final_ffn:
            for idxs in ds_test.idxs:
                mlp_margin = mlp1_mlp2_trigger_margin_from_noise(model, idxs, cfg.noise_token_id)


                wandb.log({
                    "mlp1_mlp2_trigger_margin/" + str(idxs) : mlp_margin,
                }, step=i
                )

        t1_track = time.time()

        x = torch.from_numpy(x).cuda()
        y = torch.from_numpy(y).cuda()
        outs = torch.from_numpy(outs).cuda()

        if i in freeze_until:  # unfreeze params
            for name, p in model.named_parameters():
                if name in freeze_until[i]:
                    p.requires_grad_(True)

        pred = model(x, no_residual = cfg.no_residual_train)

        
        outs_current = (outs>=1)
        outs_next = torch.zeros_like(outs_current)
        outs_next[..., 1:] = outs_current[..., :-1]

        attn_scores1_offdiaongal_mean = torch.diagonal(attn_scores, offset=-1, dim1=-2, dim2=-1).mean().cpu().detach().item()
        attn_scores1_offdiaongal_mean_first100 = torch.diagonal(attn_scores, offset=-1, dim1=-2, dim2=-1)[..., :100].mean().cpu().detach().item()

        attn_scores2_to_correct = []
        attn_scores2_to_noise = []
        for i1 in range(attn_scores2.shape[0]):
            attn_scores2_to_correct.append(attn_scores2[i1, 0, outs_current[i1]][:, outs_next[i1]].sum(dim=-1).mean().cpu().detach().item())
            noise_indices = (x[i1] == cfg.noise_token_id)
            attn_scores2_to_noise.append(attn_scores2[i1, 0, outs_current[i1]][:, torch.logical_and(outs_next[i1], noise_indices)].sum(dim=-1).mean().cpu().detach().item())

        attn_scores2_to_correct = np.mean(attn_scores2_to_correct)
        attn_scores2_to_noise = np.mean(attn_scores2_to_noise)
        attn_scores2_to_correct_real = attn_scores2_to_correct - attn_scores2_to_noise

        wandb.log({
                    "attn_scores/L1_offdiaongal_mean": attn_scores1_offdiaongal_mean,
                    "attn_scores/L1_offdiaongal_mean_first100": attn_scores1_offdiaongal_mean_first100,
                    "attn_scores/L2_to_correct_mean": attn_scores2_to_correct,
                    "attn_scores/L2_to_noise_mean": attn_scores2_to_noise,
                    "attn_scores/L2_to_real_correct_mean": attn_scores2_to_correct_real,
        }, step=i
        )
        

        if cfg.loss_head_only:
            loss = F.cross_entropy(pred[outs >= 2], y[outs >= 2])
        else:
            loss = F.cross_entropy(pred.flatten(0, 1), y.flatten(0, 1))

        train_prob_noise_full = average_prob_noise(pred[outs >= 2], cfg.noise_token_id)
        wandb.log({"average_prob_noise/train/full": train_prob_noise_full}, step=i)
        
        dt = time.time() - t
        t = time.time()

        t2_track = time.time()

        if i<= 250:
            no_residual = False
        else:
            no_residual = cfg.no_residual

        if i % cfg.eval_delta == 0:
            if cfg.data_args.k > 0:
                acc_tot = (pred.argmax(-1)[outs >= 1] == y[outs >= 1]).float().mean().item()
                sl = 10
                acc_start = (pred[:,:sl].argmax(-1)[outs[:,:sl] >= 1] == y[:,:sl][outs[:,:sl] >= 1]).float().mean().item()
                el = 500
                acc_end = (pred[:,-el:].argmax(-1)[outs[:,-el:] >= 2] == y[:,-el:][outs[:,-el:] >= 2]).float().mean().item()
                loss_bigram = F.cross_entropy(pred[outs == 0,:], y[outs == 0]).item()
                loss_head = F.cross_entropy(pred[outs >= 2,:], y[outs >= 2]).item()

                # first layer attn scores probe
                i1, i2 = torch.where(outs[:,:-1] >= 1)
                i1_start, i2_start = torch.where(outs[:,:-1] == 1)
                amax = attn_scores[:,0,:,:].argmax(-1)
                score_acc = (amax[i1, i2 + 1] == i2).float().mean().item()
                score_start_acc = (amax[i1_start, i2_start + 1] == i2_start).float().mean().item()

                # second layer attn scores probe (check that attended token's prev token has correct condition)
                i1, i2 = torch.where(outs >= 2)
                amax2 = attn_scores2.squeeze(1)[i1,i2,:].argmax(-1)
                score2_next_acc = (x[i1, amax2] == y[i1, i2]).float().mean().item()
                pred_attended_acc = (x[i1, amax2] == pred[i1,i2].argmax(-1)).float().mean().item()

                bad = (amax2 == 0).float().sum()
                tot = amax2.shape[0]
                i1 = i1[amax2 >= 1]
                i2 = i2[amax2 >= 1]
                amax2 = amax2[amax2 >= 1]
                score2_acc = (x[i1, amax2 - 1] == x[i1, i2]).float().sum().item() / tot

                wo1_softrank, wo2_softrank = compute_wo_softrank()

                # first layer attn score probe conditioned on locations attended by second layer
                score_cond_acc = (amax[i1, amax2] == amax2 - 1).float().mean().item()

                # second layer attn score probe conditioned on repeated tokens
                i1, i2 = torch.where((outs >= 2) & (x == y))
                amax1 = attn_scores.squeeze(1)[i1,i2,:].argmax(-1)
                score1_repeat_val_acc = (x[i1, amax1] == y[i1, i2]).float().mean().item()
                amax2 = attn_scores2.squeeze(1)[i1,i2,:].argmax(-1)
                score2_repeat_val_acc = (x[i1, amax2] == y[i1, i2]).float().mean().item()
                
                if True:  # cfg.log_probes:
                    wo1_acc = test_wo1()
                    if cfg.model_args.final_ffn:
                        ff1_loss = test_ff1()
                    else:
                        ff1_loss = -1
                    wk0_acc = test_wk0()
                    wk0_64_acc = test_wk0(cutoff=64)
                    wk1_acc = test_wk1()

                repeat_frac = (x[outs >= 1] == y[outs >= 1]).float().mean().item()

                # OOD test (NOTE: do this after the probes sinces it messes hooks!)
                with torch.no_grad():
                    pred_t = model(x_t, no_residual=no_residual)
                    pred_t_residual = model(x_t, no_residual=False)

                    pred_no_residual = model(x, no_residual=True)
                    train_loss_no_residual = F.cross_entropy(pred_no_residual[outs >= 2], y[outs >= 2])
                    acc_tot_no_residual = (pred_no_residual.argmax(-1)[outs >= 1] == y[outs >= 1]).float().mean().item()
                    train_PredN_LabelN_no_residual, train_PredN_LabelNN_no_residual, train_PredNN_LabelN_no_residual, train_PredNN_LabelNN_no_residual = compare_pred_diversity(pred_no_residual.argmax(-1)[outs >= 1], y[outs >= 1], shared_noise=cfg.noise_token_id)


                    if firstffn_w1w2_flag:
                        pred_t_ffn1_w1_01 = model(x_t, w1_low_rank=True, w1_sparsity=0.1, no_residual=no_residual)
                        pred_t_ffn1_w1_02 = model(x_t, w1_low_rank=True, w1_sparsity=0.2, no_residual=no_residual)
                        pred_t_ffn1_w1_0 = model(x_t, w1_low_rank=True, w1_sparsity=0, no_residual=no_residual)

                        pred_t_ffn1_w2_01 = model(x_t, w2_low_rank=True, w2_sparsity=0.1, no_residual=no_residual)
                        pred_t_ffn1_w2_02 = model(x_t, w2_low_rank=True, w2_sparsity=0.2, no_residual=no_residual)
                        pred_t_ffn1_w2_0 = model(x_t, w2_low_rank=True, w2_sparsity=0, no_residual=no_residual)

                        pred_t_ffn1_norelu = model(x_t, relu=False)

                    if secondffn_w1w2_flag:
                        pred_t_ffn2_w1_01 = model(x_t, mlp2_w1_low_rank=True, mlp2_w1_sparsity=0.1, no_residual=no_residual)
                        pred_t_ffn2_w1_02 = model(x_t, mlp2_w1_low_rank=True, mlp2_w1_sparsity=0.2, no_residual=no_residual)
                        pred_t_ffn2_w1_0 = model(x_t, mlp2_w1_low_rank=True, mlp2_w1_sparsity=0, no_residual=no_residual)

                        pred_t_ffn2_w2_01 = model(x_t, mlp2_w2_low_rank=True, mlp2_w2_sparsity=0.1, no_residual=no_residual)
                        pred_t_ffn2_w2_02 = model(x_t, mlp2_w2_low_rank=True, mlp2_w2_sparsity=0.2, no_residual=no_residual)
                        pred_t_ffn2_w2_0 = model(x_t, mlp2_w2_low_rank=True, mlp2_w2_sparsity=0, no_residual=no_residual)

                        test_prob_noise_ffn2_w1_02 = average_prob_noise(pred_t_ffn2_w1_02[:,-el:][outs_t[:,-el:] >= 2], cfg.noise_token_id)
                        test_prob_noise_ffn2_w1_0 = average_prob_noise(pred_t_ffn2_w1_0[:,-el:][outs_t[:,-el:] >= 2], cfg.noise_token_id)

                        test_prob_correct_ffn2_w1_02 = average_prob_correct(pred_t_ffn2_w1_02[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])
                        test_prob_correct_ffn2_w1_0 = average_prob_correct(pred_t_ffn2_w1_0[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])


                        wandb.log({
                            "average_prob_noise/test/ffn2_w1_02": test_prob_noise_ffn2_w1_02,
                            "average_prob_noise/test/ffn2_w1_0": test_prob_noise_ffn2_w1_0,

                            "average_prob_correct/test/ffn2_w1_02": test_prob_correct_ffn2_w1_02,
                            "average_prob_correct/test/ffn2_w1_0": test_prob_correct_ffn2_w1_0,
                            }, step=i)
                        
                    if secondffn_wlin_flag:
                        pred_t_ffn2_wlin_01 = model(x_t, wlin2_low_rank=True, wlin2_sparsity=0.1, no_residual=no_residual)
                        pred_t_ffn2_wlin_02 = model(x_t, wlin2_low_rank=True, wlin2_sparsity=0.2, no_residual=no_residual)
                        pred_t_ffn2_wlin_0 = model(x_t, wlin2_low_rank=True, wlin2_sparsity=0, no_residual=no_residual)

                        test_prob_noise_ffn2_wlin_02 = average_prob_noise(pred_t_ffn2_wlin_02[:,-el:][outs_t[:,-el:] >= 2], cfg.noise_token_id)
                        test_prob_noise_ffn2_wlin_0 = average_prob_noise(pred_t_ffn2_wlin_0[:,-el:][outs_t[:,-el:] >= 2], cfg.noise_token_id)

                        test_prob_correct_ffn2_wlin_02 = average_prob_correct(pred_t_ffn2_wlin_02[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])
                        test_prob_correct_ffn2_wlin_0 = average_prob_correct(pred_t_ffn2_wlin_0[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])


                        wandb.log({
                            "average_prob_noise/test/ffn2_wlin_02": test_prob_noise_ffn2_wlin_02,
                            "average_prob_noise/test/ffn2_wlin_0": test_prob_noise_ffn2_wlin_0,

                            "average_prob_correct/test/ffn2_wlin_02": test_prob_correct_ffn2_wlin_02,
                            "average_prob_correct/test/ffn2_wlin_0": test_prob_correct_ffn2_wlin_0,
                            }, step=i)

                    test_prob_noise_full = average_prob_noise(pred_t[:,-el:][outs_t[:,-el:] >= 2], cfg.noise_token_id)
                    test_prob_correct_full = average_prob_correct(pred_t[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])
                    wandb.log({
                        "average_prob_noise/test/full": test_prob_noise_full,
                        "average_prob_correct/test/full": test_prob_correct_full,
                        }, step=i)

                    test_loss = F.cross_entropy(pred_t[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])
                    test_loss_residual = F.cross_entropy(pred_t_residual[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])

                    if firstffn_w1w2_flag:
                        test_loss_ffn1_w1_02 = F.cross_entropy(pred_t_ffn1_w1_02[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])
                        test_loss_ffn1_w1_01 = F.cross_entropy(pred_t_ffn1_w1_01[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])
                        test_loss_ffn1_w1_0 = F.cross_entropy(pred_t_ffn1_w1_0[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])
                        
                        test_loss_ffn1_w2_02 = F.cross_entropy(pred_t_ffn1_w2_02[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])
                        test_loss_ffn1_w2_01 = F.cross_entropy(pred_t_ffn1_w2_01[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])
                        test_loss_ffn1_w2_0 = F.cross_entropy(pred_t_ffn1_w2_0[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])

                        test_loss_ffn1_norelu = F.cross_entropy(pred_t_ffn1_norelu[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])

                    if secondffn_w1w2_flag:
                        test_loss_ffn2_w1_02 = F.cross_entropy(pred_t_ffn2_w1_02[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])
                        test_loss_ffn2_w1_01 = F.cross_entropy(pred_t_ffn2_w1_01[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])
                        test_loss_ffn2_w1_0 = F.cross_entropy(pred_t_ffn2_w1_0[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])

                        test_loss_ffn2_w2_02 = F.cross_entropy(pred_t_ffn2_w2_02[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])
                        test_loss_ffn2_w2_01 = F.cross_entropy(pred_t_ffn2_w2_01[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])
                        test_loss_ffn2_w2_0 = F.cross_entropy(pred_t_ffn2_w2_0[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])

                    if secondffn_wlin_flag:
                        test_loss_ffn2_wlin_02 = F.cross_entropy(pred_t_ffn2_wlin_02[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])
                        test_loss_ffn2_wlin_01 = F.cross_entropy(pred_t_ffn2_wlin_01[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])
                        test_loss_ffn2_wlin_0 = F.cross_entropy(pred_t_ffn2_wlin_0[:,-el:][outs_t[:,-el:] >= 2], y_t[:,-el:][outs_t[:,-el:] >= 2])


                train_PredN_LabelN, train_PredN_LabelNN, train_PredNN_LabelN, train_PredNN_LabelNN = compare_pred_diversity(pred.argmax(-1)[outs >= 1], y[outs >= 1], shared_noise=cfg.noise_token_id)

                test_PredN_LabelN, test_PredN_LabelNN, test_PredNN_LabelN, test_PredNN_LabelNN = compare_pred_diversity(pred_t.argmax(-1)[outs_t >= 1], y_t[outs_t >= 1], shared_noise=cfg.noise_token_id)
                

                acc_end_test = (pred_t[:,-el:].argmax(-1)[outs_t[:,-el:] >= 2] == y_t[:,-el:][outs_t[:,-el:] >= 2]).float().mean().item()
                acc_end_test_residual = (pred_t_residual[:,-el:].argmax(-1)[outs_t[:,-el:] >= 2] == y_t[:,-el:][outs_t[:,-el:] >= 2]).float().mean().item()


                if firstffn_w1w2_flag:
                    acc_end_test_ffn1_w1_01 = (pred_t_ffn1_w1_01[:,-el:].argmax(-1)[outs_t[:,-el:] >= 2] == y_t[:,-el:][outs_t[:,-el:] >= 2]).float().mean().item()
                    acc_end_test_ffn1_w1_02 = (pred_t_ffn1_w1_02[:,-el:].argmax(-1)[outs_t[:,-el:] >= 2] == y_t[:,-el:][outs_t[:,-el:] >= 2]).float().mean().item()
                    acc_end_test_ffn1_w1_0 = (pred_t_ffn1_w1_0[:,-el:].argmax(-1)[outs_t[:,-el:] >= 2] == y_t[:,-el:][outs_t[:,-el:] >= 2]).float().mean().item()

                    acc_end_test_ffn1_w2_01 = (pred_t_ffn1_w2_01[:,-el:].argmax(-1)[outs_t[:,-el:] >= 2] == y_t[:,-el:][outs_t[:,-el:] >= 2]).float().mean().item()
                    acc_end_test_ffn1_w2_02 = (pred_t_ffn1_w2_02[:,-el:].argmax(-1)[outs_t[:,-el:] >= 2] == y_t[:,-el:][outs_t[:,-el:] >= 2]).float().mean().item()
                    acc_end_test_ffn1_w2_0 = (pred_t_ffn1_w2_0[:,-el:].argmax(-1)[outs_t[:,-el:] >= 2] == y_t[:,-el:][outs_t[:,-el:] >= 2]).float().mean().item()

                    acc_end_test_ffn1_norelu = (pred_t_ffn1_norelu[:,-el:].argmax(-1)[outs_t[:,-el:] >= 2] == y_t[:,-el:][outs_t[:,-el:] >= 2]).float().mean().item()

                if secondffn_w1w2_flag:
                    acc_end_test_ffn2_w1_01 = (pred_t_ffn2_w1_01[:,-el:].argmax(-1)[outs_t[:,-el:] >= 2] == y_t[:,-el:][outs_t[:,-el:] >= 2]).float().mean().item()
                    acc_end_test_ffn2_w1_02 = (pred_t_ffn2_w1_02[:,-el:].argmax(-1)[outs_t[:,-el:] >= 2] == y_t[:,-el:][outs_t[:,-el:] >= 2]).float().mean().item()
                    acc_end_test_ffn2_w1_0 = (pred_t_ffn2_w1_0[:,-el:].argmax(-1)[outs_t[:,-el:] >= 2] == y_t[:,-el:][outs_t[:,-el:] >= 2]).float().mean().item()

                    acc_end_test_ffn2_w2_01 = (pred_t_ffn2_w2_01[:,-el:].argmax(-1)[outs_t[:,-el:] >= 2] == y_t[:,-el:][outs_t[:,-el:] >= 2]).float().mean().item()
                    acc_end_test_ffn2_w2_02 = (pred_t_ffn2_w2_02[:,-el:].argmax(-1)[outs_t[:,-el:] >= 2] == y_t[:,-el:][outs_t[:,-el:] >= 2]).float().mean().item()
                    acc_end_test_ffn2_w2_0 = (pred_t_ffn2_w2_0[:,-el:].argmax(-1)[outs_t[:,-el:] >= 2] == y_t[:,-el:][outs_t[:,-el:] >= 2]).float().mean().item()

                if secondffn_wlin_flag:
                    acc_end_test_ffn2_wlin_01 = (pred_t_ffn2_wlin_01[:,-el:].argmax(-1)[outs_t[:,-el:] >= 2] == y_t[:,-el:][outs_t[:,-el:] >= 2]).float().mean().item()
                    acc_end_test_ffn2_wlin_02 = (pred_t_ffn2_wlin_02[:,-el:].argmax(-1)[outs_t[:,-el:] >= 2] == y_t[:,-el:][outs_t[:,-el:] >= 2]).float().mean().item()
                    acc_end_test_ffn2_wlin_0 = (pred_t_ffn2_wlin_0[:,-el:].argmax(-1)[outs_t[:,-el:] >= 2] == y_t[:,-el:][outs_t[:,-el:] >= 2]).float().mean().item()

                t3_track = time.time()

                wandb.log(
                    {
                     'diversity/train/pred_Noise_label_Noise': train_PredN_LabelN,
                     'diversity/train/pred_Noise_label_NoNoise': train_PredN_LabelNN,
                     'diversity/train/pred_NoNoise_label_Noise': train_PredNN_LabelN,
                     'diversity/train/pred_NoNoise_label_NoNoise': train_PredNN_LabelNN,

                     'diversity/train/no_residual/pred_Noise_label_Noise': train_PredN_LabelN_no_residual,
                     'diversity/train/no_residual/pred_Noise_label_NoNoise': train_PredN_LabelNN_no_residual,
                     'diversity/train/no_residual/pred_NoNoise_label_Noise': train_PredNN_LabelN_no_residual,
                     'diversity/train/no_residual/pred_NoNoise_label_NoNoise': train_PredNN_LabelNN_no_residual,

                     'diversity/test/full/pred_Noise_label_Noise': test_PredN_LabelN,
                     'diversity/test/full/pred_Noise_label_NoNoise': test_PredN_LabelNN,
                     'diversity/test/full/pred_NoNoise_label_Noise': test_PredNN_LabelN,
                     'diversity/test/full/pred_NoNoise_label_NoNoise': test_PredNN_LabelNN,

                     "train_acc/tot": acc_tot,
                     "train_acc/end": acc_end,

                     "train_acc/tot_no_residual": acc_tot_no_residual,
                     "train_loss/no_residual": train_loss_no_residual,

                     },
                     step = i,
                )

                if firstffn_w1w2_flag:
                    wandb.log(
                        {   
                            'test_loss/ffn1_w1/0': test_loss_ffn1_w1_0,
                            'test_loss/ffn1_w1/0.10': test_loss_ffn1_w1_01,
                            'test_loss/ffn1_w1/0.20': test_loss_ffn1_w1_02,
                            'test_loss/ffn1_w1/1.00': test_loss,
                            'test_loss/ffn1_w1/1.00-res': test_loss_residual,

                            'test_acc_end/ffn1_w1/0': acc_end_test_ffn1_w1_0,
                            'test_acc_end/ffn1_w1/0.10': acc_end_test_ffn1_w1_01,
                            'test_acc_end/ffn1_w1/0.20': acc_end_test_ffn1_w1_02,
                            'test_acc_end/ffn1_w1/1.00': acc_end_test,
                            'test_acc_end/ffn1_w1/1.00-res': acc_end_test_residual,


                            'test_loss/ffn1_w2/0': test_loss_ffn1_w2_0,
                            'test_loss/ffn1_w2/0.10': test_loss_ffn1_w2_01,
                            'test_loss/ffn1_w2/0.20': test_loss_ffn1_w2_02,
                            'test_loss/ffn1_w2/1.00': test_loss,
                            'test_loss/ffn1_w2/1.00-res': test_loss_residual,

                            'test_acc_end/ffn1_w2/0': acc_end_test_ffn1_w2_0,
                            'test_acc_end/ffn1_w2/0.10': acc_end_test_ffn1_w2_01,
                            'test_acc_end/ffn1_w2/0.20': acc_end_test_ffn1_w2_02,
                            'test_acc_end/ffn1_w2/1.00': acc_end_test,
                            'test_acc_end/ffn1_w2/1.00-res': acc_end_test_residual,

                            'test_loss/ffn1/norelu': test_loss_ffn1_norelu,
                            'test_acc_end/ffn1/norelu': acc_end_test_ffn1_norelu,
                        },
                        step=i
                    )

                if secondffn_w1w2_flag:
                    wandb.log(
                        {   
                            'test_loss/ffn2_w1/0': test_loss_ffn2_w1_0,
                            'test_loss/ffn2_w1/0.10': test_loss_ffn2_w1_01,
                            'test_loss/ffn2_w1/0.20': test_loss_ffn2_w1_02,
                            'test_loss/ffn2_w1/1.00': test_loss,
                            'test_loss/ffn2_w1/1.00-res': test_loss_residual,

                            'test_acc_end/ffn2_w1/0': acc_end_test_ffn2_w1_0,
                            'test_acc_end/ffn2_w1/0.10': acc_end_test_ffn2_w1_01,
                            'test_acc_end/ffn2_w1/0.20': acc_end_test_ffn2_w1_02,
                            'test_acc_end/ffn2_w1/1.00': acc_end_test,
                            'test_acc_end/ffn2_w1/1.00-res': acc_end_test_residual,

                            'test_loss/ffn2_w2/0': test_loss_ffn2_w2_0,
                            'test_loss/ffn2_w2/0.10': test_loss_ffn2_w2_01,
                            'test_loss/ffn2_w2/0.20': test_loss_ffn2_w2_02,
                            'test_loss/ffn2_w2/1.00': test_loss,
                            'test_loss/ffn2_w2/1.00-res': test_loss_residual,

                            'test_acc_end/ffn2_w2/0': acc_end_test_ffn2_w2_0,
                            'test_acc_end/ffn2_w2/0.10': acc_end_test_ffn2_w2_01,
                            'test_acc_end/ffn2_w2/0.20': acc_end_test_ffn2_w2_02,
                            'test_acc_end/ffn2_w2/1.00': acc_end_test,
                            'test_acc_end/ffn2_w2/1.00-res': acc_end_test_residual,
                        },
                        step=i
                    )

                if secondffn_wlin_flag:
                    wandb.log(
                        {
                            'test_loss/ffn2_wlin/0': test_loss_ffn2_wlin_0,
                            'test_loss/ffn2_wlin/0.10': test_loss_ffn2_wlin_01,
                            'test_loss/ffn2_wlin/0.20': test_loss_ffn2_wlin_02,
                            'test_loss/ffn2_wlin/1.00': test_loss,
                            'test_loss/ffn2_wlin/1.00-res': test_loss_residual,

                            'test_acc_end/ffn2_wlin/0': acc_end_test_ffn2_wlin_0,
                            'test_acc_end/ffn2_wlin/0.10': acc_end_test_ffn2_wlin_01,
                            'test_acc_end/ffn2_wlin/0.20': acc_end_test_ffn2_wlin_02,
                            'test_acc_end/ffn2_wlin/1.00': acc_end_test,
                            'test_acc_end/ffn2_wlin/1.00-res': acc_end_test_residual,
                        },
                        step=i
                    )

                t4_track = time.time()

                if cfg.log_probes:
                    logging.info(f'memory probes wk0: {wk0_acc:.4f} ({wk0_64_acc:.4f}), wk1: {wk1_acc:.4f}, wo1: {wo1_acc:.4f}, ff1: {ff1_loss:.4f}, wo1 srank: {wo1_softrank:.4f}, wo2 srank: {wo2_softrank:.4f}')

                curr_res = {'iter': i, 'loss': loss.item(), 'loss_bigram': loss_bigram, 'loss_head': loss_head,
                            'acc_tot': acc_tot, 'acc_start': acc_start, 'acc_end': acc_end, 'acc_end_test': acc_end_test,
                            'score_acc': score_acc, 'score_start_acc': score_start_acc, 'score2_acc': score2_acc,
                            'score_cond_acc': score_cond_acc,
                            'pred_attended_acc': pred_attended_acc, 'repeat_frac': repeat_frac,
                            'wk0_acc': wk0_acc, 'wk0_64_acc': wk0_64_acc, 'wk1_acc': wk1_acc, 'wo1_acc': wo1_acc, 'ff1_loss': ff1_loss}

                for name, p in model.named_parameters():
                    if p.requires_grad:
                        curr_res['norm_' + name] = p.norm().item()
                        # curr_res['gradnorm_' + name] = p.grad.norm().item()

                
                param_norms = {
                        'wk': [layer.attention.wk.weight.norm().item() for layer in model.layers],
                        'wv': [layer.attention.wv.weight.norm().item() for layer in model.layers],
                        'wo': [layer.attention.wo.weight.norm().item() for layer in model.layers],
                        }
                if cfg.log_norms:
                    logging.info(repr(param_norms))



                if cfg.save_dir is not None:
                    print(json.dumps(curr_res), file=outfile, flush=True)
                res.append(curr_res)
            else:
                logging.info(f'{i} ({dt_data:.2f}, {dt:.2f}, {t - t0:.2f}): {loss.item():.4f}')
                res.append({'loss': loss.item()})

    
        optimizer.zero_grad()
        pred = model(x, no_residual = cfg.no_residual_train)

        if cfg.loss_head_only:
            loss = F.cross_entropy(pred[outs >= 2], y[outs >= 2])
        else:
            loss = F.cross_entropy(pred.flatten(0, 1), y.flatten(0, 1))

        loss.backward()

        optimizer.step()
'''
    This is adjusted from https://github.com/albietz/transformer-birth/blob/main/ihead_data.py
'''


from dataclasses import dataclass
import itertools
import logging
import random
import math
import numpy as np
import pickle
import time
import sys

from typing import List, Optional, Tuple

logging.getLogger().setLevel(logging.INFO)


@dataclass
class DataArgs:
    k: int = 5
    seq_length: int = 256
    show_latents: bool = False
    fixed_special_toks: bool = True
    special_toks_offset: int = 0
    output_counter: bool = True
    no_repeat: bool = False


class Dataset:
    def __init__(self, args: DataArgs,
                 train_test: Optional[str] = None,
                 bigram_outs: Optional[bool] = False,
                 noise_prob: Optional[float] = 0.,
                 noise_token_id: Optional[int] = 2,
                 ):
        self.k = args.k
        self.seq_length = args.seq_length
        self.show_latents = args.show_latents
        self.train_test = train_test
        self.output_counter = args.output_counter
        self.no_repeat = args.no_repeat
        self.bigram_outs = bigram_outs

        self.noise_prob = noise_prob
        self.noise_token_id = noise_token_id

        # init distributions
        self.meta = pickle.load(open('data/meta.pkl', 'rb'))
        self.itos = self.meta['itos']
        self.stoi = self.meta['stoi']
        self.num_tokens = self.meta['vocab_size']
        self.tok_range = list(np.arange(self.num_tokens))

        # OOD
        if self.train_test is not None:
            assert not self.bigram_outs  # this requires distributions over all tokens
            self.n_train_toks = int(0.75 * self.num_tokens)
        else:
            self.n_train_toks = self.num_tokens

        # marginal probabilities over characters
        self.marginal = np.zeros(self.num_tokens)
        for k, cnt in self.meta['unigrams'].items():
            self.marginal[self.stoi[k]] = cnt
        self.marginal /= self.marginal.sum()

        # conditionals
        self.cond = [np.zeros(self.num_tokens) for _ in range(self.num_tokens)]
        for (w1, w2), cnt in self.meta['bigrams'].items():
            self.cond[self.stoi[w1]][self.stoi[w2]] += cnt
        for i in range(self.num_tokens):
            self.cond[i] /= self.cond[i].sum()

        # special tokens
        self.idxs = None
        if args.fixed_special_toks:
            # use unigram marginals
            self.idxs = list(self.marginal.argsort()[self.num_tokens-args.special_toks_offset-self.k:self.num_tokens-args.special_toks_offset])

    def decode(self, idxs: List[int]) -> str:
        return ''.join(self.itos[idx] for idx in idxs)

    def gen_seq(self, rng: np.random.Generator):
        # select special tokens for this sequence
        if self.idxs is not None:
            idxs = self.idxs
        else:
            idxs = list(rng.choice(self.tok_range, p=self.marginal, size=self.k, replace=False))
        # for each special token, select a special next token
        # outs = [rng.choice(self.tok_range, p=self.cond[idx]) for idx in idxs]

        if self.no_repeat:  # prevent next token to be same as idx
            pools = [self.tok_range.copy() for idx in idxs]
            for i, idx in enumerate(idxs):
                pools[i].remove(idx)
        else:
            pools = [self.tok_range for idx in idxs]

        if self.train_test is None:
            # outs = [rng.choice(self.tok_range) for idx in idxs]
            if self.bigram_outs:
                outs = [rng.choice(pool, p=(self.cond[idx][pool] / self.cond[idx][pool].sum())) for pool, idx in zip(pools, idxs)]
            else:
                outs = [rng.choice(pool) for pool in pools]
        elif self.train_test == 'train':
            # outs = [rng.choice(self.tok_range[:n_train_toks]) for idx in idxs]
            outs = [rng.choice(pool[:self.n_train_toks]) for pool in pools]
        elif self.train_test == 'test':
            # outs = [rng.choice(self.tok_range[n_train_toks:]) for idx in idxs]
            outs = [rng.choice(pool[self.n_train_toks:]) for pool in pools]
        else:
            assert False

        cnts = {}

        if self.show_latents:
            seq = idxs.copy()
            outputs_seq = [-1] * len(idxs) #  []
        else:
            seq = []
            outputs_seq = []
        seq += [rng.choice(self.tok_range, p=self.marginal)]
        while len(seq) < self.seq_length + 1:
            last = seq[-1]
            if last in idxs:
                # first is random
                if self.noise_prob > 0:
                    random_float = random.uniform(0, 1)
                    if random_float >= self.noise_prob:
                        seq.append(outs[idxs.index(last)])
                    else:
                        seq.append(self.noise_token_id) # This makes a fixed noise with probability
                else:
                    seq.append(outs[idxs.index(last)]) 

                if self.output_counter:
                    cnts[last] = cnts.get(last, 0) + 1
                    outputs_seq.append(cnts[last])
                else:
                    outputs_seq.append(1)
            else:
                probs = self.cond[last]
                outputs_seq.append(0)
                seq.append(rng.choice(self.tok_range, p=probs))
        outputs_seq.append(0)

        return seq, outputs_seq

    def gen_seqs(self, rng: np.random.Generator) -> List[str]:
        while True:
            seq, outputs_seq = self.gen_seq(rng)
            yield (seq, outputs_seq)

    def gen_batch(self, rng: np.random.Generator, batch_size: int):
        seqs = []
        outs = []
        for _ in range(batch_size):
            seq, out = self.gen_seq(rng)
            seqs += seq
            outs += out
        x = np.array(seqs).reshape(batch_size, self.seq_length + 1)
        outs = np.array(outs).reshape(batch_size, self.seq_length + 1)
        return x, outs


def iterate_batches(dataset: Dataset,
                    batch_size: int = 20,
                    num_workers: int = 60,
                    seed: int = 42):
    def worker(queue, rng):
        while True:
            x, outs = dataset.gen_batch(rng, batch_size)
            queue.put((x, outs))

    import multiprocessing as mp
    q = mp.Queue(maxsize=1000)
    processes = [mp.Process(target=worker, args=(q, np.random.default_rng([seed, i]))) for i in range(num_workers)]
    for p in processes:
        p.start()

    seq = []
    outputs_seq = []
    count = 0
    try:
        while True:
            x, outs = q.get()
            yield (x[:,:-1], x[:,1:], outs[:,:-1])
    except:
        for p in processes:
            p.kill()

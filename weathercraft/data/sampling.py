"""Shuffle without replacement; retain only complete global optimizer batches."""

import torch
from torch.utils.data import Sampler


class FullGlobalBatchSampler(Sampler):
    def __init__(self, dataset, rank, world_size, global_batch=32, seed=0):
        assert global_batch % world_size == 0
        self.size = len(dataset)
        self.rank, self.world = rank, world_size
        self.retained = self.size // global_batch * global_batch
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.retained // self.world

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=g).tolist()[: self.retained]
        return iter(indices[self.rank :: self.world])

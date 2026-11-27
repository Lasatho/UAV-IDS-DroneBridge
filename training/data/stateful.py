"""
Stateful chunk loader: whole missions in consecutive chunks (opt-in
`training.stateful`, RSSM only).

Instead of overlapping windows with a fresh state each, `lanes` missions
are walked in parallel in non-overlapping chunks of `chunk_length` steps;
the model carries its state from chunk to chunk of the same mission
(truncated backpropagation through time) and resets it only when a lane
starts a new mission. This matches on-board use, where the model is
stepped once per incoming sample with constant cost. A mission tail
shorter than one chunk is dropped. When no missions are left, finished
lanes stay empty and the batch shrinks until all lanes are done.

Each batch: (x (B, C, F) normalized, reset (B,) bool, lane (B,) long,
chunk index within the mission (B,) long). `lane` indexes the state the
caller keeps for every lane (self.lanes rows).

Pool mode (opt-in `pool` > batch, training; Vault modellumbau_rssm_mtsjepa,
Befund Validierungsverlust im Missionsmodus, decided 2026-11-11): the plain
mode gives correlated batches (the same missions for ~50 consecutive
updates, so the weights adapt online to them) and shrinking batches at the
end of an epoch (the last ~180 updates see only a few long missions, right
before validation and checkpoint). In pool mode `pool` missions run at the
same time, each with its own state slot; every step draws `batch` of them
at random (without replacement) and advances only those. A finished slot
takes the next mission of a shuffled order; when the order is used up a new
shuffle follows, so every batch is full. An epoch is a fixed number of
steps (default: chunks of all missions / batch, the same amount of data as
the plain mode).
"""

from __future__ import annotations

import numpy as np
import torch


class StatefulChunkLoader:
    def __init__(self, missions, normalizer, chunk_length: int, lanes: int,
                 shuffle: bool, seed: int = 0, pool: int | None = None,
                 steps: int | None = None):
        self.missions = missions
        self.norm = normalizer
        self.C = int(chunk_length)
        self.batch = int(lanes)
        self.pool = int(pool) if pool else None
        if self.pool is not None and self.pool < self.batch:
            raise ValueError(f"pool {self.pool} < batch {self.batch}")
        self.lanes = self.pool or self.batch  # state rows the caller keeps
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.n_chunks = sum(m.shape[0] // self.C for m in missions)
        self.steps = int(steps) if steps else int(np.ceil(self.n_chunks / self.batch))

    def __len__(self) -> int:  # number of batches (approximate in plain mode)
        return self.steps if self.pool else int(np.ceil(self.n_chunks / self.batch))

    def __iter__(self):
        if self.pool:
            yield from self._iter_pool()
            return
        yield from self._iter_plain()

    def _iter_pool(self):
        valid = [i for i in range(len(self.missions)) if self.missions[i].shape[0] >= self.C]
        order = []

        def next_mission():
            nonlocal order
            if not order:
                order = list(valid)
                if self.shuffle:
                    self.rng.shuffle(order)
            return order.pop()
        cur = [next_mission() for _ in range(self.pool)]
        pos = [0] * self.pool
        fresh = [True] * self.pool
        for _ in range(self.steps):
            pick = np.sort(self.rng.choice(self.pool, self.batch, replace=False))
            xs, rs, ps = [], [], []
            for k in pick:
                m = cur[k]; s = pos[k] * self.C
                xs.append(self.norm(np.asarray(self.missions[m][s:s + self.C], np.float32)))
                rs.append(fresh[k]); ps.append(pos[k])
                fresh[k] = False
                pos[k] += 1
                if (pos[k] + 1) * self.C > self.missions[m].shape[0]:  # mission done
                    cur[k] = next_mission(); pos[k] = 0; fresh[k] = True
            yield (torch.from_numpy(np.stack(xs)), torch.tensor(rs), torch.from_numpy(pick.astype(np.int64)),
                   torch.tensor(ps))

    def _iter_plain(self):
        order = list(range(len(self.missions)))
        if self.shuffle:
            self.rng.shuffle(order)
        order = [i for i in order if self.missions[i].shape[0] >= self.C]
        nxt = 0
        cur = [None] * self.batch      # mission index per lane
        pos = [0] * self.batch         # chunk index within the mission
        fresh = [True] * self.batch
        for k in range(self.batch):
            if nxt < len(order):
                cur[k] = order[nxt]; nxt += 1
        while any(c is not None for c in cur):
            xs, rs, ls, ps = [], [], [], []
            for k in range(self.batch):
                m = cur[k]
                if m is None:
                    continue
                s = pos[k] * self.C
                xs.append(self.norm(np.asarray(self.missions[m][s:s + self.C], np.float32)))
                rs.append(fresh[k]); ls.append(k); ps.append(pos[k])
                fresh[k] = False
                pos[k] += 1
                if (pos[k] + 1) * self.C > self.missions[m].shape[0]:  # mission done
                    cur[k] = order[nxt] if nxt < len(order) else None
                    nxt += cur[k] is not None
                    pos[k] = 0; fresh[k] = True
            yield (torch.from_numpy(np.stack(xs)), torch.tensor(rs), torch.tensor(ls),
                   torch.tensor(ps))

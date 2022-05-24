import numpy as np
from collections import namedtuple

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, limit):
        self.obs_buf = Memory(shape=(obs_dim,), limit=limit)
        self.act_buf = Memory(shape=(act_dim,), limit=limit)
        self.rew_buf = Memory(shape=(1,), limit=limit)
        self.next_obs_buf = Memory(shape=(obs_dim,), limit=limit)
        self.done_buf = Memory(shape=(1,), limit=limit)

        self.limit = limit
        self.size = 0

    def append(self, obs, act, next_obs, reward, done):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(reward)
        self.next_obs_buf.append(next_obs)
        self.done_buf.append(done)

        self.size = self.obs_buf.size

    def sample_batch(self, batch_size):
        # idxs must be synchronized across different types of memories
        rng = np.random.default_rng()
        idxs = rng.choice(self.size, batch_size)

        # get batch from each buffer
        obs_batch = self.obs_buf.get_batch(idxs)
        act_batch = self.act_buf.get_batch(idxs)
        rew_batch = self.rew_buf.get_batch(idxs)
        next_obs_batch = self.next_obs_buf.get_batch(idxs)
        done_batch = self.done_buf.get_batch(idxs)

        batch = Batch(obs=obs_batch, act=act_batch, next_obs=next_obs_batch, rew=rew_batch, done=done_batch)

        return batch


# batch definition as python named tuple
Batch = namedtuple('Batch', ['obs', 'act', 'next_obs', 'rew', 'done'])
    

class Memory:
    """
    implementation of a circular buffer
    """

    def __init__(self, shape, limit=1000000, dtype=np.float):
        self.start = 0
        self.data_shape = shape
        self.size = 0
        self.dtype = dtype
        self.limit = limit
        self.data = np.zeros((self.limit,) + shape)

    def append(self, data):
        if self.size < self.limit:
            self.size += 1
        else:
            self.start = (self.start + 1) % self.limit

        self.data[(self.start + self.size - 1) % self.limit] = data

    def get_batch(self, idxs):

        return self.data[(self.start + idxs) % self.limit]
    
    @property
    def __len__(self):
        return self.size

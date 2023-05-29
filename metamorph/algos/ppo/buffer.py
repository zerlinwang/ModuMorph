import gym
import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import SubsetRandomSampler

from metamorph.config import cfg


class Buffer(object):
    def __init__(self, obs_space, act_shape):
        T, P = cfg.PPO.TIMESTEPS, cfg.PPO.NUM_ENVS

        # Temporal history info length
        K, C = cfg.HI.MAX_LENGTH, cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE
        self.context_embed = torch.zeros(T, P, C)
        self.hi_context_embed = torch.zeros(T, K, P, C)

        if isinstance(obs_space, gym.spaces.Dict):
            self.obs = {}
            for obs_type, obs_space_ in obs_space.spaces.items():
                self.obs[obs_type] = torch.zeros(T, P, *obs_space_.shape)
        else:
            self.obs = torch.zeros(T, P, *obs_space.shape)
            self.hi_obs = torch.zeros(T, P, *obs_space.shape)

        self.act = torch.zeros(T, P, *act_shape)
        self.hi_act = torch.zeros(T, K, P, *act_shape)
        self.val = torch.zeros(T, P, 1)
        self.rew = torch.zeros(T, P, 1)
        self.ret = torch.zeros(T, P, 1)
        self.logp = torch.zeros(T, P, 1)
        self.masks = torch.ones(T, P, 1)
        self.timeout = torch.ones(T, P, 1)
        self.dropout_mask_v = torch.ones(T, P, 12, 128)
        self.dropout_mask_mu = torch.ones(T, P, 12, 128)
        self.unimal_ids = torch.zeros(T, P).long()

        # indicate no-exist history (1. means exists)
        self.hi_masks = torch.ones(T, K, P, 1)

        self.step = 0

    def to(self, device):
        self.context_embed = self.context_embed.to(device)
        self.hi_context_embed = self.hi_context_embed.to(device)
        if isinstance(self.obs, dict):
            for obs_type, obs_space in self.obs.items():
                self.obs[obs_type] = self.obs[obs_type].to(device)
        else:
            self.obs = self.obs.to(device)
        self.act = self.act.to(device)
        self.hi_act = self.hi_act.to(device)
        self.val = self.val.to(device)
        self.rew = self.rew.to(device)
        self.ret = self.ret.to(device)
        self.logp = self.logp.to(device)
        self.masks = self.masks.to(device)
        self.timeout = self.timeout.to(device)
        self.dropout_mask_v = self.dropout_mask_v.to(device)
        self.dropout_mask_mu = self.dropout_mask_mu.to(device)
        self.unimal_ids = self.unimal_ids.to(device)
        self.hi_masks = self.hi_masks.to(device)

    def insert(self, obs, act, logp, val, rew, masks, timeouts, dropout_mask_v, dropout_mask_mu, unimal_ids):
        if isinstance(obs, dict):
            for obs_type, obs_val in obs.items():
                self.obs[obs_type][self.step] = obs_val
        else:
            self.obs[self.step] = obs
        self.act[self.step] = act   # [B, 24]
        self.val[self.step] = val   # [B, 1]
        self.rew[self.step] = rew   # [B, 1]
        self.logp[self.step] = logp # [B, 1]
        self.masks[self.step] = masks   # [B, 1] done env
        self.timeout[self.step] = timeouts  # [B, 1] timeouts env
        self.dropout_mask_v[self.step] = dropout_mask_v # float 0.0
        self.dropout_mask_mu[self.step] = dropout_mask_mu   # float 0.0
        self.unimal_ids[self.step] = torch.LongTensor(unimal_ids)   # list of length B

        self.step = (self.step + 1) % cfg.PPO.TIMESTEPS

    def set_history_info(self):
        """
        History informations, i.e., the lastest $K$-steps obs and acts are recorded in buffer considering the masks and timeout
        self.context_embed: torch.Tensors with shape of [T, P, C]
        self.hi_context_embed: dict of torch.Tensors with shape of [T, K, P, C]
        self.act: torch.Tensor with shape of [T, P, *act_shape]
        self.hi_act: torch.Tensor with shape of [T, K, P, *act_shape]
        self.masks: torch.Tensor with shape of [T, P, 1], denotes the timesteps where agent died or timeout (where mask = 0.)
        self.hi_masks: torch.Tensor with shape of [T, K, P, 1], denotes the timesteps where is no history infos (where hi_mask = 0.)
        """
        T, K = self.hi_context_embed.shape[:1]
        for t in range(T):
            # hi_context_embed
            self.hi_context_embed[t] = torch.stack(self.context_embed[t-K: t], dim=0)
            # hi_act
            self.hi_act[t] = torch.stack(self.act[t-K:t], dim=0)
            # hi_masks
            for k in range(K):
                if k > t:
                    break
                elif self.masks[t-k] == 0.:
                    self.hi_masks[t][:-k] == 0.
                    break

    def compute_returns(self, next_value):
        """
        We use ret as approximate gt for value function for training. When step
        is terminal state we need to handle two cases:
        1. Agent Died: timeout[step] = 1 and mask[step] = 0. This ensures
           gae is reset to 0 and self.ret[step] = 0.
        2. Agent Alive but done true due to timeout: timeout[step] = 0
           mask[step] = 0. This ensures gae = 0 and self.ret[step] = val[step].
        """
        gamma, gae_lambda = cfg.PPO.GAMMA, cfg.PPO.GAE_LAMBDA
        # val: (T+1, P, 1), self.val: (T, P, 1) next_value: (P, 1)
        val = torch.cat((self.val.squeeze(), next_value.t())).unsqueeze(2)
        gae = 0
        for step in reversed(range(cfg.PPO.TIMESTEPS)):
            delta = (
                self.rew[step]
                + gamma * val[step + 1] * self.masks[step]
                - val[step]
            ) * self.timeout[step]
            gae = delta + gamma * gae_lambda * self.masks[step] * gae
            self.ret[step] = gae + val[step]

    def get_sampler(self, adv):
        dset_size = cfg.PPO.TIMESTEPS * cfg.PPO.NUM_ENVS

        assert dset_size >= cfg.PPO.BATCH_SIZE

        sampler = BatchSampler(
            SubsetRandomSampler(range(dset_size)),
            cfg.PPO.BATCH_SIZE,
            drop_last=True,
        )

        for idxs in sampler:
            batch = {}
            batch["ret"] = self.ret.view(-1, 1)[idxs]

            if isinstance(self.obs, dict):
                batch["obs"] = {}
                for ot, ov in self.obs.items():
                        batch["obs"][ot] = ov.view(-1, *ov.size()[2:])[idxs]
            else:
                batch["obs"] = self.obs.view(-1, *self.obs.size()[2:])[idxs]

            batch["val"] = self.val.view(-1, 1)[idxs]
            batch["act"] = self.act.view(-1, self.act.size(-1))[idxs]
            batch["adv"] = adv.view(-1, 1)[idxs]
            batch["logp_old"] = self.logp.view(-1, 1)[idxs]
            batch["dropout_mask_v"] = self.dropout_mask_v.view(-1, 12, 128)[idxs]
            batch["dropout_mask_mu"] = self.dropout_mask_mu.view(-1, 12, 128)[idxs]
            batch["unimal_ids"] = self.unimal_ids.view(-1)[idxs]
            yield batch

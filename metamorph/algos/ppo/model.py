import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

from metamorph.config import cfg
from metamorph.utils import model as tu

from .transformer import TransformerEncoder
from .transformer import TransformerEncoderLayerResidual

import time
import matplotlib.pyplot as plt


# MLP model as single-robot baseline
class MLPModel(nn.Module):
    def __init__(self, obs_space, out_dim):
        super(MLPModel, self).__init__()
        self.model_args = cfg.MODEL.MLP
        self.seq_len = cfg.MODEL.MAX_LIMBS  # FT 12
        self.limb_obs_size = limb_obs_size = obs_space["proprioceptive"].shape[0] // self.seq_len
        self.limb_out_dim = out_dim // self.seq_len

        self.input_layer = nn.Linear(obs_space["proprioceptive"].shape[0], self.model_args.HIDDEN_DIM)  # FT 624 // 12 = 52

        self.output_layer = nn.Linear(self.model_args.HIDDEN_DIM, out_dim)
        
        if "hfield" in cfg.ENV.KEYS_TO_KEEP:
            self.hfield_encoder = MLPObsEncoder(obs_space.spaces["hfield"].shape[0])
            hidden_dims = [self.model_args.HIDDEN_DIM + self.hfield_encoder.obs_feat_dim] + [self.model_args.HIDDEN_DIM for _ in range(self.model_args.LAYER_NUM - 1)]
            self.hidden_layers = tu.make_mlp_default(hidden_dims)
        else:
            hidden_dims = [self.model_args.HIDDEN_DIM for _ in range(self.model_args.LAYER_NUM)]
            self.hidden_layers = tu.make_mlp_default(hidden_dims)

    def forward(self, obs, obs_mask, obs_env, obs_cm_mask, obs_context, morphology_info, return_attention=False, dropout_mask=None, unimal_ids=None):

        embedding = self.input_layer(obs)
        embedding = F.relu(embedding)

        if "hfield" in cfg.ENV.KEYS_TO_KEEP:
            hfield_embedding = self.hfield_encoder(obs_env["hfield"])
            embedding = torch.cat([embedding, hfield_embedding], 1)
        
        # hidden layers
        embedding = self.hidden_layers(embedding)

        # output layer
        output = self.output_layer(embedding)

        return output, None, 0.


# J: Max num joints between two limbs. 1 for 2D envs, 2 for unimal
class TransformerModel(nn.Module):
    def __init__(self, obs_space, decoder_out_dim):
        super(TransformerModel, self).__init__()

        self.decoder_out_dim = decoder_out_dim

        self.model_args = cfg.MODEL.TRANSFORMER
        self.seq_len = cfg.MODEL.MAX_LIMBS
        # Embedding layer for per limb obs
        limb_obs_size = obs_space["proprioceptive"].shape[0] // self.seq_len
        self.d_model = cfg.MODEL.LIMB_EMBED_SIZE
        if self.model_args.PER_NODE_EMBED:
            print ('independent weights for each node')
            initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
            self.limb_embed_weights = nn.Parameter(torch.zeros(self.seq_len, len(cfg.ENV.WALKERS), limb_obs_size, self.d_model).uniform_(-initrange, initrange))
            self.limb_embed_bias = nn.Parameter(torch.zeros(self.seq_len, len(cfg.ENV.WALKERS), self.d_model))
        else:
            self.limb_embed = nn.Linear(limb_obs_size, self.d_model)
        self.ext_feat_fusion = self.model_args.EXT_MIX

        self.tt = TemporalTransformer(self.d_model, 2)

        if self.model_args.POS_EMBEDDING == "learnt":
            print ('use PE learnt')
            seq_len = self.seq_len
            self.pos_embedding = PositionalEncoding(self.d_model, seq_len)
        elif self.model_args.POS_EMBEDDING == "abs":
            print ('use PE abs')
            self.pos_embedding = PositionalEncoding1D(self.d_model, self.seq_len)

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayerResidual(
            cfg.MODEL.LIMB_EMBED_SIZE * 2,  # 128 * 2 = 256
            self.model_args.NHEAD,
            self.model_args.DIM_FEEDFORWARD,
            self.model_args.DROPOUT,
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layers, self.model_args.NLAYERS, norm=None,
        )

        # Map encoded observations to per node action mu or critic value
        decoder_input_dim = self.d_model * 2

        # Task based observation encoder
        if "hfield" in cfg.ENV.KEYS_TO_KEEP:
            self.hfield_encoder = MLPObsEncoder(obs_space.spaces["hfield"].shape[0])

        if self.ext_feat_fusion == "late":
            decoder_input_dim += self.hfield_encoder.obs_feat_dim
        self.decoder_input_dim = decoder_input_dim  # Dim of output of each transformer block, 128

        if self.model_args.PER_NODE_DECODER:
            # only support a single output layer
            initrange = cfg.MODEL.TRANSFORMER.DECODER_INIT
            self.decoder_weights = torch.zeros(decoder_input_dim, decoder_out_dim).uniform_(-initrange, initrange)
            self.decoder_weights = self.decoder_weights.repeat(self.seq_len, len(cfg.ENV.WALKERS), 1, 1)
            self.decoder_weights = nn.Parameter(self.decoder_weights)
            self.decoder_bias = torch.zeros(decoder_out_dim).uniform_(-initrange, initrange)
            self.decoder_bias = self.decoder_bias.repeat(self.seq_len, len(cfg.ENV.WALKERS), 1)
            self.decoder_bias = nn.Parameter(self.decoder_bias)
        else:
            self.decoder = tu.make_mlp_default(
                [decoder_input_dim] + self.model_args.DECODER_DIMS + [decoder_out_dim], # FT [128, 1 (v_net) / 2 (pi)]
                final_nonlinearity=False,
            )

        if self.model_args.FIX_ATTENTION:
            print ('use fix attention')
            # the network to generate context embedding from the morphology context
            context_obs_size = obs_space["context"].shape[0] // self.seq_len
            self.context_embed_attention = nn.Linear(context_obs_size, self.model_args.CONTEXT_EMBED_SIZE)

            if self.model_args.CONTEXT_ENCODER == 'transformer':
                print ('use transformer context encoder')
                context_encoder_layers = TransformerEncoderLayerResidual(
                    self.model_args.CONTEXT_EMBED_SIZE,
                    self.model_args.NHEAD,
                    self.model_args.DIM_FEEDFORWARD,
                    self.model_args.DROPOUT,
                )
                self.context_encoder_attention = TransformerEncoder(
                    context_encoder_layers, self.model_args.CONTEXT_LAYER, norm=None,
                )
            else: # MLP context encoder: the default choice
                print ('use MLP context encoder')
                modules = [nn.ReLU()]
                for _ in range(self.model_args.LINEAR_CONTEXT_LAYER):
                    modules.append(nn.Linear(self.model_args.CONTEXT_EMBED_SIZE, self.model_args.CONTEXT_EMBED_SIZE))
                    modules.append(nn.ReLU())
                self.context_encoder_attention = nn.Sequential(*modules)

            # whether to include hfield in attention map computation: default to False
            if self.model_args.HFIELD_IN_FIX_ATTENTION:
                self.context_hfield_encoder = MLPObsEncoder(obs_space.spaces["hfield"].shape[0])
                self.context_compress = nn.Sequential(
                    nn.Linear(self.model_args.EXT_HIDDEN_DIMS[-1] + self.model_args.CONTEXT_EMBED_SIZE, self.model_args.CONTEXT_EMBED_SIZE), 
                    nn.ReLU(), 
                )

        if self.model_args.HYPERNET:
            print ('use HN')
            # the network to generate context embedding from the morphology context
            context_obs_size = obs_space["context"].shape[0] // self.seq_len
            self.context_embed_HN = nn.Linear(context_obs_size, self.model_args.CONTEXT_EMBED_SIZE)
            
            if self.model_args.HN_CONTEXT_ENCODER == 'linear': # the default architecture choice
                modules = [nn.ReLU()]
                for _ in range(self.model_args.HN_CONTEXT_LAYER_NUM):
                    modules.append(nn.Linear(self.model_args.CONTEXT_EMBED_SIZE, self.model_args.CONTEXT_EMBED_SIZE))
                    modules.append(nn.ReLU())
                self.context_encoder_HN = nn.Sequential(*modules)
            elif self.model_args.HN_CONTEXT_ENCODER == 'transformer':
                context_encoder_layers = TransformerEncoderLayerResidual(
                    self.model_args.CONTEXT_EMBED_SIZE,
                    self.model_args.NHEAD,
                    self.model_args.DIM_FEEDFORWARD,
                    self.model_args.DROPOUT,
                )
                self.context_encoder_HN = TransformerEncoder(
                    context_encoder_layers, self.model_args.HN_CONTEXT_LAYER_NUM, norm=None,
                )

            HN_input_dim = self.model_args.CONTEXT_EMBED_SIZE

            self.hnet_embed_weight = nn.Linear(HN_input_dim, limb_obs_size * self.d_model)
            self.hnet_embed_bias = nn.Linear(HN_input_dim, self.d_model)

            self.decoder_dims = [decoder_input_dim] + self.model_args.DECODER_DIMS + [decoder_out_dim]

            self.hnet_decoder_weight = []
            self.hnet_decoder_bias = []
            for i in range(len(self.decoder_dims) - 1):
                layer_w = nn.Linear(HN_input_dim, self.decoder_dims[i] * self.decoder_dims[i + 1])
                self.hnet_decoder_weight.append(layer_w)
                layer_b = nn.Linear(HN_input_dim, self.decoder_dims[i + 1])
                self.hnet_decoder_bias.append(layer_b)
            self.hnet_decoder_weight = nn.ModuleList(self.hnet_decoder_weight)
            self.hnet_decoder_bias = nn.ModuleList(self.hnet_decoder_bias)

        # whether to use SWAT PE and RE: default to False
        if self.model_args.USE_SWAT_PE:
            self.swat_PE_encoder = SWATPEEncoder(self.d_model, self.seq_len)
        if self.model_args.USE_SEPARATE_PE:
            self.separate_PE_encoder = SeparatePEEncoder(self.d_model, self.seq_len)

        self.dropout = nn.Dropout(p=0.1)

        self.init_weights()
        self.count = 0

    def init_weights(self):
        # init obs embedding
        if not self.model_args.PER_NODE_EMBED:
            initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
            self.limb_embed.weight.data.uniform_(-initrange, initrange)
        # init decoder
        if not self.model_args.PER_NODE_DECODER:
            initrange = cfg.MODEL.TRANSFORMER.DECODER_INIT
            self.decoder[-1].bias.data.zero_()
            self.decoder[-1].weight.data.uniform_(-initrange, initrange)

        if self.model_args.FIX_ATTENTION:
            initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
            self.context_embed_attention.weight.data.uniform_(-initrange, initrange)

        if self.model_args.HYPERNET:
            initrange = cfg.MODEL.TRANSFORMER.HN_EMBED_INIT
            self.context_embed_HN.weight.data.uniform_(-initrange, initrange)

            # initialize the hypernet following https://arxiv.org/abs/2210.11348
            initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
            self.hnet_embed_weight.weight.data.zero_()
            self.hnet_embed_weight.bias.data.uniform_(-initrange, initrange)
            self.hnet_embed_bias.weight.data.zero_()
            self.hnet_embed_bias.bias.data.zero_()

            initrange = cfg.MODEL.TRANSFORMER.DECODER_INIT
            for i in range(len(self.hnet_decoder_weight)):
                self.hnet_decoder_weight[i].weight.data.zero_()
                self.hnet_decoder_weight[i].bias.data.uniform_(-initrange, initrange)
                self.hnet_decoder_bias[i].weight.data.zero_()
                self.hnet_decoder_bias[i].bias.data.zero_()

    def forward(self, obs, hi_encoded_feature, hi_act, hi_masks, obs_mask, obs_env, obs_cm_mask, obs_context, morphology_info, return_attention=False, dropout_mask=None, unimal_ids=None):
        # (num_limbs, batch_size, limb_obs_size) -> (num_limbs, batch_size, d_model)
        _, batch_size, limb_obs_size = obs.shape

        hi_act = hi_act.reshape(batch_size, cfg.HI.MAX_LENGTH, hi_act.shape[-1] // 2, 2)    # [N]
        morph_context, dynamic_context = self.tt(hi_encoded_feature, hi_act, hi_masks)

        if "hfield" in cfg.ENV.KEYS_TO_KEEP:
            # (batch_size, embed_size)
            hfield_obs = self.hfield_encoder(obs_env["hfield"])

        if self.ext_feat_fusion in ["late"]:
            hfield_obs = hfield_obs.repeat(self.seq_len, 1)
            hfield_obs = hfield_obs.reshape(self.seq_len, batch_size, -1)

        if self.model_args.FIX_ATTENTION:
            context_embedding_attention = self.context_embed_attention(obs_context)

            if self.model_args.CONTEXT_ENCODER == 'transformer':
                context_embedding_attention = self.context_encoder_attention(
                    context_embedding_attention, 
                    src_key_padding_mask=obs_mask, 
                    morphology_info=morphology_info)
            else:
                context_embedding_attention = self.context_encoder_attention(context_embedding_attention)

            if self.model_args.HFIELD_IN_FIX_ATTENTION:
                hfield_embedding = self.context_hfield_encoder(obs_env["hfield"])
                hfield_embedding = hfield_embedding.repeat(self.seq_len, 1).reshape(self.seq_len, batch_size, -1)
                context_embedding_attention = torch.cat([context_embedding_attention, hfield_embedding], dim=-1)
                context_embedding_attention = self.context_compress(context_embedding_attention)

        if self.model_args.HYPERNET:
            context_embedding_HN = self.context_embed_HN(obs_context)
            context_embedding_HN = self.context_encoder_HN(context_embedding_HN)

        if self.model_args.HYPERNET and self.model_args.HN_EMBED:
            embed_weight = self.hnet_embed_weight(context_embedding_HN).reshape(self.seq_len, batch_size, limb_obs_size, self.d_model)
            embed_bias = self.hnet_embed_bias(context_embedding_HN)
            obs_embed = (obs[:, :, :, None] * embed_weight).sum(dim=-2, keepdim=False) + embed_bias
        else:
            if self.model_args.PER_NODE_EMBED:
                obs_embed = (obs[:, :, :, None] * self.limb_embed_weights[:, unimal_ids, :, :]).sum(dim=-2, keepdim=False) + self.limb_embed_bias[:, unimal_ids, :]
            else:
                obs_embed = self.limb_embed(obs)    # [N, B, obs_dim52] -> [N, B, h_dim128]

        # concate morph context
        obs_embed = torch.cat([obs_embed, morph_context], axis=-1)
        
        if self.model_args.EMBEDDING_SCALE:
            obs_embed *= math.sqrt(self.d_model)

        attention_maps = None

        # add PE
        if self.model_args.POS_EMBEDDING in ["learnt", "abs"]:
            obs_embed = self.pos_embedding(obs_embed)
        if self.model_args.USE_SWAT_PE:
            obs_embed = self.swat_PE_encoder(obs_embed, morphology_info['traversals'])

        # dropout
        if self.model_args.EMBEDDING_DROPOUT:
            if self.model_args.CONSISTENT_DROPOUT:
                # do dropout in a consistent way. Refer to Appendix in the paper
                if dropout_mask is None:
                    obs_embed_after_dropout = self.dropout(obs_embed)
                    dropout_mask = torch.where(obs_embed_after_dropout == 0., 0., 1.).permute(1, 0, 2)
                    obs_embed = obs_embed_after_dropout
                else:
                    obs_embed = obs_embed * dropout_mask.permute(1, 0, 2) / 0.9
            else:
                # do dropout in an inconsistent way, as in MetaMorph
                obs_embed = self.dropout(obs_embed)
                dropout_mask = 0.
        else:
            # do not do dropout
            dropout_mask = 0.

        if self.model_args.FIX_ATTENTION:
            context_to_base = context_embedding_attention
        else:
            context_to_base = None
        
        if self.model_args.USE_SWAT_RE:
            attn_mask = morphology_info['SWAT_RE']
        else:
            attn_mask = None
        src_key_padding_mask = obs_mask

        if return_attention:
            obs_embed_t, attention_maps = self.transformer_encoder.get_attention_maps(
                obs_embed, 
                mask=attn_mask, 
                src_key_padding_mask=src_key_padding_mask, 
                context=context_to_base, 
                morphology_info=morphology_info
            )
        else:
            # (num_limbs, batch_size, d_model)
            obs_embed_t = self.transformer_encoder(
                obs_embed, 
                mask=attn_mask, 
                src_key_padding_mask=src_key_padding_mask, 
                context=context_to_base, 
                morphology_info=morphology_info
            )

        encoded_feature = obs_embed_t.detach().clone().permute(1, 0, 2)
        
        decoder_input = obs_embed_t # [N, B, 128] as the dims of output and input of transformer model are the same
        decoder_input = torch.cat([obs_embed_t, dynamic_context], dim=-1)
        if "hfield" in cfg.ENV.KEYS_TO_KEEP and self.ext_feat_fusion == "late":
            decoder_input = torch.cat([decoder_input, hfield_obs], axis=2)

        # (num_limbs, batch_size, J)
        if self.model_args.HYPERNET and self.model_args.HN_DECODER:
            output = decoder_input
            layer_num = len(self.hnet_decoder_weight)
            for i in range(layer_num):
                layer_w = self.hnet_decoder_weight[i](context_embedding_HN).reshape(self.seq_len, batch_size, self.decoder_dims[i], self.decoder_dims[i + 1])
                layer_b = self.hnet_decoder_bias[i](context_embedding_HN)
                output = (output[:, :, :, None] * layer_w).sum(dim=-2, keepdim=False) + layer_b
                if i != (layer_num - 1):
                    output = F.relu(output)
        else:
            if self.model_args.PER_NODE_DECODER:
                output = (decoder_input[:, :, :, None] * self.decoder_weights[:, unimal_ids, :, :]).sum(dim=-2, keepdim=False) + self.decoder_bias[:, unimal_ids, :]
            else:
                output = self.decoder(decoder_input)

        # (batch_size, num_limbs, J)
        output = output.permute(1, 0, 2)
        # (batch_size, num_limbs * J)
        output = output.reshape(batch_size, -1)

        return output, attention_maps, dropout_mask, encoded_feature


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0., batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        if batch_first:
            self.pe = nn.Parameter(torch.randn(1, seq_len, d_model))
        else:
            # self.pe = nn.Parameter(torch.randn(seq_len, 1, d_model))
            self.pe = nn.Parameter(torch.zeros(seq_len, 1, d_model))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return x


class SWATPEEncoder(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pe_dim = [d_model // len(cfg.MODEL.TRANSFORMER.TRAVERSALS) for _ in cfg.MODEL.TRANSFORMER.TRAVERSALS]
        self.pe_dim[-1] = d_model - self.pe_dim[0] * (len(cfg.MODEL.TRANSFORMER.TRAVERSALS) - 1)
        print (self.pe_dim)
        self.swat_pe = nn.ModuleList([nn.Embedding(seq_len, dim) for dim in self.pe_dim])
        # self.compress_layer = nn.Linear(len(cfg.MODEL.TRANSFORMER.TRAVERSALS) * d_model, d_model)

    def forward(self, x, indexes):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        embeddings = []
        batch_size = x.size(1)
        for i in range(len(cfg.MODEL.TRANSFORMER.TRAVERSALS)):
            idx = indexes[:, :, i]
            pe = self.swat_pe[i](idx)
            embeddings.append(pe)
        embeddings = torch.cat(embeddings, dim=-1)
        # x = x + self.compress_layer(embeddings)
        x = x + embeddings
        return x


class MLPObsEncoder(nn.Module):
    """Encoder for env obs like hfield."""

    def __init__(self, obs_dim):
        super(MLPObsEncoder, self).__init__()
        mlp_dims = [obs_dim] + cfg.MODEL.TRANSFORMER.EXT_HIDDEN_DIMS
        self.encoder = tu.make_mlp_default(mlp_dims)
        self.obs_feat_dim = mlp_dims[-1]

    def forward(self, obs):
        return self.encoder(obs)


class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space):
        super(ActorCritic, self).__init__()
        self.seq_len = cfg.MODEL.MAX_LIMBS
        if cfg.MODEL.TYPE == 'transformer':
            self.v_net = TransformerModel(obs_space, 1)
        else:
            self.v_net = MLPModel(obs_space, cfg.MODEL.MAX_LIMBS)

        if cfg.ENV_NAME == "Unimal-v0":
            if cfg.MODEL.TYPE == 'transformer':
                self.mu_net = TransformerModel(obs_space, 2)    # In Transformer, all nodes share the same action decoder
            else:
                self.mu_net = MLPModel(obs_space, cfg.MODEL.MAX_LIMBS * 2)
            self.num_actions = cfg.MODEL.MAX_LIMBS * 2  # FT 12 * 2 = 24
        elif cfg.ENV_NAME == 'Modular-v0':
            if cfg.MODEL.TYPE == 'transformer':
                self.mu_net = TransformerModel(obs_space, 1)
            else:
                self.mu_net = MLPModel(obs_space, cfg.MODEL.MAX_LIMBS)
            self.num_actions = cfg.MODEL.MAX_LIMBS
        else:
            raise ValueError("Unsupported ENV_NAME")

        if cfg.MODEL.ACTION_STD_FIXED:  # fix the standard variance to 1 and only optimize the mean of Gaussian Distribution
            log_std = np.log(cfg.MODEL.ACTION_STD)
            self.log_std = nn.Parameter(
                log_std * torch.ones(1, self.num_actions), requires_grad=False,
            )
        else:
            self.log_std = nn.Parameter(torch.zeros(1, self.num_actions))

    def forward(self, obs, hi_encoded_feature, hi_act, hi_masks, act=None, return_attention=False, dropout_mask_v=None, dropout_mask_mu=None, unimal_ids=None, compute_val=True):
                
        if act is not None:
            batch_size = cfg.PPO.BATCH_SIZE
        else:
            batch_size = cfg.PPO.NUM_ENVS

        obs_env = {k: obs[k] for k in cfg.ENV.KEYS_TO_KEEP}
        if "obs_padding_cm_mask" in obs:    # False during collection
            obs_cm_mask = obs["obs_padding_cm_mask"]
        else:
            obs_cm_mask = None
        obs_dict = obs
        obs, obs_mask, act_mask, obs_context, edges = (
            obs["proprioceptive"],
            obs["obs_padding_mask"],
            obs["act_padding_mask"],
            obs["context"], 
            obs["edges"], 
        )

        morphology_info = {}
        if cfg.MODEL.TRANSFORMER.USE_SWAT_PE:
            # (batch_size, seq_len, traversal_num) ->(seq_len, batch_size, traversal_num)
            morphology_info['traversals'] = obs_dict['traversals'].permute(1, 0, 2).long()
        if cfg.MODEL.TRANSFORMER.USE_SWAT_RE:
            # (batch_size, seq_len, traversal_num) ->(seq_len, batch_size, traversal_num)
            morphology_info['SWAT_RE'] = obs_dict['SWAT_RE']
        
        if len(morphology_info.keys()) == 0:
            morphology_info = None

        obs_mask = obs_mask.bool()
        act_mask = act_mask.bool()

        # reshape the obs for transformer input
        if cfg.MODEL.TYPE == 'transformer':
            obs = obs.reshape(batch_size, self.seq_len, -1).permute(1, 0, 2)    # [B, 625] - > [B, N, 52] -> [N, B, 52] Why permutation? PyTorch Transformer batch_first=False
            obs_context = obs_context.reshape(batch_size, self.seq_len, -1).permute(1, 0, 2)

        # do not need to compute value function during evaluation to save time
        if compute_val:
            # Per limb critic values
            limb_vals, v_attention_maps, dropout_mask_v, _ = self.v_net(
                obs, hi_encoded_feature, hi_act, hi_masks, obs_mask, obs_env, obs_cm_mask, obs_context, morphology_info, 
                return_attention=return_attention, dropout_mask=dropout_mask_v, 
                unimal_ids=unimal_ids, 
            )
            # Zero out mask values
            limb_vals = limb_vals * (1 - obs_mask.int())
            # Use avg/max to keep the magnitidue same instead of sum
            num_limbs = self.seq_len - torch.sum(obs_mask.int(), dim=1, keepdim=True)
            val = torch.divide(torch.sum(limb_vals, dim=1, keepdim=True), num_limbs)    # [B, 1] All limbs share the same value
        else:
            val, v_attention_maps, dropout_mask_v = 0., None, 0.

        mu, mu_attention_maps, dropout_mask_mu, encoded_feature = self.mu_net(   # FT [B, N*2]
            obs, hi_encoded_feature, hi_act, hi_masks, obs_mask, obs_env, obs_cm_mask, obs_context, morphology_info, 
            return_attention=return_attention, dropout_mask=dropout_mask_mu, 
            unimal_ids=unimal_ids, 
        )
        std = torch.exp(self.log_std)
        pi = Normal(mu, std)

        if act is not None:
            logp = pi.log_prob(act)
            logp[act_mask] = 0.0
            self.limb_logp = logp
            logp = logp.sum(-1, keepdim=True)   # sum all the logp, corresponding to the sum value (classic multi-dim action)
            entropy = pi.entropy()
            entropy[act_mask] = 0.0
            entropy = entropy.mean()
            return val, pi, logp, entropy, dropout_mask_v, dropout_mask_mu, encoded_feature  # [B, 1], _, [B, N*2], [B, 1], 0.0, 0.0, [B, N, 128]
        else:
            if return_attention:
                return val, pi, v_attention_maps, mu_attention_maps, dropout_mask_v, dropout_mask_mu
            else:
                return val, pi, None, None, dropout_mask_v, dropout_mask_mu, encoded_feature


class Agent:
    def __init__(self, actor_critic):
        self.ac = actor_critic

    @torch.no_grad()
    def act(self, obs, hi_encoded_feature, hi_act, hi_masks, return_attention=False, dropout_mask_v=None, dropout_mask_mu=None, unimal_ids=None, compute_val=True):
        val, pi, v_attention_maps, mu_attention_maps, dropout_mask_v, dropout_mask_mu, encoded_feature = self.ac(obs, hi_encoded_feature, hi_act, hi_masks, return_attention=return_attention, dropout_mask_v=dropout_mask_v, dropout_mask_mu=dropout_mask_mu, unimal_ids=unimal_ids, compute_val=compute_val)
        self.pi = pi
        if not cfg.DETERMINISTIC:
            act = pi.sample()
        else:
            act = pi.loc
        logp = pi.log_prob(act)
        act_mask = obs["act_padding_mask"].bool()
        logp[act_mask] = 0.0
        logp = logp.sum(-1, keepdim=True)
        self.v_attention_maps = v_attention_maps
        self.mu_attention_maps = mu_attention_maps
        return val, act, logp, dropout_mask_v, dropout_mask_mu, encoded_feature

    @torch.no_grad()
    def get_value(self, obs, dropout_mask_v=None, dropout_mask_mu=None, unimal_ids=None):
        val, _, _, _, _, _, _ = self.ac(obs, dropout_mask_v=dropout_mask_v, dropout_mask_mu=dropout_mask_mu, unimal_ids=unimal_ids)
        return val


class TemporalTransformer(nn.Module):
    """
    Transformer to predict Morphology context and Dynamics context based on historial context emb and act
    """

    def __init__(self, encoded_feature_dim, act_dim) -> None:
        super().__init__()
        self.h_dim = cfg.TT.HDIM
        self.context_len = cfg.HI.MAX_LENGTH
        # embedding layer
        self.timestep_embed = nn.Embedding(self.context_len, self.h_dim)
        self.act_embed = nn.Linear(act_dim, self.h_dim)
        self.obs_embed = nn.Linear(encoded_feature_dim, self.h_dim)
        # transformer layer
        blocks = TransformerEncoderLayerResidual(
            self.h_dim,
            cfg.TT.NHEAD,
            cfg.TT.DIM_FEEDFORWARD,
            cfg.TT.DROPOUT,
            batch_first=True
        )
        self.transformer = TransformerEncoder(
            blocks, cfg.TT.NLAYERS, norm=None
        )
        # context predictor
        self.morph_predictor = nn.Linear(2 * self.context_len * self.h_dim, cfg.HI.MORPH_CONTEXT_DIM)
        self.dynamic_predictor = nn.Linear(2 * self.context_len * self.h_dim, cfg.HI.DYNAMIC_CONTEXT_DIM)

        self.register_buffer("timestep", torch.arange(self.context_len).unsqueeze(0))

        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        """
        Init weights of the network
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # module.weight.data.uniform_(math.sqrt(6.0/sum(module.weight.size())))
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, hi_encoded_feature: torch.Tensor, hi_act: torch.Tensor, hi_masks: torch.Tensor):
        """
        Compute contexts based on history info
        We don't care about the obs mask because all limbs share the same network
        Input:
            hi_encoded_feature: torch.Tensor (B, K, N, encoded_feature_dim)
            hi_act: torch.Tensor (B, K, N, act_dim)
            hi_masks: torch.Tensor (B, K, N, 1)
        Return:
            morph_context: torch.Tensor (N, B, morph_context_dim), context to be concatenated with obs embedding indicating the morphology
            dynamic_context: torch.Tensor (N, B, dynamic_context_dim), context to be concatenated with decoder input indicating the dynamic of MDP
        """
        B, K, N = hi_encoded_feature.shape[:-1]
        assert K == self.context_len, "K should be equal to context length"

        # permute to fit with PyTorch Network settings
        hi_encoded_feature = hi_encoded_feature.permute(2, 0, 1, 3)   # (N, B, K(length), encoded_feature_dim)
        hi_act = hi_act.permute(2, 0, 1, 3)
        hi_masks = hi_masks.permute(2, 0, 1, 3).reshape(N*B, K, 1).repeat(1, 1, 2).reshape(N*B, 2*K)    # 2*K because of two modalities

        # embedding
        timestep_embedding = self.timestep_embed(self.timestep) # -> (1, K, h_dim)
        obs_embedding = self.obs_embed(hi_encoded_feature) + timestep_embedding   # -> (N, B, K, h_dim)
        act_embedding = self.act_embed(hi_act) + timestep_embedding  # -> (N, B, K, h_dim)

        # stack -> (N*B, 2*K, h_dim)
        # (s_1,a_1,s_2,a_2,...,s_K,a_K)
        h = torch.stack([obs_embedding, act_embedding], dim=2).permute(0, 1, 3, 2, 4).reshape(N*B, 2*K, self.h_dim)

        # transformer
        h = self.transformer(
            h, src_key_padding_mask=hi_masks
        )   # (N*B, 2*K, h_dim)

        # context predict
        h = h.reshape(N, B, -1)
        morph_context = self.morph_predictor(h) # -> (N, B, morph_dim)
        dynamic_context = self.dynamic_predictor(h) # -> (N, B, dynamic_dim)

        return morph_context, dynamic_context
    

# for test
if __name__ == "__main__":
    from tools.train_ppo import parse_args
    from pprint import pprint

    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    hi_encoded_feature, act_dim = 128, 2

    # model initialization
    tt = TemporalTransformer(hi_encoded_feature, act_dim)
    pprint(tt)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    tt.to(device)

    # fake data manufication
    B, K, N = 32, 16, 12
    hi_encoded_feature = torch.randn((B, K, N, hi_encoded_feature)).cuda()
    hi_act = torch.randn((B, K, N, act_dim)).cuda()
    hi_masks = torch.randint(0, 2, (B, K, N, 1)).to(torch.float).cuda()

    # forward
    morph_context, dynamic_context = tt(hi_encoded_feature, hi_act, hi_masks)

    pprint(morph_context.shape)
    pprint(dynamic_context.shape)
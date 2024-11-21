import re
import numpy as np

import torch.nn as nn
import torch
import torch.distributions as D
import torch.nn.functional as F
from agent.utils import *
import omegaconf

from collections import defaultdict 

# We thank the authors of the repo: https://github.com/jsikyoon/dreamer-torch
# For their open source re-implementation, which was used as a reference to develop our code faster

Module = nn.Module

def static_scan_for_lambda_return(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    indices = reversed(indices)
    flag = True
    for index in indices:
        inp = lambda x: (_input[x].unsqueeze(0) for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            outputs = last
            flag = False
        else:
            outputs = torch.cat([last, outputs], dim=0)
    return outputs

def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    # assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward, device=reward.device)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1 :]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1], device=reward.device)
    if len(bootstrap.shape) < len(value.shape):
        bootstrap = bootstrap[None]
    next_values = torch.cat([value[1:], bootstrap], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    returns = static_scan_for_lambda_return(
        lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
        (inputs, pcont),
        bootstrap,
    )
    if axis != 0:
        returns = returns.permute(dims)
    return returns

def static_scan(fn, inputs, start, reverse=False, unpack=False):
    last = start
    indices = range(inputs[0].shape[0])
    flag = True
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)
        if unpack:
            last = fn(last, inputs[0][index])
        else:
            last = fn(
                last, inp(index)
            )  # NOTE: I removed an unpacking(*) before inp
        if flag:
            if type(last) == type({}):
                outputs = {
                    key: value.clone().unsqueeze(0)
                    for key, value in last.items()
                }
            else:
                outputs = []
                for _last in last:
                    if type(_last) == type({}):
                        outputs.append(
                            {
                                key: value.clone().unsqueeze(0)
                                for key, value in _last.items()
                            }
                        )
                    else:
                        outputs.append(_last.clone().unsqueeze(0))
            flag = False
        else:
            if type(last) == type({}):
                for key in last.keys():
                    outputs[key] = torch.cat(
                        [outputs[key], last[key].unsqueeze(0)], dim=0
                    )
            else:
                for j in range(len(outputs)):
                    if type(last[j]) == type({}):
                        for key in last[j].keys():
                            outputs[j][key] = torch.cat(
                                [outputs[j][key], last[j][key].unsqueeze(0)],
                                dim=0,
                            )
                    else:
                        outputs[j] = torch.cat(
                            [outputs[j], last[j].unsqueeze(0)], dim=0
                        )
    if type(last) == type({}):
        outputs = [outputs]
    return outputs

class EnsembleRSSM(Module):
    def __init__(
        self,
        ensemble=5,
        stoch=30,
        deter=200,
        hidden=200,
        discrete=False,
        act=nn.ELU,
        norm="none",
        std_act="softplus",
        min_std=0.1,
        action_dim=None,
        embed_dim=1536,
        device="cuda",
        full_posterior=True
    ):
        super().__init__()
        assert action_dim is not None
        self.device = device
        self._embed_dim = embed_dim
        self._action_dim = action_dim
        self._ensemble = ensemble
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = None if discrete in ["none", "None", None] else discrete
        self._act = act()
        self._norm = norm
        self._std_act = std_act
        self._min_std = min_std
        self._full_posterior = full_posterior
        
        self._cell = GRUCell(
            self._hidden, self._deter, norm=True, device=self.device
        )

        if self._discrete:
            inp_dim = stoch * discrete + action_dim
        else:
            inp_dim = stoch + action_dim
        self._img_in = nn.Sequential(
            nn.Linear(inp_dim, hidden), NormLayer(norm, hidden)
        )

        self._ensemble_img_out = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(deter, hidden), NormLayer(norm, hidden)
                )
                for _ in range(ensemble)
            ]
        )
        if self._discrete:
            self._ensemble_img_dist = nn.ModuleList(
                [nn.Linear(hidden, stoch * discrete) for _ in range(ensemble)]
            )
            self._obs_dist = nn.Linear(hidden, stoch * discrete)
        else:
            self._ensemble_img_dist = nn.ModuleList(
                [nn.Linear(hidden, 2 * stoch) for _ in range(ensemble)]
            )
            self._obs_dist = nn.Linear(hidden, 2 * stoch)

        if self._full_posterior:
            self._obs_out = nn.Sequential(nn.Linear(deter + embed_dim, hidden), NormLayer(norm, hidden))
        else:
            self._obs_out = nn.Sequential(nn.Linear(embed_dim, hidden), NormLayer(norm, hidden))

    def initial(self, batch_size):
        if self._discrete:
            state = dict(
                logit=torch.zeros(
                    [batch_size, self._stoch, self._discrete],
                    device=self.device,
                ),
                stoch=torch.zeros(
                    [batch_size, self._stoch, self._discrete],
                    device=self.device,
                ),
                deter=self._cell.get_initial_state(None, batch_size),
            )
        else:
            state = dict(
                mean=torch.zeros(
                    [batch_size, self._stoch], device=self.device
                ),
                std=torch.zeros([batch_size, self._stoch], device=self.device),
                stoch=torch.zeros(
                    [batch_size, self._stoch], device=self.device
                ),
                deter=self._cell.get_initial_state(None, batch_size),
            )
        return state

    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])

        if self._full_posterior:
            post, prior = static_scan(
                lambda prev, inputs: self.obs_step(prev[0], *inputs),
                (swap(action), swap(embed), swap(is_first)), (state, state))
        else:
            post = self.obs_posterior_all(swap(embed))
            prior = self.obs_prior_all(state, swap(action), post['stoch'], swap(is_first))
            post['deter'] = prior['deter']
            
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior
    
    def obs_prior_all(self, init_state, actions, stochs, is_first, should_sample=True):
        priors = defaultdict(list)

        prev_state = init_state
        for prev_action, post_stoch, this_is_first in zip(actions, stochs, is_first):
            # Deal with firsts
            prev_state = { k: torch.einsum('b,b...->b...', 1.0 - this_is_first.float(), x) for k, x in prev_state.items()}
            prev_action = torch.einsum('b,b...->b...', 1.0 - this_is_first.float(), prev_action)
            # 
            prior = self.img_step(prev_state, prev_action, should_sample)
            for k,v in prior.items():
                priors[k].append(v)
            prev_state = prior
            prev_state['stoch'] = post_stoch
        
        for k in priors:
            priors[k] = torch.stack(priors[k], dim=0)
        return priors
    
    def obs_posterior_all(self, embed, should_sample=True):
        x = self._obs_out(embed)
        x = self._act(x)
        stats = self._suff_stats_layer('_obs_dist', x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if should_sample else None 
        post = {'stoch': stoch, **stats}
        return post
    
    def imagine(self, action, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        assert isinstance(state, dict), state
        action = swap(action)
        prior = static_scan(self.img_step, [action], state, unpack=True)[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state, ensemble=False):
        if ensemble:
            state = self._suff_stats_ensemble(state["deter"])
        if self._discrete:
            logit = state["logit"]
            dist = D.Independent(OneHotDist(logit), 1)
        else:
            mean, std = state['mean'], state['std']
            dist = D.Independent(D.Normal(mean, std), 1)
            dist.sample = dist.rsample
        return dist

    def obs_step(
        self, prev_state, prev_action, embed, is_first, should_sample=True
    ):
        prev_state = {
            k: torch.einsum("b,b...->b...", 1.0 - is_first.float(), x)
            for k, x in prev_state.items()
        }
        prev_action = torch.einsum(
            "b,b...->b...", 1.0 - is_first.float(), prev_action
        )
        #
        prior = self.img_step(prev_state, prev_action, should_sample)
        if self._full_posterior:
            x = torch.cat([prior['deter'], embed], -1)
        else:
            x = embed
        x = self._obs_out(x)
        x = self._act(x)
        stats = self._suff_stats_layer("_obs_dist", x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if should_sample else None
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [
                self._stoch * self._discrete
            ]
            prev_stoch = prev_stoch.reshape(shape)
        x = torch.cat([prev_stoch, prev_action], -1)
        x = self._img_in(x)
        x = self._act(x)
        deter = prev_state["deter"]
        x, deter = self._cell(x, [deter])
        deter = deter[0]  # It's wrapped in a list.
        stats = self._suff_stats_ensemble(x)
        index = torch.randint(0, self._ensemble, ())
        stats = {k: v[index] for k, v in stats.items()}
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else None
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def _suff_stats_ensemble(self, inp):
        bs = list(inp.shape[:-1])
        inp = inp.reshape([-1, inp.shape[-1]])
        stats = []
        for k in range(self._ensemble):
            x = self._ensemble_img_out[k](inp)
            x = self._act(x)
            stats.append(self._suff_stats_layer("_ensemble_img_dist", x, k=k))
        stats = {
            k: torch.stack([x[k] for x in stats], 0)
            for k, v in stats[0].items()
        }
        stats = {
            k: v.reshape([v.shape[0]] + bs + list(v.shape[2:]))
            for k, v in stats.items()
        }
        return stats

    def _suff_stats_layer(self, name, x, k=None):
        layer = getattr(self, name)
        if k is not None:
            layer = layer[k]
        x = layer(x)
        if self._discrete:
            logit = x.reshape(
                list(x.shape[:-1]) + [self._stoch, self._discrete]
            )
            return {"logit": logit}
        else:
            mean, std = torch.chunk(x, 2, -1)
            std = {
                "softplus": lambda: F.softplus(std),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, forward, balance, free, free_avg):
        kld = D.kl_divergence
        sg = lambda x: {k: v.detach() for k, v in x.items()}
        lhs, rhs = (prior, post) if forward else (post, prior)        

        # mixing of variable to balance the loss like in dreamer v3
        mix = balance if isinstance(balance, (np.ndarray, list, omegaconf.listconfig.ListConfig)) else [balance, 1 - balance]
        mix = mix if forward else mix[::-1]
        
        dtype = post["stoch"].dtype
        device = post["stoch"].device
        free_tensor = torch.tensor([free], dtype=dtype, device=device)
        if balance == 0.5:
            value = kld(self.get_dist(lhs), self.get_dist(rhs))
            loss = torch.maximum(value, free_tensor).mean()
        else:
            value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
            value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
            if free_avg:
                loss_lhs = torch.maximum(value_lhs.mean(), free_tensor)
                loss_rhs = torch.maximum(value_rhs.mean(), free_tensor)
            else:
                loss_lhs = torch.maximum(value_lhs, free_tensor).mean()
                loss_rhs = torch.maximum(value_rhs, free_tensor).mean()
            loss = mix[0] * loss_lhs + mix[1] * loss_rhs
        return loss, value

class Encoder(Module):
    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        act="ELU",
        norm="none",
        cnn_depth=48,
        cnn_kernels=(4, 4, 4, 4),
        mlp_layers=4,
        mlp_units=400, 
        coordConv=False,
        symlog_inputs=False,
    ):
        super().__init__()
        self.shapes = shapes
        self.cnn_keys = [k for k, v in shapes.items() if re.match(cnn_keys, k)]
        self.mlp_keys = [k for k, v in shapes.items() if re.match(mlp_keys, k)]

        print("Encoder CNN inputs:", list(self.cnn_keys))
        print("Encoder MLP inputs:", list(self.mlp_keys))
        self._act = getattr(nn, act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers
        self._mlp_units = mlp_units
        Conv = CoordConv2d if coordConv else nn.Conv2d
        
        # CNN layers
        if len(self.cnn_keys) > 0:
            self._conv_model = []
            for i, kernel in enumerate(self._cnn_kernels):
                if i == 0:
                    prev_depth = sum([v[0] for k, v in shapes.items() if k in self.cnn_keys]) # intialize depth with sum of dimensions of considered observations
                else:
                    prev_depth = 2 ** (i - 1) * self._cnn_depth
                depth = 2**i * self._cnn_depth
                self._conv_model.append(
                    Conv(prev_depth, depth, kernel, stride=2)
                )
                self._conv_model.append(NormLayer("none", depth))
                self._conv_model.append(self._act())
            self._conv_model = nn.Sequential(*self._conv_model)
       
        # MLP layers
        if len(self.mlp_keys) > 0:
            self._mlp_in_shape = np.sum([np.prod(shapes[k]) for k in self.mlp_keys])
            dist_cfg = {"dist": "none"}
            self._mlp_model = MLP(self._mlp_in_shape, None, self._mlp_layers, self._mlp_units, self._act, self._norm, symlog_inputs=symlog_inputs, **dist_cfg)

    def forward(self, data):
        key, shape = list(self.shapes.items())[0]
        
        batch_dims = data[key].shape[: -len(shape)]
        data = {
            k: v.reshape((-1,) + tuple(v.shape)[len(batch_dims) :])
            for k, v in data.items()
        }
        outputs = []
        if self.cnn_keys:
            outputs.append(self._cnn({k: data[k] for k in self.cnn_keys}))
        if self.mlp_keys:
            # flattening of objects pos into a one dimensional tensor
            if "objects_pos" in self.mlp_keys: 
                data["objects_pos"] = torch.flatten(data["objects_pos"], start_dim=1)
            outputs.append(self._mlp({k: data[k] for k in self.mlp_keys}))
        output = torch.cat(outputs, -1)
        return output.reshape(batch_dims + output.shape[1:])

    def _cnn(self, data):
        x = torch.cat(list(data.values()), 1)
        x = self._conv_model(x)
        return x.reshape(tuple(x.shape[:-3]) + (-1,))

    def _mlp(self, data):
        x = torch.cat(list(data.values()), -1)
        x = self._mlp_model(x)
        return x

class Decoder(Module):
    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        act="ELU",
        norm="none",
        cnn_depth=48,
        cnn_kernels=(4, 4, 4, 4),
        mlp_layers=4,
        mlp_units=400,
        embed_dim=1024,
        symlog_outputs=False,
    ):
        super().__init__()
        self._embed_dim = embed_dim
        self._shapes = shapes
        self.cnn_keys = [
            k
            for k, v in shapes.items()
            if re.match(cnn_keys, k)
        ]
        self.mlp_keys = [
            k
            for k, v in shapes.items()
            if re.match(mlp_keys, k)
        ]
        print("Decoder CNN outputs:", list(self.cnn_keys))
        print("Decoder MLP outputs:", list(self.mlp_keys))

        self._act = getattr(nn, act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers
        self._mlp_units = mlp_units
        self.channels = {k: self._shapes[k][0] for k in self.cnn_keys}

        if len(self.cnn_keys) > 0:

            self._conv_in = nn.Sequential(
                nn.Linear(embed_dim, 32 * self._cnn_depth)
            )
            self._conv_model = []
            for i, kernel in enumerate(self._cnn_kernels):
                if i == 0:
                    prev_depth = 32 * self._cnn_depth
                    self.layer_size = 1
                else:
                    prev_depth = (
                        2 ** (len(self._cnn_kernels) - (i - 1) - 2)
                        * self._cnn_depth
                    )
                depth = 2 ** (len(self._cnn_kernels) - i - 2) * self._cnn_depth
                act, norm = self._act, self._norm
                if i == len(self._cnn_kernels) - 1:
                    depth, act, norm = (
                        sum(self.channels.values()),
                        nn.Identity,
                        "none",
                    )

                self._conv_model.append(
                    nn.ConvTranspose2d(prev_depth, depth, kernel, stride=2)
                )

                self._conv_model.append(NormLayer("none", depth))
                self._conv_model.append(act())

            self._conv_model = nn.Sequential(*self._conv_model)

        # MLP layers
        if len(self.mlp_keys) > 0:
            dist_cfg = {"dist": "none"} 
            self._mlp_model = MLP(embed_dim, None, self._mlp_layers, self._mlp_units, self._act, self._norm, symlog_inputs=symlog_outputs, **dist_cfg)
            
            # get distrbutions out of the MLP to divide for the different keys
            for key, shape in {k: shapes[k] for k in self.mlp_keys}.items():
                dist = "symlog_mse" if symlog_outputs else "mse"
                self.add_module(f"dense_{key}", DistLayer(self._mlp_units, shape, dist=dist))

    def forward(self, features, only_mlp=False, **kwargs):
        outputs = {}
        if self.cnn_keys and not only_mlp:
            outputs.update(self._cnn(features))
        if self.mlp_keys:
            outputs.update(self._mlp(features))
        return outputs

    def _cnn(self, features):
        x = self._conv_in(features)
        x = x.reshape(
            [
                -1,
                32 * self._cnn_depth,
                1,
                1,
            ]
        )
        x = self._conv_model(x)
        x = x.reshape(list(features.shape[:-1]) + list(x.shape[1:]))
        means = torch.split(x, list(self.channels.values()), 2)

        dists = {}
        for (key, shape), mean in zip(self.channels.items(), means):
            if key == "segmentation":
                mean = mean.permute(0, 1, 3, 4, 2)
                dists[key] = D.Independent(OneHotDist(mean), 2)
            else:
                dists[key] = D.Independent(D.Normal(mean, 1.0), 3)

        return dists

    def _mlp(self, features):
        shapes = {k: self._shapes[k] for k in self.mlp_keys}
        x = features
        x = self._mlp_model(x)
        dists = {}
        for key, shape in shapes.items():
            lin = getattr(self, f"dense_{key}")
            dists[key] = lin(x)
        return dists

class ObjEncoder(Module):
    def __init__(self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        act="ELU",
        norm="none",
        cnn_depth=48,
        cnn_kernels=(4, 4, 4, 4),
        mlp_layers=4,
        mlp_units=400,
        distance_mode="mse",
        symlog_inputs=False,
        dist="mse",
        objlatent_ratio=1,
    ):
        super().__init__()
        self._shapes = {**shapes}
        self._shapes["segmentation"] = (1, *self._shapes["segmentation"][1:])
        self.cnn_keys = [
            k
            for k, v in shapes.items()
            if re.match(cnn_keys, k)
        ]
        self.mlp_keys = [
            k
            for k, v in shapes.items()
            if re.match(mlp_keys, k)
        ]
        print("Object Encoder CNN inputs:", list(self.cnn_keys))
        print("Object Encoder MLP inputs:", list(self.mlp_keys))

        self._act = getattr(nn, act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers
        self._mlp_units = mlp_units

        self.instances_dim = (
            shapes["segmentation"][0]  # this didn't got modified above
        )

        self.objects_dim = self.instances_dim - 1

        self.channels = {k: self._shapes[k][0] for k in self.cnn_keys}

        self.distance_mode = distance_mode
        
        if len(self.cnn_keys) > 0:
            self._conv_model = []
            for i, kernel in enumerate(self._cnn_kernels):
                if i == 0:
                    prev_depth = sum([v[0] for k, v in shapes.items() if k in self.cnn_keys]) # intialize depth with sum of dimensions of considered observations
                else:
                    prev_depth = 2 ** (i - 1) * self._cnn_depth
                depth = 2**i * self._cnn_depth
                self._conv_model.append(
                    nn.Conv2d(prev_depth, depth, kernel, stride=2)
                )
                self._conv_model.append(NormLayer(norm, depth))
                self._conv_model.append(self._act())    
            self._conv_model = nn.Sequential(*self._conv_model)
            
        if len(self.mlp_keys) > 0:
            self._mlp_in_shape = self._shapes[self.mlp_keys[0]][1] + self.instances_dim
            dist_cfg = {"dist": dist} #TODO generalize to output different distributions
            self._mlp_model = MLP(self._mlp_in_shape, int(32 * self._cnn_depth * objlatent_ratio), self._mlp_layers, self._mlp_units, self._act, self._norm, symlog_inputs=symlog_inputs, **dist_cfg)

            # self._mlp_model.add_module(f"multivariate_normal_dist", (MultivariateNormal(self._mlp_units, 32 * self._cnn_depth, dist_mode=obj_latent_as_dist)))
             
    def forward(self, poses):
        outputs = {}        
        obj_onehot = torch.eye(
            self.instances_dim, device=poses.device
        ).repeat(
            *poses.shape[:2], 1, 1
        )  # last dim is obj idx

        if self.cnn_keys:
            raise NotImplementedError
        if self.mlp_keys and poses != None:
            outputs["prior"] = self._mlp(poses, obj_onehot) 
        return outputs

    def _cnn(self, features, obj_onehot, masks):
        raise NotImplementedError

    def _mlp(self, poses, obj_onehot):
        input = []
        # concatenate corresponding (index i) one-hot encoding of instance to the full embedding
        obj_feat = obj_onehot[..., 0, :]
        input.append(torch.cat((poses[..., 0, :], obj_feat), dim=-1))

        input = torch.stack(input, dim=2)
        prior = self._mlp_model.compose_output(self._mlp_model(input))
    
        return prior

class ObjDecoder(Module):
    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        act="ELU",
        norm="none",
        cnn_depth=48,
        cnn_kernels=(4, 4, 4, 4),
        mlp_layers=4,
        mlp_units=400,
        embed_dim=1024,
        symlog_outputs=False,
        obj_extractor_cfg=None
    ):
        super().__init__()
        self._embed_dim = embed_dim
        self._shapes = {**shapes}
        self._shapes["segmentation"] = (1, *self._shapes["segmentation"][1:])
        self.cnn_keys = [
            k
            for k, v in shapes.items()
            if re.match(cnn_keys, k)
        ]
        self.mlp_keys = [
            k
            for k, v in shapes.items()
            if re.match(mlp_keys, k)
        ]
        print("Decoder CNN outputs:", list(self.cnn_keys))
        print("Decoder MLP outputs:", list(self.mlp_keys))

        self._act = getattr(nn, act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers
        self._mlp_units = mlp_units

        self.instances_dim = (
            shapes["segmentation"][0]  # this didn't got modified above
            if len(self.cnn_keys) > 0
            else shapes[self.mlp_keys[0]][0] + 1
        )

        self.objects_dim = self.instances_dim - 1

        self.channels = {k: self._shapes[k][0] for k in self.cnn_keys}
        
        # tilling tensors
        self.tile_tensors = {}
        for key in self.cnn_keys:
            init_dim = self.instances_dim
            n_tile = self._shapes[key][0]
            self.tile_tensors[key]  = torch.cat([init_dim * torch.arange(n_tile) + i for i in range(init_dim)]).cuda()
            
        if obj_extractor_cfg:
            starting_width = embed_dim + self.instances_dim
            out_shape = 32 * self._cnn_depth
            self._object_extractor = MLP(starting_width, out_shape, **obj_extractor_cfg)
            
        if len(self.cnn_keys) > 0:
            self._conv_model = []
            for i, kernel in enumerate(self._cnn_kernels):
                if i == 0:
                    prev_depth = 32 * self._cnn_depth  # * self.instances_dim
                else:
                    prev_depth = (
                        2 ** (len(self._cnn_kernels) - (i - 1) - 2)
                        * self._cnn_depth
                    )
                depth = 2 ** (len(self._cnn_kernels) - i - 2) * self._cnn_depth
                act, norm = self._act, self._norm
                if i == len(self._cnn_kernels) - 1:
                    depth, act, norm = (
                        sum(self.channels.values()),
                        nn.Identity,
                        "none",
                    )

                self._conv_model.append(
                    nn.ConvTranspose2d(prev_depth, depth, kernel, stride=2)
                )

                self._conv_model.append(NormLayer(norm, depth))
                self._conv_model.append(act())

            self._conv_model = nn.Sequential(*self._conv_model)

        if len(self.mlp_keys) > 0:
            self._mlp_in_shape = 32 * self._cnn_depth
            dist_cfg = {"dist": "none"}
            self._mlp_model = MLP(self._mlp_in_shape, None, self._mlp_layers, self._mlp_units, self._act, self._norm, **dist_cfg)
            
            # get distrbutions out of the MLP to divide for the different keys
            for key, shape in {k: shapes[k] for k in self.mlp_keys}.items():
                dist = "symlog_mse" if symlog_outputs else "mse" # apply symlog_mse in case the input was modified
                self.add_module(f"dense_{key}", DistLayer(self._mlp_units, shape[1], dist=dist))     

    def forward(self, features, masks=None, only_mlp=False, **kwargs):
        outputs = {}
        obj_onehot = torch.eye(
            self.instances_dim, device=features.device
        ).repeat(
            *features.shape[:2], 1, 1
        )  # last dim is obj idx

        if self.cnn_keys and not only_mlp:
            dist = self._cnn(features, obj_onehot, masks)
            outputs.update(dist)
        if self.mlp_keys:
            outputs.update(self._mlp(features, obj_onehot))
        return outputs
    
    def object_latent_extractor(self, features):
        outputs = {}
        obj_onehot = torch.eye(
            self.instances_dim, device=features.device
        ).repeat(
            *features.shape[:2], 1, 1
        )  # last dim is obj idx
        outputs["post"], _ = self._object_latent_extractor(features, obj_onehot)
        return outputs
    
    def tile(self, a, dim, n_tile, key):
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = self.tile_tensors[key]

        return torch.index_select(a, dim, order_index)

    def _cnn(self, features, obj_onehot, masks):

        dists = {}

        # x is extracted feat, feat is = (embed, obj_onehot)
        dists["post"], _ = self._object_latent_extractor(features, obj_onehot)

        x = dists["post"]["sample"].reshape(
            [
                -1,  # batch_dim x batch_time x num_obj
                32 * self._cnn_depth,  # object_extractor output dimension
                1,
                1,
            ]
        )
        
        x = self._conv_model(x)
        x = x.reshape(
            *features.shape[:2],
            self.instances_dim,
            sum(self.channels.values()),
            *x.shape[-2:],
        )
        means = torch.split(x, list(self.channels.values()), dim=-3)


        for key, mean in zip(self.channels.keys(), means):
            if key == "segmentation":
                mean = mean.reshape(
                    *features.shape[:2], self.instances_dim, *x.shape[-2:]
                ).permute(0, 1, 3, 4, 2)

                dists[key] = D.Independent(
                    OneHotDist(mean), 2
                )  # output is a binary mask
            else:
                mean = torch.reshape(mean,
                    (*features.shape[:2],
                    self.instances_dim * self.channels[key],
                    *x.shape[-2:])
                )
                if masks != None:
                    ch = int(mean.shape[2] / self.instances_dim)
                    mask = self.tile(masks, 2, ch, key)
                    mean = (
                        mean * mask
                    )  # mask means to avoid reconstruction in pixels out of objects
                
                mean = torch.reshape(mean,
                    (*mean.shape[:2],
                    self.instances_dim,
                    self.channels[key],
                    *mean.shape[-2:])
                )
                
                dists[key] = D.Independent(D.Normal(mean, torch.tensor(1.0, device="cuda:0", dtype=torch.float32)), 3) #TODO cause 30ms slow down everytime it is called
        return dists

    def _object_latent_extractor(self, features, obj_onehot, instances=None):
        instances = self.instances_dim if instances == None else instances

        # TODO(pmazzagl) : batchify this loop
        feat = []
        for i in range(instances):
            # concatenate corresponding (index i) one-hot encoding of instance to the full embedding
            obj_feat = obj_onehot[..., i, :]
            feat.append(torch.cat((features, obj_feat), dim=-1))
        
        feat = torch.stack(feat, dim=2)

        extracted_feat = self._object_extractor.compose_output(self._object_extractor(feat))

        return extracted_feat, feat

    def _mlp(self, features, obj_onehot):

        dists = {}
        shapes = {k: self._shapes[k] for k in self.mlp_keys}

        x, _ = self._object_latent_extractor(
            features, obj_onehot, self.objects_dim
        )

        x = self._mlp_model(x["sample"])

        for key, shape in shapes.items():
            lin = getattr(self, f"dense_{key}")
            dists[key] = lin(x)

        return dists    

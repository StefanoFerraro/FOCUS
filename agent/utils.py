import numpy as np
import math

import torch.nn as nn
import torch
import torch.distributions as D
import torch.nn.functional as F
from torch.distributions.utils import _standard_normal

# Collection of utility functions for the agents definition

Module = nn.Module

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
            
def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

class Optimizer:
    def __init__(
        self,
        name,
        parameters,
        lr,
        eps=1e-4,
        clip=None,
        wd=None,
        opt="adam",
        wd_pattern=r".*",
        use_amp=False,
    ):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            "adam": lambda: torch.optim.Adam(parameters, lr, eps=eps),
            "nadam": lambda: torch.optim.Nadam(parameters, lr, eps=eps),
            "adamax": lambda: torch.optim.Adamax(parameters, lr, eps=eps),
            "sgd": lambda: torch.optim.SGD(parameters, lr),
            "momentum": lambda: torch.optim.SGD(lr, momentum=0.9),
        }[opt]()
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self._once = True

    def __call__(self, loss, params):
        params = list(params)
        assert len(loss.shape) == 0 or (
            len(loss.shape) == 1 and loss.shape[0] == 1
        ), (self._name, loss.shape)
        metrics = {}

        # Count parameters.
        if self._once:
            count = sum(p.numel() for p in params if p.requires_grad)
            print(f"Found {count} {self._name} parameters.")
            self._once = False

        # Check loss.
        metrics[f"{self._name}_loss"] = loss.detach().cpu().numpy()

        # Compute scaled gradient.
        self._scaler.scale(loss).backward()
        self._scaler.unscale_(self._opt)

        # Gradient clipping.
        if self._clip:
            norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
            metrics[f"{self._name}_grad_norm"] = norm.item()

        # Weight decay.
        if self._wd:
            self._apply_weight_decay(params)

        # # Apply gradients.
        self._scaler.step(self._opt)
        self._scaler.update()

        self._opt.zero_grad()
        return metrics

    def _apply_weight_decay(self, varibs):
        nontrivial = self._wd_pattern != r".*"
        if nontrivial:
            raise NotImplementedError("Non trivial weight decay")
        else:
            for var in varibs:
                var.data = (1 - self._wd) * var.data

class Interpolate(Module):
    def __init__(self, size, mode):
        Module.__init__(self)
        self.size = size
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, size=self.size, mode=self.mode)

### DISTRIBUTIONS ###
class SampleDist:
    def __init__(self, dist: D.Distribution, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    @property
    def mean(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = (
            torch.argmax(logprob, dim=0)
            .reshape(1, batch_size, 1)
            .expand(1, batch_size, feature_size)
        )
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self, num_samples=1):
        if num_samples==1:
            return self._dist.rsample()
        dist = dist.expand((num_samples, *self._dist.batch_shape))
        sample = torch.mean(self._dist.rsample())
        return sample 

class OneHotDist(D.OneHotCategorical):
    def __init__(self, logits=None, probs=None):
        super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()  # ST-gradients
        return sample


class BernoulliDist(D.Bernoulli):
    def __init__(self, logits=None, probs=None):
        super().__init__(logits=logits, probs=probs)

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()  # ST-gradients
        return sample

class TruncatedNormal(D.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, sample_shape=torch.Size(), clip=None):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

class TanhTransform(D.transforms.Transform):
    domain = D.constraints.real
    codomain = D.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))

class SquashedNormal(D.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = D.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

class DiscDist:
# We thanks the authors of the original implementation https://github.com/NM512/dreamerv3-torch/blob/2c7a81a0e2f5f0c7659ba73b0ddbedf2a7e2ecf4/tools.py
    def __init__(
        self,
        logits,
        low=-20.0,
        high=20.0,
        transfwd=symlog,
        transbwd=symexp,
        device="cuda",
    ):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255).to(device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd
        
        self.mean = self.mode = self.transbwd(torch.sum(self.probs * self.buckets, dim=-1, keepdim=True))

    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    def mode(self):
        _mode = self.probs * self.buckets
        return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        # this is implemented using clip at the original repo as the gradients are not backpropagated for the out of limits.
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)
    
class SymlogDist:
# We thanks the authors of the original implementation https://github.com/NM512/dreamerv3-torch/blob/2c7a81a0e2f5f0c7659ba73b0ddbedf2a7e2ecf4/tools.py
    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self.mean = symexp(self._mode)
        self.mode = symexp(self._mode)
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def log_prob(self, value):
        assert self._mode.shape == value.shape
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:]) # assume that the batch dimensions are the first 2
        elif self._agg == "sum":

            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss

class DistLosses(Module):      
    @staticmethod
    def kl_loss(post, prior, balance):
        def _get_dist(state):
            dist = D.Normal(state["mean"], state["std"])
            dist =  D.Independent(dist, 1)
            return dist 

        if 'std' not in post.keys() and 'std' not in prior.keys():
            raise ValueError("Distance mode kl is available with distribution mode") 
        kld = D.kl_divergence
        sg = lambda x: {k: v.detach() for k, v in x.items()}
        lhs, rhs = post, prior
        dtype = post["mean"].dtype
        device = post["mean"].device
        free_tensor = torch.tensor([1], dtype=dtype, device=device)

        value_lhs = kld(_get_dist(lhs), _get_dist(sg(rhs)))
        value_rhs = kld(_get_dist(sg(lhs)), _get_dist(rhs))
        loss_lhs = torch.maximum(value_lhs.mean(), free_tensor)
        loss_rhs = torch.maximum(value_rhs.mean(), free_tensor)
        loss = balance * loss_rhs + (1 - balance) * loss_lhs
        return loss
    
    @staticmethod 
    def mse_loss(post, prior, balance):
        if 'std' in post.keys() and 'std' in prior.keys():
            raise ValueError("Distance mode mse is not available with distribution mode") 

        post = post["mean"]
        prior = prior["mean"]
        
        prior_loss = torch.sum(
            ((prior - post.detach()) ** 2), dim=-1
        ).mean()
        post_loss = torch.sum(((prior.detach() - post) ** 2), dim=-1
        ).mean()
        
        loss = balance * prior_loss + (1 - balance) * post_loss
        return loss
    
    @staticmethod 
    def cosine_loss(post, prior, balance):
        if 'std' in post.keys() and 'std' in prior.keys():
            raise ValueError("Distance mode max_cosine is not available with distribution mode")         
        post = post["mean"]
        prior = prior["mean"]
        
        # cosine similarity dot(post, prior)/(norm(post)*norm(prior))
        prior_loss = torch.einsum("ijkl,ijkl->ijk", (post.detach(), prior)) / (torch.norm(post.detach(), dim=-1) * torch.norm(prior, dim=-1) + 1e-8) # be sure that we are never dividing by 0
        post_loss = torch.einsum("ijkl,ijkl->ijk", (post, prior.detach())) / (torch.norm(post, dim=-1) * torch.norm(prior.detach(), dim=-1) + 1e-8 )# be sure that we are never dividing by 0
        
        loss = balance * prior_loss.mean() + (1 - balance) * post_loss.mean()
        
        return -loss # 1 = max similarity | -1 = max dissimilarity
    
    
    @staticmethod 
    def max_cosine_loss(post, prior, balance):
        if 'std' in post.keys() and 'std' in prior.keys():
            raise ValueError("Distance mode cosine is not available with distribution mode") 
                
        post = post["mean"].detach()
        prior = prior["mean"]
        
        norm = torch.max(torch.norm(post, dim=-1, keepdim=True), torch.norm(prior, dim=-1, keepdim=True)) + 1e-12
        loss = torch.einsum("ijkl,ijkl->ijk", post / norm, prior / norm) # be sure that we are never dividing by 0
        
        return -loss.mean() # 1 = max similarity | -1 = max dissimilarity

class DistLayer(Module):
    def __init__(
        self, in_dim, out_dim, dist="mse", min_std=0.1, init_std=0.0, bias=True):
        super().__init__()
        self._in_dim = in_dim
        self._shape = out_dim if type(out_dim) in [list, tuple] else [out_dim]
        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std
        self._out = nn.Linear(in_dim, int(np.prod(out_dim)), bias=bias)
        
        if dist in ("normal", "tanh_normal", "trunc_normal", "multivariate_normal"):
            self._std = nn.Sequential(
                nn.Linear(in_dim, int(np.prod(out_dim))), nn.Softplus()
            )
            
    def forward(self, inputs):
        out = self._out(inputs)
        out = out.reshape(list(inputs.shape[:-1]) + list(self._shape))
        if self._dist in ["none", None]: # no distribution
            return out
        if self._dist in ("normal", "tanh_normal", "trunc_normal", "multivariate_normal"):
            std = self._std(inputs)
            std = std.reshape(list(inputs.shape[:-1]) + list(self._shape))
        if self._dist == "mse":
            _ones = torch.ones_like(out, device=out.device)
            dist = D.Normal(out, _ones)
            return D.Independent(dist, len(self._shape))
        if self._dist == "normal":
            dist = D.Normal(out, std)
            return D.Independent(dist, len(self._shape))
        if self._dist == "multivariate_normal": # same as normal definition we assume that the covariance is a diagonal matrix, indipendernce between dimensions
            dist = D.Normal(out, std)
            return D.Independent(dist, len(self._shape))
        if self._dist == "binary":
            out = torch.sigmoid(out)
            dist = BernoulliDist(out)
            return D.Independent(dist, len(self._shape))
        if self._dist == "tanh_normal":
            mean = 5 * torch.tanh(out / 5)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = SquashedNormal(mean, std)
            dist = D.Independent(dist, len(self._shape))
            return SampleDist(dist)
        if self._dist == "trunc_normal":
            mean = torch.tanh(out)
            std = 2 * torch.sigmoid((std + self._init_std) / 2) + self._min_std
            dist = TruncatedNormal(mean, std)
            return D.Independent(dist, 1)
        if self._dist == "onehot":
            return OneHotDist(out)
        if self._dist == "symlog_disc":
            return DiscDist(logits=out)
        if self._dist == "symlog_mse":
            return SymlogDist(out)        
        raise NotImplementedError(self._dist)


### NN_LAYERS ###
class MLP(Module):
    def __init__(
        self, in_shape, shape, layers, units, act=nn.ELU, norm="none", symlog_inputs=False, **out
    ):
        super().__init__()
        self._in_shape = in_shape
        self._shape = (shape,) if isinstance(shape, int) else shape
        self._layers = layers
        self._units = units
        self._norm = norm
        if type(act) is str:
            act = getattr(nn, act)
        self._act = act()  
        self._out = nn.Identity()
        self._symlog_inputs = symlog_inputs
        

        last_units = in_shape
        for index in range(self._layers):
            self.add_module(f"dense{index}", nn.Linear(last_units, units))
            self.add_module(f"norm{index}", NormLayer(norm, units))
            last_units = units
        
        if shape is not None: # In case system distribution is handled outside the method
            self._out = DistLayer(units, shape, **out)

    def forward(self, features):
        x = symlog(features) if self._symlog_inputs else features
        x = x.reshape([-1, x.shape[-1]])
        for index in range(self._layers):
            x = getattr(self, f"dense{index}")(x)
            x = self._act(x)
            x = getattr(self, f"norm{index}")(x)
        x = x.reshape(list(features.shape[:-1]) + [x.shape[-1]])
        return self._out(x)
    
    def compose_output(self, input):
        out = {}
        out["dist"] = input
        out["mean"] = input.mean

        # in case of mse distribution, we take the mean as the sample
        if self._out._dist == "mse":
            out["sample"] = input.mean
        else:
            out["sample"] = SampleDist(input).sample()

        return out

class GRUCell(Module):
    def __init__(
        self,
        inp_size,
        size,
        norm=False,
        act=nn.Tanh,
        update_bias=-1,
        device="cuda",
        **kwargs,
    ):
        super().__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act()
        self._norm = norm
        self._update_bias = update_bias
        self.device = device
        self._layer = nn.Linear(
            inp_size + size, 3 * size, bias=norm is not None, **kwargs
        )
        if norm:
            self._norm = nn.LayerNorm(3 * size)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return torch.zeros((batch_size), self._size, device=self.device)

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # State is wrapped in a list.
        parts = self._layer(torch.cat([inputs, state], -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.chunk(parts, 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]

class NormLayer(Module):
    def __init__(self, name, dim=None):
        super().__init__()
        if name == "none":
            self._layer = None
        elif name == "layer":
            assert dim != None
            self._layer = nn.LayerNorm(dim)
        elif name == "batch1d":
            assert dim != None
            self._layer = nn.BatchNorm1d(dim)
        else:
            raise NotImplementedError(name)

    def forward(self, features):
        if self._layer is None:
            return features
        return self._layer(features)

class ActLayer(Module):
    def __init__(self, name):
        super().__init__()
        if name == "none":
            self._act = None
        elif name != "none":
            self._act = getattr(nn, name)()
        else:
            raise NotImplementedError(name)

    def forward(self, features):
        if self._act is None:
            return features
        return self._act(features)
    
class StreamNorm:
    def __init__(
        self, shape=(), momentum=0.99, scale=1.0, eps=1e-8, device="cuda"
    ):
        # Momentum of 0 normalizes only based on the current batch.
        # Momentum of 1 disables normalization.

        self.device = device
        self._shape = tuple(shape)
        self._momentum = momentum
        self._scale = scale
        self._eps = eps
        self.mag = torch.ones(shape, device=self.device)

    def __call__(self, inputs):
        metrics = {}
        self.update(inputs)
        metrics["mean"] = inputs.mean()
        metrics["std"] = inputs.std()
        outputs = self.transform(inputs)
        metrics["normed_mean"] = outputs.mean()
        metrics["normed_std"] = outputs.std()
        return outputs, metrics

    def reset(self):
        self.mag = torch.ones_like(self.mag, device=self.device)

    def update(self, inputs):
        batch = inputs.reshape((-1,) + self._shape)
        mag = torch.abs(batch).mean(0)
        self.mag.data = (
            self._momentum * self.mag.data + (1 - self._momentum) * mag
        )

    def transform(self, inputs):
        values = inputs.reshape((-1,) + self._shape)
        values /= self.mag[None] + self._eps
        values *= self._scale
        return values.reshape(inputs.shape)

class RequiresGrad:
    def __init__(self, model):
        self._model = model

    def __enter__(self):
        self._model.requires_grad_(requires_grad=True)

    def __exit__(self, *args):
        self._model.requires_grad_(requires_grad=False)
        
class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False, use_cuda=True):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r
        self.use_cuda = use_cuda

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        if self.rank == 1:
            batch_size_shape, channel_in_shape, dim_x = input_tensor.shape
            xx_range = torch.arange(dim_x, dtype=torch.int32)
            xx_channel = xx_range[None, None, :]

            xx_channel = xx_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
            out = torch.cat([input_tensor, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

            xx_range = torch.arange(dim_y, dtype=torch.int32)
            yy_range = torch.arange(dim_x, dtype=torch.int32)
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]

            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)

            # transpose y
            yy_channel = yy_channel.permute(0, 1, 3, 2)

            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 3:
            batch_size_shape, channel_in_shape, dim_z, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, 1, dim_y], dtype=torch.int32)
            zz_ones = torch.ones([1, 1, 1, 1, dim_z], dtype=torch.int32)

            xy_range = torch.arange(dim_y, dtype=torch.int32)
            xy_range = xy_range[None, None, None, :, None]

            yz_range = torch.arange(dim_z, dtype=torch.int32)
            yz_range = yz_range[None, None, None, :, None]

            zx_range = torch.arange(dim_x, dtype=torch.int32)
            zx_range = zx_range[None, None, None, :, None]

            xy_channel = torch.matmul(xy_range, xx_ones)
            xx_channel = torch.cat([xy_channel + i for i in range(dim_z)], dim=2)
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            yz_channel = torch.matmul(yz_range, yy_ones)
            yz_channel = yz_channel.permute(0, 1, 3, 4, 2)
            yy_channel = torch.cat([yz_channel + i for i in range(dim_x)], dim=4)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            zx_channel = torch.matmul(zx_range, zz_ones)
            zx_channel = zx_channel.permute(0, 1, 4, 2, 3)
            zz_channel = torch.cat([zx_channel + i for i in range(dim_y)], dim=3)
            zz_channel = zz_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
                zz_channel = zz_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel, zz_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) +
                                torch.pow(yy_channel - 0.5, 2) +
                                torch.pow(zz_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out

class CoordConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out
    
def policy_selector(policy1, policy2, conditions):
    return policy1 if any(conditions) else policy2
    
def stop_gradient(x):
    return x.detach()

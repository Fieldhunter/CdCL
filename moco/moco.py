import torch
import torch.nn as nn
from itertools import combinations


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=256):
        super(projection_MLP, self).__init__()

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim, affine=True)

        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.linear2(x)
        x = self.bn2(x)

        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=64, out_dim=256):  # bottleneck structure
        super(prediction_MLP, self).__init__()

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

        self.activation = nn.GELU()

    def forward(self, x):
        # layer 1
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.activation(x)
        # N C
        x = self.layer2(x)
        return x


class FastMoCo(nn.Module):
    def __init__(self, backbone, dim_fc=256, projector=projection_MLP, predictor=prediction_MLP, m=0.99, split_num=2, combs=2):
        super(FastMoCo, self).__init__()

        self.m = m
        self.split_num = split_num
        self.combs = combs
        self.dim_fc = dim_fc

        self.encoder = nn.Sequential(
            backbone(),
            projector(self.dim_fc, self.dim_fc, self.dim_fc)
        )
        self.encoder_ema = nn.Sequential(
            backbone(),
            projector(self.dim_fc, self.dim_fc, self.dim_fc)
        )
        self.predictor = predictor(self.dim_fc, self.dim_fc // 4, self.dim_fc)

        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_ema.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        for param_q, param_t in zip(self.encoder.parameters(), self.encoder_ema.parameters()):
            param_t.data = param_t.data.mul_(self.m).add_(param_q.data, alpha=1. - self.m)

    def _local_split(self, x):     # NxCxHxW --> 4NxCx(H/2)x(W/2)
        _side_indent = x.size(2) // self.split_num, x.size(3) // self.split_num
        cols = x.split(_side_indent[1], dim=3)
        xs = []
        for _x in cols:
            xs += _x.split(_side_indent[0], dim=2)
        x = torch.cat(xs, dim=0)
        return x

    def forward(self, lr):
        if self.training:
            f, h = self.encoder, self.predictor

            all_p = []
            for i in range(lr.size(1)):
                x = lr[:, i, ...]
                x_in_form = self._local_split(x)
                _, z_pre = f[0](x_in_form)
                z_splits = list(z_pre.split(z_pre.size(0) // self.split_num ** 2, dim=0))
                z_orthmix = torch.cat(
                    list(map(lambda y: sum(y) / self.combs, list(combinations(z_splits, r=self.combs)))),
                    dim=0)  # 6 of 2combs / 4 of 3combs
                z = f[1](z_orthmix)
                p = h(z)

                all_p.append(p)

            all_z = []
            with torch.no_grad():
                self._momentum_update_target_encoder()
                for i in range(lr.size(1)):
                    _, out = self.encoder_ema[0](lr[:, i, ...])
                    all_z.append(self.encoder_ema[1](out))

            embedding, _ = f[0](lr[:, 0, ...])
            return embedding, all_p, all_z
        else:
            embedding, _ = self.encoder[0](lr)

            return embedding

import torch
from torch import nn
import model.common as common
import torch.nn.functional as F
from moco.moco import FastMoCo
from timm.models.layers import DropPath


def make_model(args):
    return BlindSR(args)


class da_conv(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(da_conv, self).__init__()
        self.loc = channel_adaptation(channels_in, channels_out, reduction)

        self.sig1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1),
            nn.Conv2d(channels_in, channels_out, 3, 1, 1, groups=channels_in),
        )

    def forward(self, xs, ca, xk):
        out1 = self.conv(self.loc(xs, ca))
        sig = self.sig1(xk) * xs
        out = out1 + sig
        return out


class channel_adaptation(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super().__init__()
        # channel attention
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.avgafter = nn.Sequential(
            nn.Linear(channels_in, channels_in // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels_in // reduction, channels_out, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x, ca):
        # x B C H W
        # ca B 64
        ca_x = self.avg(x).squeeze(-1).squeeze(-1)  # B 64
        ca_x = self.avgafter(ca_x)  # B 64
        ca = self.avgafter(ca)  # B 64
        ca = ca + ca_x
        ca_output = self.sig(ca)  # B 64
        ca_output = ca_output.unsqueeze(-1).unsqueeze(-1) * x  # B 64 1 1 * B 64 H W

        return ca_output


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class DaDAU(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 2 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.da_conv = da_conv(dim, dim, reduction=4)

    def forward(self, x, ca, xk):
        input = x
        # Degradation information
        x = self.act(self.da_conv(x, ca, xk))
        x = self.dwconv(x)

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class Extradd(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 2 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class DAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_blocks):
        super(DAG, self).__init__()
        self.n_blocks = n_blocks

        modules_body = [
            DaDAU(n_feat) \
            for _ in range(n_blocks)
        ]
        modules_body.append(Extradd(n_feat))
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x, ca, xk):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        res = x
        for i in range(self.n_blocks):
            res = self.body[i](res, ca, xk)
        res = self.body[-2](res)
        res = self.body[-1](res)
        res = res + x

        return res


def check_image_size(xk, x):
    _, _, H, W = xk.size()
    _, _, h, w = x.size()
    mod_pad_h = h - H
    mod_pad_w = w - W
    xk = F.pad(xk, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return xk


class CdCL(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(CdCL, self).__init__()

        self.n_groups = 6
        n_blocks = 6
        n_feats = 64
        kernel_size = 3
        reduction = 8
        scale = int(args.scale[0])

        # RGB mean for DF2K
        rgb_mean = (0.4680, 0.4484, 0.4029)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(1.0, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(1.0, rgb_mean, rgb_std, 1)

        # head module
        modules_head = [conv(3, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        # compress
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.compress = nn.Sequential(
            nn.Linear(256, n_feats, bias=False),
            nn.GELU()
        )
        # compress
        self.pixshuffle = nn.PixelShuffle(4)
        self.afterpixshuffle = nn.Sequential(
            nn.Conv2d(16, 64, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(64, 16, 3, 1, 1),
            nn.GELU()
        )

        # body
        modules_body = [
            DAG(common.default_conv, n_feats, kernel_size, reduction, n_blocks) \
            for _ in range(self.n_groups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # tail
        modules_tail = [common.Upsampler(conv, scale, n_feats, act=False),
                        conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, k_v):
        ca = self.avg(k_v).squeeze(-1).squeeze(-1)
        ca = self.compress(ca)
        xk = self.pixshuffle(k_v)
        xk = self.afterpixshuffle(xk)
        xk = check_image_size(xk, x)

        # sub mean
        x = self.sub_mean(x)

        # head
        x = self.head(x)

        # body
        res = x
        for i in range(self.n_groups):
            res = self.body[i](res, ca, xk)
        res = self.body[-1](res)
        res = res + x

        # tail
        x = self.tail(res)

        # add mean
        x = self.add_mean(x)

        return x


class Estimator(nn.Module):
    def __init__(self):
        super(Estimator, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
        )
        self.Avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        fea = self.E(x)  # B 256 H/4 W/4
        out = self.Avg(fea).squeeze(-1).squeeze(-1)  # B 256
        return fea, out


class BlindSR(nn.Module):
    def __init__(self, args):
        super(BlindSR, self).__init__()

        # Generator
        self.G = CdCL(args)

        # Encoder
        self.E = FastMoCo(Estimator)
        if args.n_GPUs > 1:
            self.E = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.E)

    def forward(self, x):
        if self.training:
            fea, all_p, all_z = self.E(x)
            sr = self.G(x[:, 0, ...], fea)

            return sr, all_p, all_z
        else:
            fea = self.E(x)
            sr = self.G(x, fea)

            return sr

from model import common

import torch
import torch.nn as nn


class LabelLabEDSR(nn.Module):
    def __init__(self, config, conv=common.default_conv):
        super(LabelLabEDSR, self).__init__()

        n_resblocks = config.n_resblocks
        n_feats = config.n_feats
        kernel_size = 3
        scale = config.scale
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(config.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=config.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.r_conv = conv(n_feats, 256, 1)
        self.g_conv = conv(n_feats, 256, 1)
        self.b_conv = conv(n_feats, 256, 1)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        r = self.r_conv(x)
        g = self.g_conv(x)
        b = self.b_conv(x)
        out = torch.stack((r, g, b), dim=2)
        return out

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

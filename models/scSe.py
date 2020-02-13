import torch
import torch.nn as nn


class scSEBlock(nn.Module):
    def __init__(self, in_chs):
        super(scSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.channel_excitation = nn.Sequential(nn.Linear(int(in_chs), in_chs),
                                                nn.LeakyReLU(0.1, True),
                                                nn.Sigmoid())
        self.spatial_se = nn.Sequential(nn.Conv1d(in_chs, 1, kernel_size=1, stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        num_batch, chs, _ = x.size()
        chn_se = self.avg_pool(x).view(num_batch, chs)
        chn_se = self.channel_excitation(chn_se).view(num_batch, chs, 1)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)


'''
class scSEBlock(nn.Module):
    def __init__(self, in_chs, reduction=16):
        super(scSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.channel_excitation = nn.Sequential(nn.Linear(in_chs, int(in_chs//reduction)),
                                                nn.LeakyReLU(0.1, True),
                                                nn.Linear(int(in_chs//reduction), in_chs),
                                                nn.Sigmoid())
        self.spatial_se = nn.Sequential(nn.Conv1d(in_chs, 1, kernel_size=1, stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        num_batch, chs, _ = x.size()
        chn_se = self.avg_pool(x).view(num_batch, chs)
        chn_se = self.channel_excitation(chn_se).view(num_batch, chs, 1)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)
'''

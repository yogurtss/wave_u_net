import torch
import torch.nn as nn
import torch.nn.functional as F
from models.scSe import scSEBlock
from models.util import *


class DownBlock(nn.Module):
    def __init__(self, in_chs, short_chs, out_chs, k, stride=4, dilation=1):
        super(DownBlock, self).__init__()
        if dilation == 1:
            self.pre_conv = convolution(in_chs, short_chs, kernel_size=k, stride=1)
            self.post_conv = convolution(short_chs, out_chs, kernel_size=k, stride=1)
        else:
            self.pre_conv = dilated_convolution(in_chs, short_chs, kernel_size=k, dilation=dilation, stride=1)
            self.post_conv = dilated_convolution(short_chs, out_chs, kernel_size=k, dilation=dilation, stride=1)
        self.down_conv = convolution(out_chs, out_chs, kernel_size=k, stride=stride)

    def forward(self, x):
        shortcut = x
        shortcut = self.pre_conv(shortcut)
        out = shortcut
        out = self.post_conv(out)
        out = self.down_conv(out)
        return out, shortcut


class UpBlock(nn.Module):
    def __init__(self, in_chs, short_chs, out_chs, k, stride=4):
        super(UpBlock, self).__init__()
        self.up_conv = trans_convolution(in_chs, in_chs, kernel_size=k, stride=stride)
        self.pre_conv = convolution(in_chs, out_chs, kernel_size=k, stride=1)
        self.post_conv = convolution(out_chs+short_chs, out_chs, kernel_size=k, stride=1)

    def forward(self, shortcut, x):
        upsampled = self.up_conv(x)
        upsampled = self.pre_conv(upsampled)
        combined = crop(shortcut, upsampled)
        combined = self.post_conv(torch.cat([combined, upsampled], dim=1))
        return combined


class WaveUNet(nn.Module):
    '''
    Impletation of waveUnet
    layers: number of convolutional layer, default=12
    ch_in: number of input audio chaneel, default=1, means we use one microphone to measure results
    ch_out: number of output audio channel, default=1
    fd: kernel size of the DownsampleBlock, default=15
    fu: kernel size of the UpsampleBlock, default=5
    '''

    def __init__(self, layers=5, ch_in=1, ch_out=1, fd=5, fu=5, num_chs=32, stride=4, dilation=1, training=True):

        super(WaveUNet, self).__init__()
        self.layers = layers
        self.training = training
        in_chs = [num_chs * 2 ** i for i in range(layers + 1)]
        out_chs = in_chs[::-1]

        for i in range(layers):
            in_ch = ch_in if i == 0 else in_chs[i]
            short_ch = in_chs[i]
            out_ch = in_chs[i+1]
            setattr(self, 'down_{}'.format(str(i)), DownBlock(in_ch, short_ch, out_ch, fd, stride, dilation))

        self.center = convolution(in_chs[-1], out_chs[0], fd)
        for i in range(layers):
            in_ch = out_chs[i]
            short_ch = out_chs[i+1]
            out_ch = out_chs[i+1]
            setattr(self, 'up_{}'.format(str(i)), UpBlock(in_ch, short_ch, out_ch, fu, stride))
        self.out = nn.Sequential(nn.Conv1d(out_chs[-1], ch_out, 1),
                                 nn.Tanh())

    def forward(self, x):
        merge_x = []
        for i in range(self.layers):
            down_layers = getattr(self, 'down_{}'.format(str(i)))
            x, short_cut = down_layers(x)
            merge_x.append(short_cut)
        x = self.center(x)

        for i in range(self.layers):
            up_layers = getattr(self, 'up_{}'.format(str(i)))
            x = up_layers(merge_x[-1 - i], x)

        output = self.out(x)
        if not self.training:
            output = output.clamp(-1.0, 1.0)
        return output

    def initialize(self):
        print("Initilizing model")
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    '''
    a = torch.tensor([1.0, 3, 2, 4, 5, 6])
    a = a.unsqueeze(0).unsqueeze(1)
    b = F.interpolate(a, scale_factor=0.5)
    print(a)
    print(b)
    '''

    model = WaveUNet()
    print(model)
    a = torch.zeros([1, 1, 57341])
    b = model(a)
    print(model)

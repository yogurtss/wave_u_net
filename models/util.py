import torch.nn as nn
import torch.nn.functional as F

class convolution(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, with_bn=True, is_pading=False):
        super(convolution, self).__init__()
        if is_pading:
            pad = (kernel_size - 1) // 2
        else:
            pad = 0
        self.conv = nn.Conv1d(in_chs, out_chs, kernel_size, stride, pad)
        self.bn = nn.GroupNorm(out_chs//8, out_chs) if with_bn else nn.Sequential()
        # self.relu = nn.LeakyReLU(0.1, inplace=True)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x, inplace=True)
        return x


class dilated_convolution(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, dilation=2, with_bn=True):
        super(dilated_convolution, self).__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_chs, out_chs, kernel_size, stride, pad, dilation)
        self.bn = nn.BatchNorm1d(out_chs) if with_bn else nn.Sequential
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class trans_convolution(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=4, with_bn=True, is_pading=False):
        super(trans_convolution, self).__init__()
        pad = kernel_size - 1
        self.conv = nn.ConvTranspose1d(in_chs, out_chs, kernel_size, stride, pad)
        self.bn = nn.GroupNorm(out_chs//8, out_chs) if with_bn else nn.Sequential()
        # self.relu = nn.LeakyReLU(0.1, inplace=True)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x, inplace=True)
        return x


def crop(x, target):
    '''
    Center-crop 3-dim. input tensor along last axis so it fits the target tensor shape
    :param x: Input tensor
    :param target: Shape of this tensor will be used as target shape
    :return: Cropped input tensor
    '''
    if x is None:
        return None
    if target is None:
        return x

    target_shape = target.shape
    diff = x.shape[-1] - target_shape[-1]
    assert (diff % 2 == 0)
    crop = diff // 2

    if crop == 0:
        return x
    if crop < 0:
        raise ArithmeticError

    return x[:, :, crop:-crop].contiguous()


class Resample1d(nn.Module):
    def __init__(self, channels, kernel_size, stride, transpose=False, padding="reflect", trainable=False):
        '''
        Creates a resampling layer for time series data (using 1D convolution) - (N, C, W) input format
        :param channels: Number of features C at each time-step
        :param kernel_size: Width of sinc-based lowpass-filter (>= 15 recommended for good filtering performance)
        :param stride: Resampling factor (integer)
        :param transpose: False for down-, true for upsampling
        :param padding: Either "reflect" to pad or "valid" to not pad
        :param trainable: Optionally activate this to train the lowpass-filter, starting from the sinc initialisation
        '''
        super(Resample1d, self).__init__()

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.transpose = transpose
        self.channels = channels

        cutoff = 0.5 / stride

        assert(kernel_size > 2)
        assert ((kernel_size - 1) % 2 == 0)
        assert(padding == "reflect" or padding == "valid")

        filter = build_sinc_filter(kernel_size, cutoff)

        self.filter = torch.nn.Parameter(torch.from_numpy(np.repeat(np.reshape(filter, [1, 1, kernel_size]), channels, axis=0)), requires_grad=trainable)

    def forward(self, x):
        # Pad here if not using transposed conv
        input_size = x.shape[2]
        if self.padding != "valid":
            num_pad = (self.kernel_size-1)//2
            out = F.pad(x, (num_pad, num_pad), mode=self.padding)
        else:
            out = x

        # Lowpass filter (+ 0 insertion if transposed)
        if self.transpose:
            expected_steps = ((input_size - 1) * self.stride + 1)
            if self.padding == "valid":
                expected_steps = expected_steps - self.kernel_size + 1

            out = F.conv_transpose1d(out, self.filter, stride=self.stride, padding=0, groups=self.channels)
            diff_steps = out.shape[2] - expected_steps
            if diff_steps > 0:
                assert(diff_steps % 2 == 0)
                out = out[:,:,diff_steps//2:-diff_steps//2]
        else:
            assert(input_size % self.stride == 1)
            out = F.conv1d(out, self.filter, stride=self.stride, padding=0, groups=self.channels)

        return out

    def get_output_size(self, input_size):
        assert(input_size > 1)
        if self.transpose:
            if self.padding == "valid":
                return ((input_size - 1) * self.stride + 1) - self.kernel_size + 1
            else:
                return ((input_size - 1) * self.stride + 1)
        else:
            assert(input_size % self.stride == 1) # Want to take first and last sample
            if self.padding == "valid":
                return input_size - self.kernel_size + 1
            else:
                return input_size

    def get_input_size(self, output_size):
        # Strided conv/decimation
        if not self.transpose:
            curr_size = (output_size - 1)*self.stride + 1 # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = output_size

        # Conv
        if self.padding == "valid":
            curr_size = curr_size + self.kernel_size - 1 # o = i + p - k + 1

        # Transposed
        if self.transpose:
            assert ((curr_size - 1) % self.stride == 0)# We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1
        assert(curr_size > 0)
        return curr_size
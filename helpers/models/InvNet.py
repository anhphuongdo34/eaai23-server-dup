import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    def __init__(self, dim, kernel_size, stride, dropout_rate):
        super(Block, self).__init__()

        self.inv = involution(dim, kernel_size, stride)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4* dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.dropout = nn.Dropout(dropout_rate, inplace=True)

        # self.skip_connect = nn.Conv2d(dim, dim, 1, stride=2)

    def forward(self, x):
        identity = x
        x = self.inv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)

        # First GELU-linear layer
        x = self.dropout(self.act(self.pwconv1(x)))

        # Second linear layer
        x = self.pwconv2(x)

        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x += identity # Wrong identity
        return x

class involution(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels

        # Convolution No.1
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels // reduction_ratio,
            kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(channels // reduction_ratio, eps=1e-05, momentum=0.05, affine=True)
        self.relu = nn.ReLU()

        # Convolution No.2
        self.conv2 = nn.Conv2d(
            in_channels=channels // reduction_ratio,
            out_channels=kernel_size**2 * self.groups,
            kernel_size=1,
            stride=1)

        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, 1)

    def forward(self, x):
        if self.stride != 1: x = self.avgpool(x)
        weight = self.conv2(self.relu(self.batch_norm(self.conv1(x))))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out

class InvNet(nn.Module):
    def __init__(self, in_channel=3, dims=[64, 128, 256, 512], 
                num_per_layers=[3,4,6,2], dropout_rate=0.5, inv_kernel=7):
        super(InvNet, self).__init__()

        self.invnet = []
        self.dims = dims

        # Stem layer
        self.invnet.append(nn.Conv2d(in_channel, dims[0], kernel_size=7, stride=2))
        self.invnet.append(nn.MaxPool2d(2, 2))
        self.invnet.append(LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))

        # Model architecture
        for i in range(len(dims)):
            for _ in range(num_per_layers[i]):
                self.invnet.append(Block(dims[i], inv_kernel, 1, dropout_rate))
                self.invnet.append(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"))
            if i == len(dims)-1:
                break
            self.invnet.append(nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=2, bias=False))
            self.invnet.append(LayerNorm(dims[i+1], eps=1e-6, data_format="channels_first"))
        
        self.invnet.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.invnet = nn.Sequential(*self.invnet)

        self.linear = nn.Linear(dims[-1], 2)
 
    def forward(self, x):
        x = self.invnet(x)
        x = x.view(-1, self.dims[-1])
        return self.linear(x)
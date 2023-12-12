import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .Functions import Encoding, Mean, DropPath, Mlp, GroupNorm, LayerNormChannel, ConvBlock


# SE注意力机制
class SE(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# CBAM注意力机制
class CBAM(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


### ECA注意力机制
class ECA(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out
class CA_CBAM_Block(nn.Module):
   def init(self, channel, reduction=16):
     super(CA_CBAM_Block, self).init()
# Channel Attention (CA) Module
     self.ca_conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                             bias=False)
     self.ca_bn = nn.BatchNorm2d(channel // reduction)
     self.ca_relu = nn.ReLU()
     self.ca_F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                        bias=False)
     self.ca_F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                        bias=False)
     self.ca_sigmoid_h = nn.Sigmoid()
     self.ca_sigmoid_w = nn.Sigmoid()

# Spatial Attention (SA) Module
     self.sa_avg_pool = nn.AdaptiveAvgPool2d(1)
     self.sa_max_pool = nn.AdaptiveMaxPool2d(1)
     self.sa_conv_1x1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3,
                             bias=False)
     self.sa_sigmoid = nn.Sigmoid()


def forward(self, x):
    # Channel Attention (CA) Module
    _, _, h, w = x.size()
    x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
    x_w = torch.mean(x, dim=2, keepdim=True)
    x_cat_conv_relu = self.ca_relu(self.ca_bn(self.ca_conv_1x1(torch.cat((x_h, x_w), 3))))
    x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
    ca_s_h = self.ca_sigmoid_h(self.ca_F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
    ca_s_w = self.ca_sigmoid_w(self.ca_F_w(x_cat_conv_split_w))
    x = x * ca_s_h.expand_as(x) * ca_s_w.expand_as(x)

    # Spatial Attention (SA) Module
    avg_out = self.sa_avg_pool(x)
    max_out = self.sa_max_pool(x)
    x = torch.cat([avg_out, max_out], dim=1)
    x = self.sa_conv_1x1(x)
    sa_s = self.sa_sigmoid(x)
    out = x * sa_s.expand_as(x)
    return out

class GlobalContextAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(GlobalContextAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        y = y.expand_as(x)

        out = x * y
        return out
# 该模块首先对输入进行全局平均池化，然后将结果传递给两个1x1卷积层，其中第一个卷积层用于降维，第二个卷积层用于升维。最后，使用Sigmoid函数对结果进行归一化，并将结果与输入特征图相乘，以产生最终的特征图。
# 此注意力机制模块的主要创新点在于使用全局平均池化来捕捉输入特征图的全局上下文信息，并使用两个1x1卷积层来生成权重。它还通过将Sigmoid函数应用于权重来将它们归一化到0到1之间。最后，它使用权重对输入特征图进行加权，以产生输出特征图


class AdaptiveAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(AdaptiveAttention, self).__init__()

        self.ca_block = CA_Block(in_channels, reduction=ratio)
        self.eca = ECA(in_channels, gamma=2)

        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)

        self.spatial_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.channel_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ca_out = self.ca_block(x)
        eca_out = self.eca(x)

        out = ca_out + eca_out
        out = self.conv2(F.relu(self.conv1(out)))

        # spatial attention
        spa_out = F.adaptive_avg_pool2d(out, 1)
        spa_out = self.spatial_conv(spa_out)
        spa_out = self.sigmoid(spa_out)
        out = out * spa_out.expand_as(out)

        # channel attention
        cha_out = F.adaptive_avg_pool2d(out, 1)
        cha_out = self.channel_conv(cha_out)
        cha_out = self.sigmoid(cha_out)
        out = out * cha_out.expand_as(out)

        return out
# 在新的模块中，首先通过CA_Block和ECA两个模块提取出空间注意力和通道注意力的特征，然后将它们相加得到最终的特征，接着经过一层卷积层将通道数减半，以减少计算量。
# 接下来分别对空间注意力和通道注意力进行处理。对于空间注意力，使用自适应平均池化将特征图压缩成1x1大小，再通过一个卷积层一个一个介于0和1之间的标量，最后将该值广播到特征图上，得到通道注意力的输出。
# 最终将空间注意力和通道注意力的输出相乘得到最终的输出特征。

# LCA模块主要是基于局部上下文信息进行注意力加权，具体实现包括以下步骤：
#
# 对输入特征图进行3x3的卷积，得到卷积特征map；
# 对卷积特征map沿着通道维度进行最大值池化和均值池化，得到最大值特征和均值特征；
# 将最大值特征和均值特征拼接在一起，进行3x3的卷积，得到特征权重图；
# 将特征权重图经过sigmoid激活函数归一化，得到特征权重系数；
# 对输入特征图按通道进行加权，得到加权特征图。
# 下面是LCA模块的实现代码：
class LCA(nn.Module):
    def __init__(self, channel):
        super(LCA, self).__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_att = nn.Conv2d(channel*2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_conv = self.conv(x)
        x_max = self.max_pool(x_conv)
        x_avg = self.avg_pool(x_conv)
        x_att = torch.cat([x_max, x_avg], dim=1)
        x_att = self.conv_att(x_att)
        x_att = self.sigmoid(x_att)
        out = x * x_att.expand_as(x)
        return out
# 其中，channel为输入特征图的通道数，x为输入特征图。LCA模块中使用的卷积核大小和填充方式与标准的卷积层相同。
# 在conv_att中使用的卷积核大小也为3x3，由于是针对通道方向进行操作，因此卷积核数量为1。在sigmoid中进行归一化，
# 得到特征权重系数。最后按通道进行加权得到加权特征图。

# 为了创造一个新的注意力机制模块，我们可以结合现有的注意力机制模块中的优点，对它们进行改进和组合。我们可以使用自注意力机制（self-attention）和通道注意力机制（channel attention）结合的方法，同时加入一个门控机制来控制信息的流动。
# 下面是一个可能的新注意力机制模块的代码实现：
class SAC(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(SAC, self).__init__()

        self.conv_query = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.conv_key = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.conv_value = nn.Conv2d(in_planes, in_planes, kernel_size=1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.channel_attention = ChannelAttention(in_planes, ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        query = self.conv_query(x)
        key = self.conv_key(x)
        value = self.conv_value(x)

        # calculate self-attention map
        sim_map = torch.matmul(query.view(query.size(0), -1, query.size(-1)).permute(0, 2, 1),
                               key.view(key.size(0), -1, key.size(-1)))
        sim_map = sim_map / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))
        sim_map = self.sigmoid(sim_map)

        # calculate attended value
        attended_value = torch.matmul(sim_map, value.view(value.size(0), -1, value.size(-1)))
        attended_value = attended_value.view(value.size())

        # apply channel and spatial attention
        attended_value = attended_value * self.channel_attention(attended_value)
        attended_value = attended_value * self.spatial_attention(attended_value)

        # apply gating mechanism
        out = self.gamma * attended_value + (1 - self.gamma) * x

        return out
# 在这个新的注意力机制模块中，我们使用了一个自注意力机制来捕捉图像的局部和全局上下文信息。
# 我们还加入了一个门控机制，以控制注意力机制和原始输入之间的信息流动。最后，我们在注意力机制中使用了通道注意力机制和空间注意力机制来进一步增强其表示能力。


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""  # CBL

    def __init__(
            self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
            self,
            in_channels,
            out_channels,
            shortcut=True,
            expansion=0.5,
            depthwise=False,
            act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
            self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
            self,
            in_channels,
            out_channels,
            n=1,
            shortcut=True,
            expansion=0.5,
            depthwise=False,
            act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class LVCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_codes, channel_ratio=0.25, base_channel=64):
        super(LVCBlock, self).__init__()
        self.out_channels = out_channels
        self.num_codes = num_codes
        num_codes = 64

        self.conv_1 = ConvBlock(in_channels=in_channels, out_channels=in_channels, res_conv=True, stride=1)

        self.LVC = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            Encoding(in_channels=in_channels, num_codes=num_codes),
            nn.BatchNorm1d(num_codes),
            nn.ReLU(inplace=True),
            Mean(dim=1))
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels), nn.Sigmoid())

    def forward(self, x):
        x = self.conv_1(x, return_x_2=False)
        en = self.LVC(x)
        gam = self.fc(en)
        b, in_channels, _, _ = x.size()
        y = gam.view(b, in_channels, 1, 1)
        x = F.relu_(x + x * y)
        return x


# LightMLPBlock
class LightMLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu",
                 mlp_ratio=4., drop=0., act_layer=nn.GELU,
                 use_layer_scale=True, layer_scale_init_value=1e-5, drop_path=0.,
                 norm_layer=GroupNorm):  # act_layer=nn.GELU,
        super().__init__()
        self.dw = DWConv(in_channels, out_channels, ksize=1, stride=1, act="silu")
        self.linear = nn.Linear(out_channels, out_channels)  # learnable position embedding
        self.out_channels = out_channels

        self.norm1 = norm_layer(in_channels)
        self.norm2 = norm_layer(in_channels)

        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = Mlp(in_features=in_channels, hidden_features=mlp_hidden_dim, act_layer=nn.GELU,
                       drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.dw(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.dw(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# EVCBlock
class EVCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, channel_ratio=4, base_channel=16):
        super().__init__()
        expansion = 2
        ch = out_channels * expansion
        # Stem stage: get the feature maps by conv block (copied form resnet.py) 进入conformer框架之前的处理
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=1, padding=3,
                               bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 1 / 4 [56, 56]

        # LVC
        self.lvc = LVCBlock(in_channels=in_channels, out_channels=out_channels, num_codes=64)  # c1值暂时未定
        # LightMLPBlock
        self.l_MLP = LightMLPBlock(in_channels, out_channels, ksize=1, stride=1, act="silu", act_layer=nn.GELU,
                                   mlp_ratio=4., drop=0.,
                                   use_layer_scale=True, layer_scale_init_value=1e-5, drop_path=0.,
                                   norm_layer=GroupNorm)
        self.cnv1 = nn.Conv2d(ch, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        # LVCBlock
        x_lvc = self.lvc(x1)
        # LightMLPBlock
        x_lmlp = self.l_MLP(x1)
        # concat
        x = torch.cat((x_lvc, x_lmlp), dim=1)
        x = self.cnv1(x)
        return x

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
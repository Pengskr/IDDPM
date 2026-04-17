import torch
import torch.nn as nn

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: intermediate channels (growth channel)
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias) # 输出channel和输入channel nf 相同
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RRDBMapEncoder(nn.Module):
    def __init__(self, in_nc=3, mc=64, gc=32, channel_mult=[1, 2, 4, 8]):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, mc, 3, padding=1)
        
        # 构建分层结构以匹配 U-Net 的分辨率
        self.layers = nn.ModuleList()
        curr_mc = mc
        for i, mult in enumerate(channel_mult):
            stride = 2 if i < len(channel_mult) - 1 else 1  # 只有当前层不是最后一层时，才进行 stride=2 的下采样
            layer = nn.Sequential(
                RRDB(curr_mc, gc=gc),
                nn.Conv2d(curr_mc, mc * mult, 3, stride=stride, padding=1) # 下采样以匹配层级
            )
            self.layers.append(layer)
            curr_mc = mc * mult

    def forward(self, m):
        m_fea = self.conv_first(m)
        hierarchical_features = []
        for layer in self.layers:
            m_fea = layer(m_fea)
            hierarchical_features.append(m_fea)
        return hierarchical_features # 返回各层特征列表供 MFF 使用

import torch
import torch.nn as nn

class MFFModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # DiffRP 使用拼接，所以输入通道是 2 * channels
        self.depthwise_sep_conv = nn.Sequential(
            # 深度卷积
            nn.Conv2d(2 * channels, 2 * channels, 3, padding=1, groups=2 * channels),
            # 逐点卷积
            nn.Conv2d(2 * channels, channels, 1),
            nn.LayerNorm([channels, None, None]), # DiffRP 指定使用 LayerNorm 
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x_f, m_f):
        # 1. 拼接地图特征 m_f 和 U-Net 特征 x_f
        combined = torch.cat([x_f, m_f], dim=1)
        
        # 2. 深度可分离卷积处理并加上残差连接
        out = self.depthwise_sep_conv(combined)
        out = out + x_f 
        
        # 3. 最终卷积
        return self.final_conv(out)
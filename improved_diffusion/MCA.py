import torch
import torch.nn as nn
import torch.nn.functional as F

class MCAModule(nn.Module):
    """
    Map-Conditioned Attention (MCA) Module
    实现逻辑：
    - Spatial Branch: cat([avg_x, max_x, avg_m, max_m]) -> Bx4xHxW -> Conv3x3 -> alpha_sa
    - Channel Branch: cat([gap_x, gmp_x, gap_m, gmp_m]) -> Bx4Cx1x1 -> Conv3x3 -> alpha_ca
    - Fusion: F_out = x_f * alpha + m_f * (1 - alpha)
    - Final: concat([F_out_sa, F_out_ca]) -> Conv3x3 -> Out
    """
    def __init__(self, channels):
        super().__init__()
        
        # --- Spatial Attention (SA) Branch ---
        # 处理 Bx4xHxW 的输入
        self.sa_conv = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        
        # --- Channel Attention (CA) Branch ---
        # 处理 Bx4Cx1x1 的输入
        # 虽然空间尺寸是 1x1，但按照要求使用 3x3 卷积（padding=1 保证尺寸不变）
        self.ca_conv = nn.Sequential(
            nn.Conv2d(4 * channels, channels, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        
        # --- Final Fusion Layer ---
        self.final_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x_f, m_f):
        """
        x_f: U-Net 中间层特征 [B, C, H, W]
        m_f: RRDB middle_layer 输出 [B, C, H, W]
        """
        
        # 1. Spatial Attention Path
        # 提取空间统计量 [B, 1, H, W]
        s_mean_x = torch.mean(x_f, dim=1, keepdim=True)
        s_max_x  = torch.max(x_f, dim=1, keepdim=True)[0]
        s_mean_m = torch.mean(m_f, dim=1, keepdim=True)
        s_max_m  = torch.max(m_f, dim=1, keepdim=True)[0]
        
        # 拼接并生成权重 alpha_sa [B, 1, H, W]
        alpha_sa = self.sa_conv(torch.cat([s_mean_x, s_max_x, s_mean_m, s_max_m], dim=1))
        # 分支融合
        f_out_sa = x_f * alpha_sa + m_f * (1.0 - alpha_sa)
        
        # 2. Channel Attention Path
        # 提取全局通道统计量 [B, C, 1, 1]
        c_gap_x = F.adaptive_avg_pool2d(x_f, 1)
        c_gmp_x = F.adaptive_max_pool2d(x_f, 1)
        c_gap_m = F.adaptive_avg_pool2d(m_f, 1)
        c_gmp_m = F.adaptive_max_pool2d(m_f, 1)
        
        # 拼接后的尺寸为 [B, 4C, 1, 1]
        cat_ca = torch.cat([c_gap_x, c_gmp_x, c_gap_m, c_gmp_m], dim=1)
        # 生成通道权重 alpha_ca [B, C, 1, 1]
        alpha_ca = self.ca_conv(cat_ca)
        # 分支融合
        f_out_ca = x_f * alpha_ca + m_f * (1.0 - alpha_ca)
        
        # 3. Final Fusion
        # 按照图中右侧方式：SA 和 CA 相加的结果进入 3x3 卷积
        out = self.final_conv(f_out_sa + f_out_ca)
        
        return out
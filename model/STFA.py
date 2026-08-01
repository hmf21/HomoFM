import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设的特征维度
IN_CHANNELS = 256  # 输入特征的通道数
OUT_CHANNELS = 256  # 输出特征的通道数
TOKEN_DIM = 512  # MLP和LayerNorm后的Token维度 (对应描述中的 C)


class ChannelAttentionAggregation(nn.Module):
    """
    通道注意力聚合模块 (CAA) - 关注通道一致性。
    使用 Transformer 风格的 Query-Key-Value 注意力机制实现跨通道聚合。
    """

    def __init__(self, in_channels, token_dim):
        super().__init__()

        # 1. 通道压缩和MLP (对应描述中的 F_S -> F_S_LN 过程)
        # F_S' -> F_S_LN = LN(MLP(F_S'))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim)
        )
        self.norm = nn.LayerNorm(token_dim)

        # 2. Key 和 Value 的投影（假设 F_T 用于 K 和 V）
        # 描述中 F_T_LN 用作 K 和 V
        self.kv_proj = nn.Linear(in_channels, token_dim * 2)  # K 和 V
        self.q_proj = nn.Linear(token_dim, token_dim)  # Q

    def forward(self, F_S, F_T):
        # 假设输入特征形状: (B, C, H, W)
        B, C, H, W = F_S.shape
        N = H * W  # 空间维度展平成 Token 数量 N

        # --- 步骤 1: 对齐与标准化（模拟） ---

        # F_S 特征处理 (作为 Query 的来源)
        # F_S' (对齐) -> Reshape to (B, N, C)
        F_S_flat = F_S.flatten(2).transpose(1, 2)  # (B, C, N) -> (B, N, C)

        # F_S_LN = LN(MLP(F_S')) (对应描述中的 Query Q)
        Q = self.norm(self.mlp(F_S_flat))  # (B, N, C_token)

        # F_T 特征处理 (作为 Key/Value 的来源)
        F_T_flat = F_T.flatten(2).transpose(1, 2)  # (B, N, C) -> (B, N, C)

        # F_T_LN 作为 Key 和 Value (K, V)
        kv = self.kv_proj(F_T_flat)
        K, V = kv.chunk(2, dim=-1)  # K, V 形状均为 (B, N, C_token)

        # --- 步骤 2: 跨通道注意力聚合 ---

        # 注意力计算 (基于 Transformer 的 Self-Attention 逻辑)
        # 这里的注意力是在 Token 维度上 (N)，但目的是实现“跨通道”聚合
        # 我们使用标准的点积注意力：Q @ K.T

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, N, N)

        # 加权求和得到输出
        F_T_CAA_flat = torch.matmul(attn_weights, V)  # (B, N, C_token)

        # 还原回图像形状
        F_T_CAA = F_T_CAA_flat.transpose(1, 2).view(B, -1, H, W)  # (B, C_token, H, W)

        return F_T_CAA


class CorrectedChannelAttentionAggregation(nn.Module):
    def __init__(self, in_channels, token_dim):
        super().__init__()

        self.norm_s = nn.LayerNorm(in_channels)  # 通常在通道维做Norm
        self.q_proj = nn.Linear(in_channels, token_dim)
        self.norm_t = nn.LayerNorm(in_channels)
        self.kv_proj = nn.Linear(in_channels, token_dim * 2)
        self.out_proj = nn.Linear(token_dim, in_channels)
        self.temperature = nn.Parameter(torch.ones(1) * (token_dim ** -0.5))

    def forward(self, F_S, F_T):
        B, C, H, W = F_S.shape

        F_S_flat = F_S.flatten(2).transpose(1, 2)
        F_T_flat = F_T.flatten(2).transpose(1, 2)

        Q = self.q_proj(self.norm_s(F_S_flat))  # (B, N, dim)
        K, V = self.kv_proj(self.norm_t(F_T_flat)).chunk(2, dim=-1)  # (B, N, dim)

        q = Q.transpose(-2, -1)  # (B, dim, N)
        k = K  # (B, N, dim) (保持 N 在中间用于消去)

        attn = torch.matmul(q, k) * self.temperature  # (B, dim, dim)
        attn = F.softmax(attn, dim=-1)  # 对 Key 的通道维度归一化

        v = V.transpose(-2, -1)
        out = torch.matmul(attn, v)
        out = out.transpose(-2, -1)
        out = self.out_proj(out)  # (B, N, C)
        out = out + F_T_flat

        return out.transpose(1, 2).view(B, C, H, W)


import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialCrossAttentionFusion(nn.Module):
    """
    基于空间交叉注意力的融合模块 (Standard Transformer Decoder Layer style)

    Query: 来自 CNN 特征 F_T (保留高分辨率和几何细节)
    Key/Value: 来自 DINO 特征 F_S (提供强语义上下文)
    """

    def __init__(self, in_channels_s, in_channels_t, hidden_dim=256, num_heads=4, dropout=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == hidden_dim, "hidden_dim 必须能被 num_heads 整除"

        # 1. 特征投影层 (将不同通道数的输入映射到同一维度)
        # 使用 1x1 卷积代替 Linear，方便处理 (B, C, H, W)
        self.proj_q = nn.Conv2d(in_channels_t, hidden_dim, kernel_size=1)
        self.proj_k = nn.Conv2d(in_channels_s, hidden_dim, kernel_size=1)
        self.proj_v = nn.Conv2d(in_channels_s, hidden_dim, kernel_size=1)

        # 2. 输出投影
        self.out_proj = nn.Linear(hidden_dim, in_channels_t)  # 还原回 F_T 的通道数

        # 3. Norm 和 Dropout
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 4. FFN (Feed Forward Network) - 增强非线性表达能力
        self.ffn = nn.Sequential(
            nn.Linear(in_channels_t, in_channels_t * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels_t * 2, in_channels_t),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(in_channels_t)

    def forward(self, F_T, F_S):
        """
        输入:
        F_S: Source (DINO) -> (B, Cs, Hs, Ws)
        F_T: Target (CNN)  -> (B, Ct, Ht, Wt)
        """
        B, Ct, Ht, Wt = F_T.shape
        _, Cs, Hs, Ws = F_S.shape

        # --- 1. 准备 Query, Key, Value ---

        # Query 来自 F_T (我们希望输出保持 F_T 的分辨率)
        q = self.proj_q(F_T)  # (B, dim, Ht, Wt)
        q = q.flatten(2).transpose(1, 2)  # (B, Nt, dim), Nt = Ht*Wt
        q = self.norm_q(q)

        # Key, Value 来自 F_S (无需插值，直接展平)
        # 即使 Hs != Ht 也没关系，CrossAttention 允许序列长度不同
        k = self.proj_k(F_S).flatten(2).transpose(1, 2)  # (B, Ns, dim)
        v = self.proj_v(F_S).flatten(2).transpose(1, 2)  # (B, Ns, dim)
        k = self.norm_k(k)

        # --- 2. 多头注意力计算 (Multi-Head Attention) ---

        # Reshape for multi-head: (B, N, num_heads, head_dim)
        q_h = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, Nt, head_dim)
        k_h = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, Ns, head_dim)
        v_h = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, Ns, head_dim)

        # Attention Score: (B, heads, Nt, head_dim) @ (B, heads, head_dim, Ns) -> (B, heads, Nt, Ns)
        # 物理含义: 每个 CNN 像素 (Nt) 去看所有 DINO 像素 (Ns) 的相关性
        attn_scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)  # 在 Ns 维度归一化
        attn_probs = self.dropout(attn_probs)

        # Weighted Sum: (B, heads, Nt, Ns) @ (B, heads, Ns, head_dim) -> (B, heads, Nt, head_dim)
        out_h = torch.matmul(attn_probs, v_h)

        # Merge Heads: (B, Nt, hidden_dim)
        out = out_h.transpose(1, 2).contiguous().view(B, -1, self.hidden_dim)

        # --- 3. 输出投影与残差连接 ---

        out = self.out_proj(out)  # (B, Nt, Ct)

        # 第一次残差: Attention Output + Original CNN Features
        # 注意: 需要先把 F_T 展平才能相加
        F_T_flat = F_T.flatten(2).transpose(1, 2)  # (B, Nt, Ct)
        out = out + F_T_flat

        # --- 4. FFN Block (类似于 Transformer Block 的后半部分) ---
        # 这一步对于融合效果至关重要，它负责整合特征
        out_ffn = self.ffn(self.norm_ffn(out))
        out = out + out_ffn  # 第二次残差

        # Reshape 回图像格式
        return out.transpose(1, 2).view(B, Ct, Ht, Wt)
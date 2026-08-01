import torch
import torch.nn as nn
import torch.nn.functional as F
from model.replknet import create_RepLKNet31B


class ConvBnAct(nn.Module):
    """ 基础卷积块 """

    def __init__(self, in_c, out_c, k, s, p):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RepLKNetBackboneWrapper(nn.Module):
    """
    一个包装器，用于加载官方 RepLKNet 并提取中间层特征
    """

    def __init__(self, pretrained=True):
        super().__init__()
        # 创建模型 (num_classes 不重要，因为我们只取特征)
        self.model = create_RepLKNet31B(num_classes=21841)

        if pretrained:
            # 这里你需要手动指定下载好的 .pth 权重路径
            # 下载地址: https://github.com/DingXiaoH/RepLKNet-pytorch
            # 假设你下载了权重并命名为 RepLKNet-31B_ImageNet-1K.pth
            weight_path = './model/weights/RepLKNet-31B_ImageNet-22K.pth'
            try:
                checkpoint = torch.load(weight_path, map_location='cpu')
                if 'model' in checkpoint:
                    checkpoint = checkpoint['model']
                self.model.load_state_dict(checkpoint, strict=False)
                print(f"RepLKNet Loaded in : {weight_path}")
            except FileNotFoundError:
                print(f"Warrning: Cannot Find the Weights Files {weight_path}，Randomly Initialization")

        # 准备存储特征的字典
        self._features = {}

        # 注册钩子 (Hook)
        # RepLKNet 的结构通常是 model.stages[0], [1], [2], [3]
        self.model.stages[0].register_forward_hook(self._get_feat('stage0'))  # Stride 4
        self.model.stages[1].register_forward_hook(self._get_feat('stage1'))  # Stride 8
        self.model.stages[2].register_forward_hook(self._get_feat('stage2'))  # Stride 16
        # self.model.stages[3] 是 Stride 32，如果你不需要就不注册

    def _get_feat(self, name):
        def hook(model, input, output):
            self._features[name] = output

        return hook

    def forward(self, x):
        self._features = {}  # 清空上一轮
        _ = self.model(x)  # 前向传播，触发钩子
        # 返回 C2(S4), C3(S8), C4(S16)
        return [self._features['stage0'], self._features['stage1'], self._features['stage2']]

    def get_channels(self):
        # RepLKNet-31B 的标准通道数
        return [128, 256, 512, 1024]



class HighResRepLKAdapter(nn.Module):
    def __init__(self, target_dims=[8, 16, 32, 64]):
        """
        target_dims: 对应你原模型 feat_chs 的 [Stride1, Stride2, Stride4, Stride8] 通道数
        """
        super().__init__()

        # ==========================================
        # 1. 语义流 (Backbone): RepLKNet
        # ==========================================
        # 输出特征:
        # idx 0 -> Stride 4
        # idx 1 -> Stride 8
        # idx 2 -> Stride 16 (我们需要用到它的深层语义来增强浅层)
        self.backbone = RepLKNetBackboneWrapper(pretrained=True)
        # 获取 RepLKNet 的通道数，例如 [128, 256, 512]
        bb_dims = self.backbone.get_channels()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # ==========================================
        # 2. 空间流 (Spatial Path): 你的“老”代码
        # ==========================================
        # 这里的目的是以极小的计算代价提取 Stride 1 和 Stride 2 的几何细节
        # 我们直接复用你原代码的前两层结构
        print(target_dims)
        self.spatial_s1 = nn.Sequential(
            ConvBnAct(3, target_dims[0], 7, 1, 3),  # Stride 1
            ConvBnAct(target_dims[0], target_dims[0], 5, 1, 2)
        )
        self.spatial_s2 = ConvBnAct(target_dims[0], target_dims[1], 5, 2, 2)  # Stride 2

        # ==========================================
        # 3. 融合层 (Fusion & Adaptation)
        # ==========================================

        # --- Stage 3 (Stride 8) ---
        # 融合 RepLK(S8) 和 RepLK(S16上采样) 以增强语义
        # 输出通道调整为 target_dims[3]
        self.adapter_s8 = nn.Conv2d(bb_dims[1] + bb_dims[2], target_dims[3], 1)

        # --- Stage 2 (Stride 4) ---
        # 融合 RepLK(S4) 和 Stage 3(上采样)
        # 输出通道调整为 target_dims[2]
        self.adapter_s4 = nn.Conv2d(bb_dims[0] + target_dims[3], target_dims[2], 1)

        # --- Stage 1 (Stride 2) ---
        # 融合 Spatial(S2) 和 Stage 2(上采样)
        # 注意：这里是语义与细节汇合的关键点
        self.adapter_s2 = nn.Conv2d(target_dims[1] + target_dims[2], target_dims[1], 1)

        # --- Stage 0 (Stride 1) ---
        # 融合 Spatial(S1) 和 Stage 1(上采样)
        self.adapter_s1 = nn.Conv2d(target_dims[0] + target_dims[1], target_dims[0], 1)


    def train(self, mode=True):
        """
        重写 train 方法，确保在主模型训练时，Backbone 永远保持 eval 模式。
        这对于冻结的 BN 层至关重要，否则数据分布不同会导致崩塌。
        """
        super().train(mode)
        # 强制 backbone 保持 eval 模式 (即使整体在 train)
        self.backbone.eval()


    def forward(self, x):
        input_size = x.shape[-2:]

        # ---------------------------
        # A. 运行 RepLKNet (语义提取)
        # ---------------------------
        # feats[0]: S4, feats[1]: S8, feats[2]: S16
        with torch.no_grad():
            bb_feats = self.backbone(x)

        # ---------------------------
        # B. 运行空间流 (细节提取)
        # ---------------------------
        feat_s1 = self.spatial_s1(x)  # Stride 1 (High Res)
        feat_s2 = self.spatial_s2(feat_s1)  # Stride 2

        # ---------------------------
        # C. 自底向上 + 自顶向下 融合
        # ---------------------------

        # 1. 生成最终的 Stride 8 特征
        # 将 S16 上采样并与 S8 拼接 (利用 RepLKNet 的深层上下文)
        up_s16 = F.interpolate(bb_feats[2], size=bb_feats[1].shape[-2:], mode='bilinear', align_corners=False)
        cat_s8 = torch.cat([bb_feats[1], up_s16], dim=1)
        out_s8 = self.adapter_s8(cat_s8)  # -> 对应你原模型的 feat_chs[3]

        # 2. 生成最终的 Stride 4 特征
        # 将 out_s8 上采样并与 RepLK S4 拼接
        up_s8 = F.interpolate(out_s8, size=bb_feats[0].shape[-2:], mode='bilinear', align_corners=False)
        cat_s4 = torch.cat([bb_feats[0], up_s8], dim=1)
        out_s4 = self.adapter_s4(cat_s4)  # -> 对应你原模型的 feat_chs[2]

        # 3. 生成最终的 Stride 2 特征 (关键步骤：注入高频细节)
        # 将 out_s4 (语义强) 上采样，与 feat_s2 (细节强) 拼接
        up_s4 = F.interpolate(out_s4, size=feat_s2.shape[-2:], mode='bilinear', align_corners=False)
        cat_s2 = torch.cat([feat_s2, up_s4], dim=1)
        out_s2 = self.adapter_s2(cat_s2)  # -> 对应你原模型的 feat_chs[1]

        # 4. 生成最终的 Stride 1 特征
        up_s2 = F.interpolate(out_s2, size=feat_s1.shape[-2:], mode='bilinear', align_corners=False)
        cat_s1 = torch.cat([feat_s1, up_s2], dim=1)
        out_s1 = self.adapter_s1(cat_s1)  # -> 对应你原模型的 feat_chs[0]

        # 返回列表，严格对应原模型的输出层级
        return [out_s8, out_s4, out_s2, out_s1]


if __name__ == '__main__':
    # --------------------------
    # 测试代码
    # --------------------------
    # 假设你原来的配置是 [64, 128, 256, 512]
    model = HighResRepLKAdapter(target_dims=[64, 128, 256, 512])
    dummy_input = torch.randn(1, 3, 512, 512)

    outputs = model(dummy_input)
    print("输出尺寸检查:")
    for i, out in enumerate(outputs):
        stride = 512 // out.shape[2]
        print(f"Output {i}: Shape {out.shape}, Stride {stride}")
        # 预期输出:
        # Output 0: Shape [1, 64, 512, 512], Stride 1
        # Output 1: Shape [1, 128, 256, 256], Stride 2
        # Output 2: Shape [1, 256, 128, 128], Stride 4
        # Output 3: Shape [1, 512, 64, 64], Stride 8
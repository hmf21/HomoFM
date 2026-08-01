import torch.nn as nn
from torch.autograd import Function

class GradientReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReverseLayer.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_channels=512):
        super(DomainDiscriminator, self).__init__()
        # 简单的 3层 1x1 或 3x3 卷积，逐渐压缩通道
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 最后一层输出 1 通道，不加 Sigmoid (使用 BCEWithLogitsLoss 更稳定)
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)
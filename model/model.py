"""Mask-conditioned residual U-Net for RGB comic panels. White denotes a hole."""
import torch
from torch import nn
from torch.nn import functional as F


def context_fill(rgb, mask):
    """Fill holes from progressively coarser, normalized visible neighborhoods."""
    visible = 1 - mask
    value = rgb * visible
    levels = [(value, visible)]
    while max(value.shape[-2:]) > 1:
        value = F.avg_pool2d(value, 2, ceil_mode=True)
        visible = F.avg_pool2d(visible, 2, ceil_mode=True)
        levels.append((value, visible))
    filled = torch.where(visible > 0, value / visible.clamp_min(1e-8), 0.5)
    for value, visible in reversed(levels[:-1]):
        filled = F.interpolate(filled, size=value.shape[-2:], mode="bilinear", align_corners=False)
        filled = torch.where(visible > 0, value / visible.clamp_min(1e-8), filled)
    return filled


def block(cin, cout, stride=1, dilation=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, dilation, dilation=dilation),
                         nn.GroupNorm(4, cout), nn.SiLU())


class NemesisUNet(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.entry = block(7, width)
        self.down1 = block(width, width * 2, stride=2)
        self.down2 = block(width * 2, width * 4, stride=2)
        self.context = nn.Sequential(*(block(width * 4, width * 4, dilation=d) for d in (1, 2, 4)))
        self.up1 = block(width * 6, width * 2)
        self.up0 = block(width * 3, width)
        self.head = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, rgb, mask):
        base = context_fill(rgb, mask)
        h0 = self.entry(torch.cat((rgb * (1 - mask), mask, base), 1))
        h1 = self.down1(h0)
        h2 = self.down2(h1)
        h2 = h2 + self.context(h2)
        h1 = self.up1(torch.cat((F.interpolate(h2, size=h1.shape[-2:], mode="nearest"), h1), 1))
        h0 = self.up0(torch.cat((F.interpolate(h1, size=h0.shape[-2:], mode="nearest"), h0), 1))
        return torch.where(mask.bool(), (base + self.head(h0)).clamp(0, 1), rgb)


def masked_mean(values, mask):
    return (values * mask).sum() / (mask.sum() * values.shape[1]).clamp_min(1)


def objective(pred, target, mask):
    err = pred - target
    pixel = masked_mean(err.abs(), mask)
    dx = err.diff(dim=3).abs()
    dy = err.diff(dim=2).abs()
    edges = masked_mean(dx, torch.maximum(mask[..., 1:], mask[..., :-1]))
    edges += masked_mean(dy, torch.maximum(mask[:, :, 1:], mask[:, :, :-1]))
    return pixel + 0.25 * edges


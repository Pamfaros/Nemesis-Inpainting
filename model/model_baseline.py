"""Residual RGB U-Net and its color-contour variant for manhwa."""
import torch
from torch import nn
from torch.nn import functional as F

from .model import visible_gradients
from .model_hybrid import boundary_loss, context_fill, masked_mean, multiscale_loss


def block(cin, cout, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=dilation, dilation=dilation),
        nn.GroupNorm(4, cout),
        nn.SiLU(),
    )


class NemesisUNet(nn.Module):
    def __init__(self, width=24, color_contours=False):
        super().__init__()
        self.color_contours = color_contours
        self.entry = block(13 if color_contours else 7, width)
        self.down1 = block(width, width * 2, stride=2)
        self.down2 = block(width * 2, width * 4, stride=2)
        self.context = nn.Sequential(*(block(width * 4, width * 4, dilation=d) for d in (1, 2, 4)))
        self.up1 = block(width * 6, width * 2)
        self.up0 = block(width * 3, width)
        self.head = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, rgb, mask):
        mask = mask.to(rgb.dtype)
        base = context_fill(rgb, mask)
        visible = rgb * (1 - mask)
        features = (visible, mask, base)
        if self.color_contours:
            features += visible_gradients(visible, mask.bool())
        h0 = self.entry(torch.cat(features, dim=1))
        h1 = self.down1(h0)
        h2 = self.down2(h1)
        h2 = h2 + self.context(h2)
        h1 = self.up1(torch.cat((F.interpolate(h2, size=h1.shape[-2:], mode="nearest"), h1), dim=1))
        h0 = self.up0(torch.cat((F.interpolate(h1, size=h0.shape[-2:], mode="nearest"), h0), dim=1))
        return torch.where(mask.bool(), (base + self.head(h0)).clamp(0, 1), rgb)


def manhwa_objective(prediction, target, mask):
    """Supervise color and give extra weight to clean edges touching the hole."""
    prediction, target = prediction.float(), target.float()
    error = prediction - target
    mask = mask.float()
    edges = error.new_zeros(())
    for axis in (-1, -2):
        length = mask.shape[axis] - 1
        touches_hole = torch.maximum(mask.narrow(axis, 0, length), mask.narrow(axis, 1, length))
        contrast = target.diff(dim=axis).abs().amax(dim=1, keepdim=True)
        weights = touches_hole * (1 + 3 * contrast)
        edges += (error.diff(dim=axis).abs() * weights).sum() / (3 * weights.sum()).clamp_min(1)
    return (
        masked_mean(error.abs(), mask)
        + .50 * edges
        + .20 * multiscale_loss(prediction, target, mask)
        + .25 * boundary_loss(prediction, target, mask)
    )

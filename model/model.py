"""Nemesis ROI inpainting. RGB tensors are BCHW in [0, 1]. True marks a hole."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def validate(rgb, mask):
    if rgb.ndim != 4 or rgb.shape[1] != 3 or not rgb.is_floating_point():
        raise ValueError("expected floating-point Bx3xHxW RGB")
    if mask.shape != rgb[:, :1].shape or mask.dtype != torch.bool:
        raise ValueError("expected boolean Bx1xHxW mask")
    if rgb.device != mask.device:
        raise ValueError("RGB and mask must be on the same device")


def visible_gradients(values, mask):
    """Signed differences between visible neighbors, aligned to the image grid."""
    dx = values.diff(dim=-1).masked_fill(mask[..., 1:] | mask[..., :-1], 0)
    dy = values.diff(dim=-2).masked_fill(mask[..., 1:, :] | mask[..., :-1, :], 0)
    return F.pad(dx, (1, 0)), F.pad(dy, (0, 0, 1, 0))


def context(rgb, mask):
    """Known RGB and signed luminance differences across known pairs only."""
    known = (rgb * 2 - 1).masked_fill(mask, 0)
    luma = known[:, :1] * .299 + known[:, 1:2] * .587 + known[:, 2:] * .114
    return torch.cat((known, mask.to(rgb.dtype), *visible_gradients(luma, mask)), 1)


class Block(nn.Module):
    def __init__(self, inputs, width, dilation=1):
        super().__init__()
        self.skip = nn.Identity() if inputs == width else nn.Conv2d(inputs, width, 1)
        self.conv1 = nn.Conv2d(inputs, width, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, width)
        self.film = nn.Linear(64, 2 * width)
        self.conv2 = nn.Conv2d(width, width, 3, dilation=dilation, padding=dilation)
        self.norm2 = nn.GroupNorm(8, width)

    def forward(self, x, embedding):
        scale, shift = self.film(embedding)[:, :, None, None].chunk(2, dim=1)
        h = self.norm1(self.conv1(x)) * (1 + scale) + shift
        return self.skip(x) + self.norm2(self.conv2(F.silu(h)))


class ManhwaUNet(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in ("rgb", "diffusion"):
            raise ValueError("mode must be rgb or diffusion")
        self.mode = mode
        widths = (24, 48, 72, 96)
        self.stem = nn.Conv2d(9 if mode == "diffusion" else 6, widths[0], 3, padding=1)
        self.register_buffer("frequency", torch.exp(-math.log(10000) * torch.arange(16) / 15))
        self.condition = nn.Sequential(nn.Linear(38, 64), nn.SiLU(), nn.Linear(64, 64))
        self.encoder = nn.ModuleList(Block(w, w) for w in widths[:-1])
        self.down = nn.ModuleList(nn.Conv2d(a, b, 3, stride=2, padding=1)
                                  for a, b in zip(widths, widths[1:]))
        self.middle = nn.ModuleList(Block(96, 96, d) for d in (1, 2, 4))
        self.decoder = nn.ModuleList(Block(a + b, a) for a, b in reversed(list(zip(widths, widths[1:]))))
        self.out = nn.Conv2d(widths[0], 3, 3, padding=1)
        beta = torch.linspace(1e-4, .02, 1000)
        self.register_buffer("alpha_bar", (1 - beta).cumprod(0))

    def forward(self, cond, noisy=None, step=None):
        if self.mode == "diffusion":
            if noisy is None or step is None:
                raise ValueError("diffusion requires noisy RGB and timestep")
            x = torch.cat((cond, noisy), 1)
        else:
            x = cond
            step = torch.zeros(len(cond), device=cond.device)
        phase = step[:, None] * self.frequency[None]
        embedding = self.condition(torch.cat((cond.mean((2, 3)), phase.sin(), phase.cos()), 1))
        h = self.stem(x)
        skips = []
        for block, downsample in zip(self.encoder, self.down):
            h = block(h, embedding)
            skips.append(h)
            h = downsample(F.silu(h))
        for block in self.middle:
            h = block(h, embedding)
        for block, skip in zip(self.decoder, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat((h, skip), 1), embedding)
        return self.out(F.silu(h))


def training_loss(net, rgb, mask, clean):
    validate(rgb, mask)
    if clean.shape != rgb.shape or not mask.flatten(1).any(1).all():
        raise ValueError("training requires aligned RGB targets and nonempty masks per ROI")
    cond = context(rgb, mask)
    target = clean * 2 - 1
    if net.mode == "rgb":
        error = (net(cond).tanh() - target).abs()
    else:
        step = torch.randint(len(net.alpha_bar), (len(rgb),), device=rgb.device)
        alpha = net.alpha_bar[step, None, None, None]
        noise = torch.randn_like(rgb)
        noisy = torch.where(mask, alpha.sqrt() * target + (1 - alpha).sqrt() * noise, cond[:, :3])
        error = (net(cond, noisy, step) - noise).square()
    return ((error * mask).sum((1, 2, 3)) / (3 * mask.sum((1, 2, 3)))).mean()


@torch.no_grad()
def sample(net, rgb, mask, steps=8, seed=27):
    validate(rgb, mask)
    if not mask.any():
        return rgb.clone()
    cond = context(rgb, mask)
    if net.mode == "rgb":
        result = net(cond).tanh()
    else:
        if steps not in (4, 8, 16):
            raise ValueError("use 4, 8 or 16 DDIM steps")
        rng = torch.Generator(device=rgb.device).manual_seed(seed)
        noise = torch.randn(rgb.shape, generator=rng, device=rgb.device, dtype=rgb.dtype)
        known = cond[:, :3]
        state = torch.where(mask, noise, known)
        times = torch.linspace(999, 0, steps).round().long().tolist()
        for i, t in enumerate(times):
            step = torch.full((len(rgb),), t, device=rgb.device, dtype=torch.long)
            eps = net(cond, state, step)
            alpha = net.alpha_bar[t]
            result = ((state - (1 - alpha).sqrt() * eps) / alpha.sqrt()).clamp(-1, 1)
            if i + 1 < steps:
                # Recompute epsilon after clipping x0 for a consistent DDIM update.
                eps = (state - alpha.sqrt() * result) / (1 - alpha).sqrt()
                prev = net.alpha_bar[times[i + 1]]
                state = torch.where(mask, prev.sqrt() * result + (1 - prev).sqrt() * eps, known)
    return torch.where(mask, (result + 1) * .5, rgb)


def flat_fill(rgb, mask, tolerance=6 / 255):
    """One ROI at a time. Return None when the rule abstains."""
    validate(rgb, mask)
    if len(rgb) != 1:
        raise ValueError("expected one ROI")
    if not mask.any():
        return rgb.clone()
    # L1 dilation, three pixel steps. Zero padding never contributes false ring pixels.
    region = mask
    for _ in range(3):
        pad = F.pad(region, (1, 1, 1, 1))
        region = (region | pad[..., :-2, 1:-1] | pad[..., 2:, 1:-1]
                  | pad[..., 1:-1, :-2] | pad[..., 1:-1, 2:])
    ring = region & ~mask
    if ring.sum() < 16:
        return None
    colors = rgb[0, :, ring[0, 0]]
    center = torch.quantile(colors, .5, dim=1)
    if (colors - center[:, None]).abs().max() > tolerance:
        return None
    return torch.where(mask, center.view(1, 3, 1, 1), rgb)


@torch.no_grad()
def restore(net, rgb, mask, steps=8, seed=27, use_router=True):
    if use_router:
        flat = flat_fill(rgb, mask)
        if flat is not None:
            return flat, "median" if mask.any() else "identity"
    return sample(net, rgb, mask, steps, seed), net.mode

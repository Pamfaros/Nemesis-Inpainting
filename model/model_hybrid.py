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


def conv_block(cin, cout, stride=1, kernel_size=3):
    return nn.Sequential(
        
        nn.Conv2d(
            cin, cin, kernel_size, stride=stride, 
            padding=kernel_size // 2, groups=cin, 
            bias=False
        ),
        
        nn.Conv2d(
            cin, cout, kernel_size=1, bias=False
        ),
         
        nn.GroupNorm(4, cout), 
        nn.SiLU())


class GatedResidualBlock(nn.Module):
    def __init__(self, channels, expansion=2, 
                 kernel_size=5, dilation=1):
        super().__init__()
        
        hidden = channels * expansion
        
        self.norm = nn.GroupNorm(4, channels)
        
        self.expand = nn.Conv2d(
            channels, hidden * 2, kernel_size=1
        )
        
        self.depthwise = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size, 
            padding=(kernel_size // 2) * dilation, 
            dilation=dilation, groups=hidden * 2
        )
        
        self.project = nn.Conv2d(
            hidden, channels, kernel_size=1
        )
        
        self.scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        residual = x
        
        x = self.norm(x)
        x = self.expand(x)
        x = self.depthwise(x)
        
        features, gate = x.chunk(2, dim=1)
        
        x = F.silu(features) * torch.sigmoid(gate)
        
        x = self.project(x)
        
        return residual + self.scale * x

class MaskAwareAttention(nn.Module):
    def __init__(self, channels, heads=4, kv_size=16):
        super().__init__()
        
        assert channels % heads == 0
        
        self.kv_size = kv_size
        
        self.norm = nn.LayerNorm(channels)
        
        self.position = nn.Conv2d(
            channels, channels,kernel_size=3, 
            padding=1, groups=channels
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=channels, num_heads=heads, 
            batch_first=True
        )
        
        self.null_token = nn.Parameter(
            torch.zeros(1, 1, channels)
        )
        
        self.scale = nn.Parameter(
            torch.tensor(0.1)
        )
    
    def forward(self, x, mask):
        b, c, h, w = x.shape
        
        kh = min(h, self.kv_size)
        kw = min(w, self.kv_size)
        
        # Spatial positional information
        features = x + self.position(x)
        
        # Query: all bottleneck positions
        q = features.flatten(2).transpose(1, 2)
        q = self.norm(q)
        
        # Reduced spatial context for keys and values
        context = F.adaptive_avg_pool2d(
            features, (kh, kw)
        )
        
        kv = context.flatten(2).transpose(1, 2)
        kv = self.norm(kv)
        
        # Exclude context cells intersacting the mask
        blocked = F.adaptive_max_pool2d(
            mask.float(), (kh, kw)
        )
        
        coverage = F.adaptive_avg_pool2d(
            mask.float(), (kh, kw)
        )

        blocked = coverage.flatten(1) > 0.25
        
        # Always provide one valid fallback token
        null = self.null_token.to(kv.dtype)
        null = null.expand(b, -1, -1)
        
        kv = torch.cat((kv, null), dim=1)
        
        valid_null = torch.zeros(
            b, 1, dtype=torch.bool, device=x.device
        )
        
        blocked = torch.cat(
            (blocked, valid_null), dim=1
        )
        
        attended, _ = self.attention(
            q, kv, kv, key_padding_mask=blocked, 
            need_weights=False
        )
        
        attended = attended.transpose(1, 2)
        attended = attended.reshape(b, c, h, w)
        
        return x + self.scale * attended
        
class DecoderBloсk(nn.Module):
    def __init__(self, cin, skip_channels, 
                 cout, expansion=1):
        super().__init__()
        
        self.reduce = nn.Conv2d(cin, cout, kernel_size=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(
                cout+skip_channels, cout, 
                kernel_size=1, bias=False
            ),
            
            nn.GroupNorm(4, cout),
            nn.SiLU()
        )
        
        self.refine = GatedResidualBlock(cout, expansion, kernel_size=3)
        
    def forward(self, x, skip):
        x = self.reduce(x)
        
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat((x, skip), dim=1)
        x = self.fusion(x)
        return self.refine(x)

class AxialStateSpace(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.norm = nn.GroupNorm(4, channels)
        self.in_proj = nn.Conv2d(
            channels, channels, kernel_size=1
        )
        self.decay = nn.Parameter(
            torch.full((channels,), 2.0)
        )
        
        self.out_proj = nn.Conv2d(
            channels, channels, kernel_size=1
        )
        
        self.scale = nn.Parameter(
            torch.tensor(0.1)
        )
    
    def scan(self, x, dim, reverse=False):
        sequence = x.unbind(dim=dim)
        if reverse:
            sequence = sequence[::-1]
        decay = torch.sigmoid(self.decay)
        decay = decay.to(x.dtype).view(1, -1, 1)
        
        state = torch.zeros_like(sequence[0])
        outputs = []
        
        for token in sequence:
            state = (
                decay * state + (1 - decay) * token
            )
            outputs.append(state)
        
        if reverse:
            outputs.reverse()
        
        return torch.stack(outputs, dim=dim)
    
    def forward(self, x):
        residual = x
        x = self.in_proj(self.norm(x))
        
        left = self.scan(x, dim=3)
        right = self.scan(x, dim=3, reverse=True)
        
        top = self.scan(x, dim=2)
        bottom = self.scan(x, dim=2, reverse=True)
        
        x = (left + right + top + bottom) / 4
        x = self.out_proj(x)
        
        return residual + self.scale * x
    

class SelectiveAxialStateSpace(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.norm = nn.GroupNorm(4, channels)

        self.local = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels
        )

        self.value_proj = nn.Conv2d(
            channels, channels, 1
        )

        self.gate_proj = nn.Conv2d(
            channels, channels, 1
        )

        self.out_proj = nn.Conv2d(
            channels, channels, 1
        )

        self.direction_weights = nn.Parameter(
            torch.zeros(4, channels)
        )

        self.scale = nn.Parameter(
            torch.tensor(0.1)
        )

        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0)

    def scan(self, values, gates, dim, reverse=False):

        values = values.unbind(dim=dim)
        gates = gates.unbind(dim=dim)

        if reverse:
            values = values[::-1]
            gates = gates[::-1]

        state = torch.zeros_like(values[0])
        outputs = []

        for value, gate in zip(values, gates):

            state = (
                gate * state
                + (1.0 - gate) * value
            )

            outputs.append(state)

        if reverse:
            outputs.reverse()

        return torch.stack(outputs, dim=dim)

    def forward(self, x):

        residual = x

        z = self.local(self.norm(x))

        values = self.value_proj(z).float()

        gates = torch.sigmoid(
            self.gate_proj(z).float()
        ).clamp(0.01, 0.99)

        left = self.scan(values, gates, dim=3)

        right = self.scan(
            values, gates, dim=3, reverse=True
        )

        top = self.scan(values, gates, dim=2)

        bottom = self.scan(
            values, gates, dim=2, reverse=True
        )

        weights = self.direction_weights.softmax(dim=0)

        directions = (left, right, top, bottom)

        result = sum(
            direction * weights[i].view(1, -1, 1, 1)
            for i, direction in enumerate(directions)
        )

        result = self.out_proj(result.to(z.dtype))

        return residual + self.scale * result


class NemesisUNet(nn.Module):
    def __init__(self, width=24, 
                 context_dim=128, 
                 heads=4, kv_size=16, 
                 use_attention=True, 
                 use_ssm=False):
        super().__init__()
        
        assert width % 4 == 0
        assert context_dim % 4 == 0
        assert context_dim % heads == 0
        
        c0, c1 = width, width*2
        c2, c3 = width*4, context_dim
        
        self.use_attention = use_attention
        
        # =======================
        # ENCODER
        # =======================
        
        # 512 → 256
        self.entry = nn.Sequential(
            nn.Conv2d(
                7, c0, kernel_size=3, stride=2, 
                padding=1, bias=False    
            ),
            
            nn.GroupNorm(4, c0),
            nn.SiLU(),
            
            GatedResidualBlock(
                c0, expansion=1, kernel_size=3
            )
        )
        
        # 256 → 128
        self.down1 = nn.Sequential(
            conv_block(c0, c1, stride=2), 
            
            GatedResidualBlock(
                c1, expansion=1
            )
        )
        
        # 128 → 64
        self.down2 = nn.Sequential(
            conv_block(c1, c2, stride=2),
            GatedResidualBlock(c2, expansion=2)
        )
        
        # 64 → 32
        self.down3 = conv_block(c2, c3, stride=2)
        
        # =========================
        # HYBRID CONTEXT BOTTLENECK
        # =========================
        
        self.context = nn.Sequential(
            GatedResidualBlock(c3, dilation=1), 
            GatedResidualBlock(c3, dilation=2),
            GatedResidualBlock(c3, dilation=4)
        )
        
        # Mask-aware spatial attention
        if use_attention:
            self.attention = MaskAwareAttention(
                c3, heads=heads, kv_size=kv_size
            )
        else:
            self.attention = None
            
        self.ssm = (
            AxialStateSpace(c3)
            if use_ssm
            else nn.Identity()
        )
        
        # Feature refinement after attention
        self.refine = GatedResidualBlock(c3, expansion=2)
        
        # =======
        # DECODER
        # =======
        
        # 32 → 64
        self.up2 = DecoderBloсk(c3, c2, c2)
        
        # 64 → 128
        self.up1 = DecoderBloсk(c2, c1, c1)
        
        # 128 → 256
        self.up0 = DecoderBloсk(c1, c0, c0)
        
        # =======================
        # RGB RECONSTRUCTION HEAD
        # =======================
        
        self.head = nn.Conv2d(c0, 3 * 4, kernel_size=3, padding=1)
        
        self.upsample = nn.PixelShuffle(2)
        
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, rgb, mask):
        # Ensure a binary floating-point mask
        mask = (mask > 0.5).to(rgb.dtype)
        
        height, width = rgb.shape[-2:]
        
        # =============
        # PREPROCESSING
        # =============
        
        # FP32 for numerical stability
        base = context_fill(
            rgb.float(), mask.float()
        ).to(rgb.dtype)
        
        visible = rgb * (1 - mask)
        x = torch.cat(
            (visible, mask, base), dim=1
        )
        
        # =======
        # ENCODER
        # =======
        
        h0 = self.entry(x)
        h1 = self.down1(h0)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        
        # ==============
        # GLOBAL CONTEXT
        # ==============
        
        h3 = self.context(h3)
        if self.attention is not None:
            h3 = self.attention(h3, mask)
        
        h3 = self.ssm(h3)
        
        h3 = self.refine(h3)
        
        # =======
        # DECODER
        # =======
        
        x = self.up2(h3, h2)
        x = self.up1(x, h1)
        x = self.up0(x, h0)
        
        # ==================
        # RGB RECONSTRUCTION
        # ==================
        
        residual = self.head(x)
        residual = self.upsample(residual)
        
        # Support odd input dimensions
        residual = residual[..., :height, :width]
        
        prediction = (base + residual).clamp(0, 1)
        
        # Preserve original unmasked pixels
        return torch.where(mask.bool(), prediction, rgb)
        
def masked_mean(values, mask):

    values = values.float()
    mask = (mask > 0.5).float()

    numerator = (values * mask).sum()

    denominator = (
        mask.sum() * values.shape[1]
    ).clamp_min(1)

    return numerator / denominator

def gradient_loss(pred, target, mask):

    error = pred - target

    dx = error.diff(dim=3).abs()
    dy = error.diff(dim=2).abs()

    mx = torch.maximum(
        mask[..., 1:],
        mask[..., :-1]
    )

    my = torch.maximum(
        mask[:, :, 1:],
        mask[:, :, :-1]
    )

    return (
        masked_mean(dx, mx)
        + masked_mean(dy, my)
    )


def multiscale_loss(pred, target, mask):

    losses = []

    for scale in (2, 4, 8):

        if min(pred.shape[-2:]) < scale:
            continue

        p = F.avg_pool2d(pred, scale, ceil_mode=True)
        t = F.avg_pool2d(target, scale, ceil_mode=True)
        m = F.max_pool2d(mask, scale, ceil_mode=True)

        losses.append(
            masked_mean((p - t).abs(), m)
        )

    if not losses:
        return pred.new_zeros(())

    return torch.stack(losses).mean()


def boundary_loss(pred, target, mask):

    eroded = 1 - F.max_pool2d(
        1 - mask,
        kernel_size=5,
        stride=1,
        padding=2
    )

    boundary = (mask - eroded).clamp(0, 1)

    return masked_mean(
        (pred - target).abs(),
        boundary
    )

def objective(pred, target, mask):

    pred = pred.float()
    target = target.float()
    mask = mask.float()

    pixel = masked_mean(
        (pred - target).abs(),
        mask
    )

    edges = gradient_loss(
        pred, target, mask
    )

    multiscale = multiscale_loss(
        pred, target, mask
    )

    boundary = boundary_loss(
        pred, target, mask
    )

    return (
        1.00 * pixel
        + 0.50 * edges
        + 0.20 * multiscale
        + 0.25 * boundary
    )


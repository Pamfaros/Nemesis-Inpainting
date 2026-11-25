"""Simulate the masks supplied by editors and lettering detectors."""
from functools import lru_cache
from math import ceil, floor

from PIL import Image, ImageDraw, ImageFont

MASK_TYPES = ("dialogue", "sfx", "detector")
DIALOGUE = ("괜찮아", "잠깐만", "어떻게 된 거야", "이쪽이야", "정말이야", "뭐라고?", "어서 가자", "알겠어")
SFX = ("쾅", "콰앙", "휘익", "쿵", "철컥", "두근", "파앗", "슥", "번쩍", "탕", "툭")


@lru_cache(maxsize=128)
def _font(path, size):
    return ImageFont.truetype(str(path), size)


def draw_manhwa_mask(side, rng, font_path, kind):
    large = kind == "sfx" or (kind == "detector" and rng.random() < 0.5)
    if large:
        text = rng.choice(SFX)
        if rng.random() < 0.4:
            text = "\n".join(text)
        size = max(8, round(side * rng.uniform(0.16, 0.42)))
        outline = max(1, round(size * rng.uniform(0.04, 0.13)))
    else:
        text = "\n".join(rng.choices(DIALOGUE, k=rng.randint(1, 3)))
        size = max(7, round(side * rng.uniform(0.055, 0.13)))
        outline = max(1, round(size * rng.uniform(0.03, 0.08)))

    font = _font(font_path, size)
    spacing = max(1, size // 5)
    canvas = Image.new("L", (1, 1))
    left, top, right, bottom = ImageDraw.Draw(canvas).multiline_textbbox(
        (0, 0), text, font=font, spacing=spacing, stroke_width=outline, align="center"
    )
    left, top, right, bottom = floor(left), floor(top), ceil(right), ceil(bottom)
    glyphs = Image.new("L", (right - left, bottom - top))
    ImageDraw.Draw(glyphs).multiline_text(
        (-left, -top), text, font=font, spacing=spacing, stroke_width=outline, align="center", fill=255
    )
    if large:
        glyphs = glyphs.rotate(rng.uniform(-30, 30), resample=Image.Resampling.BICUBIC, expand=True)
    glyphs = glyphs.crop(glyphs.getbbox())
    limit = max(1, round(side * rng.uniform(0.55, 0.88)))
    glyphs.thumbnail((limit, limit), Image.Resampling.LANCZOS)
    if kind == "detector":
        padding = rng.randint(1, max(2, side // 20))
        glyphs = Image.new("L", (min(side, glyphs.width + padding), min(side, glyphs.height + padding)), 255)
    else:
        glyphs = glyphs.point([0] + [255] * 255)

    mask = Image.new("L", (side, side))
    mask.paste(glyphs, (rng.randint(0, side - glyphs.width), rng.randint(0, side - glyphs.height)))
    return mask

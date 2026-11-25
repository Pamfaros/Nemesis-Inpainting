"""Crop and restore masked regions in native page coordinates."""
import numpy as np
from PIL import Image

MASK_LUT = [0] * 128 + [255] * 128


def binary_mask(mask):
    return mask.convert("L").point(MASK_LUT)


def add_tile_options(parser):
    parser.add_argument("--tile-size", type=int, default=256, help="Maximum tile side in native pixels")
    parser.add_argument("--overlap", type=int, help="Minimum overlap in pixels (default: a quarter of the tile)")


def _overlap(tile_size, overlap):
    overlap = tile_size // 4 if overlap is None else overlap
    if tile_size < 32 or not 0 <= overlap < tile_size:
        raise ValueError("Use a tile size of at least 32 and overlap smaller than the tile")
    return overlap


def _windows(length, start, stop, size, overlap):
    size = min(size, length)
    margin = overlap // 2
    if stop - start <= size - overlap:
        center = max(0, min((start + stop - size) // 2, length - size))
        return [(center, center + size)]
    first = min(max(start - margin, 0), length - size)
    last = min(max(stop + margin - size, first), length - size)
    stride = max(1, size - overlap)
    count = (last - first + stride - 1) // stride
    if not count:
        return [(first, first + size)]
    starts = [first + (last - first) * i // count for i in range(count + 1)]
    return [(begin, begin + size) for begin in starts]


def tile_rows(mask, tile_size=256, overlap=None):
    """Yield each row of active tile boxes and their cropped L masks."""
    overlap = _overlap(tile_size, overlap)
    bounds = mask.getbbox()
    if not bounds:
        return
    left, top, right, bottom = bounds
    columns = _windows(mask.width, left, right, tile_size, overlap)
    for y0, y1 in _windows(mask.height, top, bottom, tile_size, overlap):
        tiles = []
        for x0, x1 in columns:
            box = (x0, y0, x1, y1)
            holes = mask.crop(box)
            if holes.getbbox():
                tiles.append((box, holes))
        yield y0, tiles


def _weights(start, stop, length, overlap):
    ramp = np.arange(1, stop - start + 1, dtype=np.float32) / (overlap + 1)
    left = ramp if start > 0 else 1
    right = ramp[::-1] if stop < length else 1
    return np.broadcast_to(np.minimum(1, np.minimum(left, right)), ramp.shape)


def _write_band(output, accum, left, top, rows):
    band = accum[:rows]
    weights = band[..., 3]
    if weights.any():
        values = band[..., :3] / np.where(weights > 0, weights, 1)[..., None]
        pixels = (values.clip(0, 1) * 255).round().astype(np.uint8)
        covered = (weights > 0).astype(np.uint8) * 255
        output.paste(Image.fromarray(pixels), (left, top), Image.fromarray(covered))
    remaining = len(accum) - rows
    accum[:remaining] = accum[rows:].copy()
    accum[remaining:] = 0


def restore_page(image, mask, predict, *, tile_size=256, overlap=None):
    """Blend RGB predictions in the binary hole while copying all visible pixels."""
    if image.size != mask.size:
        raise ValueError("Image and mask must have the same dimensions")
    overlap = _overlap(tile_size, overlap)
    mask = binary_mask(mask)
    image = image if image.mode == "RGB" else image.convert("RGB")
    output = image.copy()
    bounds = mask.getbbox()
    if not bounds:
        return output

    left, _, right, _ = bounds
    # Only the current tile band needs floating-point color and weight buffers.
    accum = np.zeros((min(tile_size, image.height), right - left, 4), dtype=np.float32)
    previous_top = None
    for top, tiles in tile_rows(mask, tile_size, overlap):
        if previous_top is not None:
            _write_band(output, accum, left, previous_top, top - previous_top)
        previous_top = top
        for box, holes in tiles:
            x0, y0, x1, y1 = box
            start, stop = max(left, x0), min(right, x1)
            span = slice(start - x0, stop - x0)
            region = accum[:y1 - y0, start - left:stop - left]
            horizontal = _weights(x0, x1, image.width, overlap)[span]
            vertical = _weights(y0, y1, image.height, overlap)
            weight = vertical[:, None] * horizontal[None, :] * (np.asarray(holes)[:, span] > 0)
            pred = predict(image.crop(box), holes)
            region[..., :3] += pred[:, span] * weight[..., None]
            region[..., 3] += weight
    _write_band(output, accum, left, previous_top, len(accum))
    return output

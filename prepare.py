"""Prepare color inpainting patches from annotated or aligned manhwa pages."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
import urllib.request
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from manhwa_masks import MASK_TYPES, draw_manhwa_mask
from page import binary_mask, tile_rows

REVISION = "0868f943ac294af1b4ffcead8cdc1ec89ba4c4e2"
REPO = "jhc90/webtoon_text_inpainting_data"


def fingerprint(image):
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def split_for(key):
    bucket = int(key[:8], 16) % 100
    return "test" if bucket < 10 else "val" if bucket < 22 else "train"


def colored(im):
    sample = np.asarray(im.resize((64, 64))).astype(np.int16)
    return np.mean(np.ptp(sample, axis=2) > 12) >= 0.05


def draw_mask(side, rng):
    mask = Image.new("L", (side, side))
    draw = ImageDraw.Draw(mask)
    x, y = (rng.randrange(side // 8, side // 2) for _ in range(2))
    width, height = (rng.randint(side // 8, side // 2) for _ in range(2))
    if rng.random() < 0.35:
        draw.rectangle((x, y, x + width, y + height), fill=255)
    else:
        for _ in range(rng.randint(3, 9)):
            pts = [(x + rng.randrange(width), y + rng.randrange(height)) for _ in range(3)]
            draw.line(pts, fill=255, width=rng.randint(2, max(3, side // 18)), joint="curve")
    return mask


def save_row(out, rows, group, i, split, source, target, mask, image=None, **metadata):
    folder = out / split
    folder.mkdir(exist_ok=True)
    stem = f"{group[:20]}-{i:04d}"
    target.save(folder / (stem + ".png"))
    mask.save(folder / (stem + "-mask.png"))
    if image is not None:
        image.save(folder / (stem + "-input.png"))
    rows.append(dict(group=group, stem=stem, split=split, paired=image is not None, source=source, **metadata))


def pages(out, args):
    rows, seen = [], set()
    font = getattr(args, "hangul_font", None)
    if font is not None and not font.is_file():
        raise ValueError(f"Hangul font not found: {font}")
    # Process held-out archive first so an identical page cannot enter training.
    for archive, heldout in ((args.test_zip, True), (args.train_zip, False)):
        with zipfile.ZipFile(archive) as z:
            entries = set(z.namelist())
            for name in sorted(n for n in entries if n.lower().endswith((".jpeg", ".jpg", ".png")) and n.startswith("images/")):
                with Image.open(io.BytesIO(z.read(name))) as raw:
                    page = raw.convert("RGB")
                if font is not None and min(page.size) < args.size:
                    continue
                group = fingerprint(page)
                if group in seen:
                    continue
                seen.add(group)
                label = "labels/" + Path(name).stem + ".txt"
                if label not in entries:
                    continue
                boxes = []
                for line in z.read(label).decode("utf-8-sig").splitlines():
                    if not line.strip():
                        continue
                    vals = [float(v) for v in line.split()[1:]]
                    if len(vals) == 4:
                        cx, cy, w, h = vals
                        vals = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
                    if len(vals) < 4 or len(vals) % 2:
                        raise ValueError("Malformed label: " + label)
                    xs, ys = vals[::2], vals[1::2]
                    boxes.append((min(xs)*page.width, min(ys)*page.height,
                                  max(xs)*page.width, max(ys)*page.height))
                split = "test" if heldout else ("val" if int(group[:8], 16) % 6 == 0 else "train")
                rng = random.Random(group)
                kept = 0
                for _ in range(args.per_page * 100):
                    side = args.size if font is not None else min(page.width, page.height, rng.randint(args.size, 3*args.size))
                    x, y = rng.randint(0, page.width-side), rng.randint(0, page.height-side)
                    if any(x < r and x+side > l and y < b and y+side > t for l,t,r,b in boxes):
                        continue
                    box = (x, y, x+side, y+side)
                    crop = page.crop(box)
                    if font is None:
                        crop = crop.resize((args.size, args.size), Image.Resampling.LANCZOS)
                        if not colored(crop) or np.asarray(crop).std(axis=(0,1)).mean() < 12:
                            continue
                        mask = draw_mask(args.size, rng)
                        metadata = {}
                    else:
                        if np.asarray(crop).std(axis=(0, 1)).mean() < 8:
                            continue
                        kind = MASK_TYPES[kept % len(MASK_TYPES)]
                        mask = draw_manhwa_mask(side, rng, font, kind)
                        metadata = {"mask_type": kind, "box": box}
                    save_row(out, rows, group, kept, split, Path(archive).name + "/" + name, crop, mask, **metadata)
                    kept += 1
                    if kept >= args.per_page:
                        break
    return rows


def read_pairs(manifest):
    """Read aligned pair records and prevent a series from crossing splits."""
    rows, splits = [], {}
    fields = ("series", "split", "input", "target", "mask")
    with manifest.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or any(not isinstance(row.get(field), str) or not row[field].strip() for field in fields):
                raise ValueError(f"{manifest}:{number} requires {', '.join(fields)}")
            series, split = row["series"].strip(), row["split"]
            if split not in ("train", "val", "test"):
                raise ValueError(f"{manifest}:{number} has an invalid split: {split}")
            if splits.setdefault(series, split) != split:
                raise ValueError(f"Series appears in multiple splits: {series}")
            row["series"] = series
            rows.append(row)
    return rows


def _pad_patch(image, size):
    width, height = image.size
    if (width, height) == (size, size):
        return image
    padding = [(0, size - height), (0, size - width)]
    if image.mode == "L":
        return Image.fromarray(np.pad(np.asarray(image), padding, mode="constant"))
    return Image.fromarray(np.pad(np.asarray(image), [*padding, (0, 0)], mode="edge"))


def paired_pages(out, args):
    rows, splits = [], {}
    root = args.manifest.parent
    for pair in read_pairs(args.manifest):
        with Image.open(root / pair["input"]) as raw_input, Image.open(root / pair["target"]) as raw_target, Image.open(root / pair["mask"]) as raw_mask:
            if raw_input.size != raw_target.size or raw_input.size != raw_mask.size:
                raise ValueError(f"Unaligned page pair: {pair['input']}")
            image, target, mask = raw_input.convert("RGB"), raw_target.convert("RGB"), binary_mask(raw_mask)
        if not mask.getbbox():
            raise ValueError(f"Empty mask: {pair['mask']}")
        group, split = fingerprint(target), pair["split"]
        if splits.setdefault(group, split) != split:
            raise ValueError(f"Clean page appears in multiple splits: {pair['target']}")
        for _, tiles in tile_rows(mask, args.size):
            for box, holes in tiles:
                save_row(out, rows, group, len(rows), split, pair["input"],
                         _pad_patch(target.crop(box), args.size), _pad_patch(holes, args.size),
                         _pad_patch(image.crop(box), args.size), series=pair["series"],
                         mask_type=pair.get("group"), box=box)
    return rows


def write_patch_pairs(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stem = f"{row['split']}/{row['stem']}"
            record = dict(series=row["series"], split=row["split"], group=row["mask_type"],
                          input=stem + "-input.png", target=stem + ".png", mask=stem + "-mask.png")
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def download_shard(directory, index):
    import pyarrow.parquet as pq
    directory.mkdir(parents=True, exist_ok=True)
    name = f"train-{index:05d}-of-00050.parquet"
    dest = directory / name
    if dest.exists():
        pq.ParquetFile(dest).close()
        return dest
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/{name}"
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as stream:
        tmp = Path(stream.name)
    try:
        with urllib.request.urlopen(url, timeout=45) as resp, tmp.open("wb") as f:
            while chunk := resp.read(1024*1024):
                f.write(chunk)
        pq.ParquetFile(tmp).close()
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def hf_pairs(out, args):
    import pyarrow.parquet as pq
    rows, seen = [], set()
    for path in args.parquet or (download_shard(args.cache, i) for i in range(args.shards)):
        with pq.ParquetFile(path) as table:
            for batch in table.iter_batches(batch_size=4, columns=["input_image","output_image","mask_image"]):
                for row in batch.to_pylist():
                    def decode(key, mode):
                        with Image.open(io.BytesIO(row[key]["bytes"])) as im:
                            return im.convert(mode)
                    target = decode("output_image","RGB")
                    image = decode("input_image","RGB")
                    mask = decode("mask_image","L").point(lambda v: 255 if v >= 128 else 0)
                    if target.size != mask.size or image.size != target.size:
                        raise ValueError("Unaligned pair in " + str(path))
                    box = mask.getbbox()
                    if not box or not colored(target):
                        continue
                    group = fingerprint(target)
                    if group in seen:
                        continue
                    seen.add(group)
                    l,t,r,b = box
                    side = max(args.size, 2*max(r-l,b-t))
                    # Clip each axis before resizing; retain original aspect with padding.
                    side = min(side, max(target.size))
                    x = min(max((l+r-side)//2,0), max(0,target.width-side))
                    y = min(max((t+b-side)//2,0), max(0,target.height-side))
                    bounds = (x,y,min(x+side,target.width),min(y+side,target.height))
                    scale = args.size / max(bounds[2]-x, bounds[3]-y)
                    dims = (max(1,round((bounds[2]-x)*scale)), max(1,round((bounds[3]-y)*scale)))
                    items = []
                    for im in (target, mask, image):
                        filt = Image.Resampling.NEAREST if im.mode=="L" else Image.Resampling.LANCZOS
                        patch = im.crop(bounds).resize(dims, filt)
                        arr = np.asarray(patch)
                        padding = [(0,args.size-dims[1]), (0,args.size-dims[0])] + ([(0,0)] if arr.ndim==3 else [])
                        items.append(Image.fromarray(np.pad(arr, padding, mode="constant" if im.mode=="L" else "edge")))
                    if items[1].getbbox():
                        save_row(out,rows,group,0,split_for(group),Path(path).name,*items)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True,type=Path)
    p.add_argument("--size",type=int,default=160)
    sub = p.add_subparsers(dest="mode",required=True)
    local = sub.add_parser("pages")
    local.add_argument("--train-zip",required=True,type=Path)
    local.add_argument("--test-zip",required=True,type=Path)
    local.add_argument("--per-page",type=int,default=20)
    local.add_argument("--hangul-font", type=Path, help="Make native-scale crops with Hangul dialogue and SFX masks")
    pairs = sub.add_parser("pairs", help="Crop aligned lettered, clean and mask pages from JSONL")
    pairs.add_argument("--manifest", required=True, type=Path)
    hf = sub.add_parser("hf")
    hf.add_argument("--parquet",nargs="+",type=Path)
    hf.add_argument("--shards",type=int,choices=range(1,51),default=1)
    hf.add_argument("--cache",type=Path,default=Path("data/raw"))
    args = p.parse_args()
    if args.size < 32 or (args.mode=="pages" and args.per_page < 1):
        p.error("size must be >=32 and per-page must be positive")
    if args.out.exists():
        p.error("Choose a new output directory")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        out = Path(tmp) / "prepared"
        out.mkdir()
        rows = {"pages": pages, "pairs": paired_pages, "hf": hf_pairs}[args.mode](out, args)
        if not rows:
            raise ValueError("No usable color patches")
        (out/"manifest.json").write_text(json.dumps(rows),encoding="utf-8")
        if args.mode == "pairs":
            write_patch_pairs(out / "pairs.jsonl", rows)
        out.rename(args.out)
    for split in ("train","val","test"):
        selected = [r for r in rows if r["split"]==split]
        print(split, len(selected), "patches,",len({r["group"] for r in selected}),"distinct pages")

if __name__ == "__main__":
    main()


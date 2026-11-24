"""Prepare independent color patches from jhc90 pairs or annotated local page archives."""
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


def save_row(out, rows, group, i, split, source, target, mask, image=None):
    folder = out / split
    folder.mkdir(exist_ok=True)
    stem = f"{group[:20]}-{i:04d}"
    target.save(folder / (stem + ".png"))
    mask.save(folder / (stem + "-mask.png"))
    if image is not None:
        image.save(folder / (stem + "-input.png"))
    rows.append(dict(group=group, stem=stem, split=split, paired=image is not None, source=source))


def pages(out, args):
    rows, seen = [], set()
    # Process held-out archive first so an identical page cannot enter training.
    for archive, heldout in ((args.test_zip, True), (args.train_zip, False)):
        with zipfile.ZipFile(archive) as z:
            entries = set(z.namelist())
            for name in sorted(n for n in entries if n.lower().endswith((".jpeg", ".jpg", ".png")) and n.startswith("images/")):
                with Image.open(io.BytesIO(z.read(name))) as raw:
                    page = raw.convert("RGB")
                group = fingerprint(page)
                if group in seen:
                    continue
                seen.add(group)
                label = "labels/" + Path(name).stem + ".txt"
                if label not in entries:
                    continue
                boxes = []
                for line in z.read(label).decode("utf-8-sig").splitlines():
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
                    side = min(page.width, page.height, rng.randint(args.size, 3*args.size))
                    x, y = rng.randint(0, page.width-side), rng.randint(0, page.height-side)
                    if any(x < r and x+side > l and y < b and y+side > t for l,t,r,b in boxes):
                        continue
                    crop = page.crop((x, y, x+side, y+side)).resize((args.size, args.size), Image.Resampling.LANCZOS)
                    if not colored(crop) or np.asarray(crop).std(axis=(0,1)).mean() < 12:
                        continue
                    save_row(out, rows, group, kept, split, Path(archive).name + "/" + name,
                             crop, draw_mask(args.size, rng))
                    kept += 1
                    if kept >= args.per_page:
                        break
    return rows


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
        rows = (pages if args.mode=="pages" else hf_pairs)(out,args)
        if not rows:
            raise ValueError("No usable color patches")
        (out/"manifest.json").write_text(json.dumps(rows),encoding="utf-8")
        out.rename(args.out)
    for split in ("train","val","test"):
        selected = [r for r in rows if r["split"]==split]
        print(split, len(selected), "patches,",len({r["group"] for r in selected}),"distinct pages")

if __name__ == "__main__":
    main()


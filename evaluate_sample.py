"""Evaluate artificial holes on a separate directory of decoded source images."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw
import torch

from prepare import colored, draw_mask, fingerprint
from model import NemesisUNet, context_fill
from train import tensor, evaluate


@torch.inference_mode()
def external(images, weights, out, device):
    if not images.is_dir():
        raise ValueError("Provide a directory of source JPEGs")
    net = NemesisUNet().to(device).eval()
    net.load_state_dict(torch.load(weights,map_location=device,weights_only=True)["model"])
    side = 160
    rows = []
    seen = set()
    examples = []
    audit = []
    for path in sorted(images.glob("*.jpg")):
        with Image.open(path) as im:
            page = im.convert("RGB")
        page_id = fingerprint(page)
        if page_id in seen or not colored(page) or min(page.size)<side:
            continue
        seen.add(page_id)
        rng = random.Random(page_id)
        count = 0
        for _ in range(100):
            edge = min(min(page.size), rng.randint(side,3*side))
            x, y = rng.randint(0,page.width-edge),rng.randint(0,page.height-edge)
            target = page.crop((x,y,x+edge,y+edge)).resize((side,side),Image.Resampling.LANCZOS)
            if not colored(target) or np.asarray(target).std(axis=(0,1)).mean()<12:
                continue
            mask = tensor(draw_mask(side,rng))
            clean = tensor(target)
            rgb = clean * (1-mask)
            rows.append((rgb,mask,clean))
            audit.append(dict(source=path.name,sha256=page_id,box=[x,y,x+edge,y+edge]))
            if count==0 and len(examples)<4:
                def to_image(a):
                    return Image.fromarray((a.clamp(0,1)*255).round().byte().cpu().permute(1,2,0).numpy())
                x0, m0 = rgb[None].to(device), mask[None].to(device)
                examples.append([to_image(rgb),to_image(context_fill(x0,m0)[0]),
                                 to_image(net(x0,m0)[0]),target])
            count += 1
            if count==3:
                break
    if not rows:
        raise ValueError("No usable color crops")
    out.mkdir(parents=True,exist_ok=True)
    report = dict(source_directory=images.name,images=len({v["sha256"] for v in audit}),
                  patches=len(rows),checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                  **evaluate(net,torch.utils.data.DataLoader(rows,batch_size=6),device))
    (out/"external.json").write_text(json.dumps(dict(**report,crops=audit),indent=2))
    sheet=Image.new("RGB",(side*4,(side+3)*len(examples)+24),"white")
    draw=ImageDraw.Draw(sheet)
    for c,title in enumerate(["Masked input","Context fill","Nemesis U-Net","Original crop"]):
        draw.text((side*c+4,5),title,fill="black")
    for r,example in enumerate(examples):
        for c,img in enumerate(example):
            sheet.paste(img,(side*c,24+r*(side+3)))
    sheet.save(out/"comparison.png")
    print(json.dumps(report))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("images",type=Path)
    p.add_argument("weights",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    a=p.parse_args()
    external(a.images,a.weights,a.out,a.device)


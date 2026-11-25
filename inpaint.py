import argparse
from pathlib import Path
from PIL import Image
import torch
from model.registry import load_model
from train import tensor


@torch.inference_mode()
def restore(weights, input_path, mask_path, output, device):
    with Image.open(input_path) as im, Image.open(mask_path) as m:
        if im.size != m.size:
            raise ValueError("Image and mask must have the same dimensions")
        image = im.convert("RGB")
        mask = m.convert("L").point(lambda v: 255 if v >= 128 else 0)
    net = load_model(weights, device)
    net.load_state_dict(torch.load(weights,map_location=device,weights_only=True)["model"])
    bounds = mask.getbbox()
    if bounds:
        l,t,r,b = bounds
        box = (max(0,l-96),max(0,t-96),min(image.width,r+96),min(image.height,b+96))
        rgb = tensor(image.crop(box))[None].to(device)
        area = tensor(mask.crop(box))[None].to(device)
        pred = net(rgb,area)[0].permute(1,2,0).mul(255).round().byte().cpu().numpy()
        image.paste(Image.fromarray(pred),box,mask.crop(box))
    image.save(output,format="PNG")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for arg in ("weights","image","mask","output"):
        p.add_argument(arg,type=Path)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=p.parse_args()
    restore(args.weights,args.image,args.mask,args.output,args.device)


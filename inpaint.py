"""Restore masked regions on a native-resolution RGB page."""
import argparse
from pathlib import Path
from PIL import Image
import torch
from model.registry import load_model
from page import add_tile_options, restore_page
from train import tensor


@torch.inference_mode()
def restore(weights, input_path, mask_path, output, device, *, tile_size=256, overlap=None):
    net = load_model(weights, device)

    def predict(crop, holes):
        rgb = tensor(crop)[None].to(device)
        area = tensor(holes)[None].to(device)
        return net(rgb, area)[0].permute(1, 2, 0).cpu().numpy()

    with Image.open(input_path) as image, Image.open(mask_path) as mask:
        result = restore_page(image, mask, predict, tile_size=tile_size, overlap=overlap)
    result.save(output, format="PNG")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("weights","image","mask","output"):
        p.add_argument(arg,type=Path)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    add_tile_options(p)
    args=p.parse_args()
    restore(args.weights, args.image, args.mask, args.output, args.device,
            tile_size=args.tile_size, overlap=args.overlap)


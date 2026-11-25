"""Train and compare ROI models or restore a masked RGB page."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import ManhwaUNet, restore, training_loss
from page import add_tile_options, restore_page
from prepare import read_pairs


def image_tensor(image):
    data = torch.from_numpy(np.array(image))
    if image.mode == "L":
        return (data >= 128).unsqueeze(0)
    return data.permute(2, 0, 1).float() / 255


def read_image(path, mode):
    with Image.open(path) as image:
        return image_tensor(image.convert(mode))


class Pairs(Dataset):
    def __init__(self, manifest, split):
        self.root = manifest.parent
        self.rows = [row for row in read_pairs(manifest) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"empty split: {split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        x = read_image(self.root / row["input"], "RGB")
        y = read_image(self.root / row["target"], "RGB")
        m = read_image(self.root / row["mask"], "L")
        if x.shape != y.shape or x[:1].shape != m.shape or not m.any():
            raise ValueError(f"unaligned pair or empty mask: {row['input']}")
        return x, m, y


def train(args):
    if args.epochs < 1 or args.batch < 1:
        raise ValueError("epochs and batch must be positive")
    torch.manual_seed(args.seed)
    net = ManhwaUNet(args.mode).to(args.device)
    pairs = Pairs(args.manifest, "train")
    # Separate generator keeps the sample order identical for both objectives.
    order = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(pairs, batch_size=args.batch, shuffle=True, generator=order)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    updates = 0
    for epoch in range(args.epochs):
        total = 0.
        for x, m, y in loader:
            if x.shape[-2:] != (256, 256):
                raise ValueError("prepare 256x256 training ROIs at native pixel scale")
            opt.zero_grad(set_to_none=True)
            loss = training_loss(net, x.to(args.device), m.to(args.device), y.to(args.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
            total += loss.item() * len(x)
            updates += 1
        print(f"epoch={epoch+1} updates={updates} loss={total / len(pairs):.6f}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_suffix(args.output.suffix + ".tmp")
        torch.save({"mode": net.mode, "weights": net.state_dict(), "updates": updates}, temp)
        temp.replace(args.output)


def load(args):
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    net = ManhwaUNet(checkpoint["mode"])
    net.load_state_dict(checkpoint["weights"])
    return net.to(args.device).eval()


@torch.inference_mode()
def infer(args):
    net = load(args)
    routes = []

    def predict(crop, holes):
        rgb = image_tensor(crop)[None].to(args.device)
        mask = image_tensor(holes)[None].to(args.device)
        prediction, route = restore(net, rgb, mask, args.steps, args.seed + len(routes), not args.no_router)
        routes.append(route)
        return prediction[0].permute(1, 2, 0).cpu().numpy()

    with Image.open(args.input) as image, Image.open(args.mask) as mask:
        result = restore_page(image, mask, predict, tile_size=args.tile_size, overlap=args.overlap)
    result.save(args.output, format="PNG")
    route = routes[0] if routes else "identity"
    if len(set(routes)) > 1:
        route = "mixed"
    print(f"route={route} tiles={len(routes)}")


def evaluate(args):
    net = load(args)
    pairs = Pairs(args.manifest, args.split)
    for i, (x, mask, clean) in enumerate(pairs):
        x, m = x[None].to(args.device), mask[None].to(args.device)
        pred, route = restore(net, x, m, args.steps, args.seed + i, not args.no_router)
        err = (pred[0].cpu() - clean)[:, mask[0]]
        mse = err.square().mean().item()
        changed = (pred != x).any(dim=1, keepdim=True) & ~m
        row = pairs.rows[i]
        print(json.dumps(dict(input=row["input"], series=row["series"], group=row.get("group"),
                              route=route, seed=args.seed + i, pixels=mask.sum().item(),
                              masked_mae=err.abs().mean().item(), masked_mse=mse,
                              masked_psnr_db=-10 * math.log10(mse) if mse else None,
                              outside_changed=changed.sum().item()), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train_cmd = sub.add_parser("train")
    train_cmd.add_argument("--manifest", type=Path, required=True)
    train_cmd.add_argument("--mode", choices=("rgb", "diffusion"), required=True)
    train_cmd.add_argument("--output", type=Path, required=True)
    train_cmd.add_argument("--epochs", type=int, default=50)
    train_cmd.add_argument("--batch", type=int, default=4)
    train_cmd.add_argument("--lr", type=float, default=2e-4)
    infer_cmd = sub.add_parser("infer")
    add_tile_options(infer_cmd)
    for key in ("input", "mask", "output"):
        infer_cmd.add_argument("--" + key, type=Path, required=True)
    eval_cmd = sub.add_parser("eval")
    eval_cmd.add_argument("--manifest", type=Path, required=True)
    eval_cmd.add_argument("--split", choices=("val", "test"), default="val")
    for cmd in (infer_cmd, eval_cmd):
        cmd.add_argument("--weights", type=Path, required=True)
        cmd.add_argument("--steps", type=int, choices=(4, 8, 16), default=8)
        cmd.add_argument("--no-router", action="store_true")
    for cmd in (train_cmd, infer_cmd, eval_cmd):
        cmd.add_argument("--seed", type=int, default=27)
        cmd.add_argument("--device", default="cpu")
    args = parser.parse_args()
    {"train": train, "infer": infer, "eval": evaluate}[args.command](args)


if __name__ == "__main__":
    main()

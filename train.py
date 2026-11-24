import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from model import NemesisUNet, context_fill, objective


def tensor(image):
    a = np.array(image, copy=True)
    if a.ndim == 2:
        a = a[..., None]
    return torch.from_numpy(a).permute(2, 0, 1).float().div(255)


class Patches(Dataset):
    def __init__(self, root, split, rows):
        self.rows = [r for r in rows if r["split"] == split]
        self.root = root / split
        if not self.rows:
            raise ValueError("No examples in split " + split)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        def read(suffix, mode):
            with Image.open(self.root / (row["stem"]+suffix+".png")) as im:
                return tensor(im.convert(mode))
        target = read("", "RGB")
        mask = (read("-mask","L") >= 0.5).float()
        image = read("-input","RGB") if row["paired"] else target * (1-mask)
        return image, mask, target


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    sums = torch.zeros(5, device=device, dtype=torch.float64)
    for rgb, mask, target in loader:
        rgb, mask, target = (v.to(device) for v in (rgb,mask,target))
        err = (model(rgb, mask)-target) * mask
        ref = (context_fill(rgb,mask)-target) * mask
        sums += torch.stack([err.abs().sum(), err.square().sum(),
                             ref.abs().sum(), ref.square().sum(), 3*mask.sum()])
    abs_err, sq, base_abs, base_sq, n = sums.tolist()
    if n == 0:
        raise ValueError("Evaluation requires nonempty masks")
    return dict(masked_mae=abs_err/n, masked_psnr_db=-10*math.log10(max(sq/n,1e-12)),
                baseline_mae=base_abs/n, baseline_psnr_db=-10*math.log10(max(base_sq/n,1e-12)))


def atomic_save(obj, path):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
    try:
        torch.save(obj, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def train(args):
    if min(args.epochs,args.batch) < 1 or args.lr <= 0:
        raise ValueError("Use positive epochs, batch size and learning rate")
    args.run.mkdir(parents=True,exist_ok=True)
    latest = args.run/"last.pt"
    if not args.resume and any(args.run.iterdir()):
        raise ValueError("Use an empty run directory or --resume")
    manifest = (args.data/"manifest.json").read_bytes()
    digest = hashlib.sha256(manifest).hexdigest()
    rows = json.loads(manifest)
    groups = {}
    for r in rows:
        if groups.setdefault(r["group"],r["split"]) != r["split"]:
            raise ValueError("Page crosses dataset splits")
    train_set, val_set, test_set = [Patches(args.data,split,rows) for split in ("train","val","test")]
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type=="cpu":
        torch.set_num_threads(min(4,torch.get_num_threads()))
    model = NemesisUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr)
    start, best, history = 0, float("inf"), []
    settings = dict(digest=digest,seed=args.seed,batch=args.batch,lr=args.lr)
    if args.resume:
        state = torch.load(latest,map_location="cpu",weights_only=True)
        if state["settings"] != settings:
            raise ValueError("Resume with the original dataset, seed, LR and batch size")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best, history = state["epoch"], state["best"], state["history"]
    val = DataLoader(val_set,batch_size=args.batch)
    for epoch in range(start+1,args.epochs+1):
        gen = torch.Generator().manual_seed(args.seed+epoch)
        batches = DataLoader(train_set,batch_size=args.batch,shuffle=True,generator=gen)
        model.train()
        loss_sum = 0.0
        for rgb, mask, target in batches:
            rgb, mask, target = (v.to(device) for v in (rgb,mask,target))
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(rgb,mask),target,mask)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            loss_sum += loss.item()*len(rgb)
        score = evaluate(model,val,device)
        record = dict(epoch=epoch,loss=loss_sum/len(train_set),**score)
        history.append(record)
        if score["masked_mae"] < best:
            best = score["masked_mae"]
            atomic_save(dict(model=model.state_dict(),epoch=epoch),args.run/"best.pt")
        atomic_save(dict(settings=settings,model=model.state_dict(),optimizer=optimizer.state_dict(),
                         epoch=epoch,best=best,history=history),latest)
        print(json.dumps(record),flush=True)
    state = torch.load(args.run/"best.pt",map_location=device,weights_only=True)
    model.load_state_dict(state["model"])
    test = evaluate(model,DataLoader(test_set,batch_size=args.batch),device)
    summary = dict(best_epoch=state["epoch"],device=str(device),torch=str(torch.__version__),
                   parameters=sum(p.numel() for p in model.parameters()),settings=settings,
                   patches=dict(train=len(train_set),val=len(val_set),test=len(test_set)),
                   history=history,test=test)
    (args.run/"metrics.json").write_text(json.dumps(summary,indent=2))
    print("Test",test,flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=6)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--seed",type=int,default=27)
    p.add_argument("--lr",type=float,default=0.0005)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume",action="store_true")
    train(p.parse_args())


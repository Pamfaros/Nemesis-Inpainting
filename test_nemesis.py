import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from inpaint import restore
from model import NemesisUNet, context_fill, objective
from prepare import hf_pairs
from train import Patches, train


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_input_under_mask_cannot_leak(self):
        rgb = torch.rand(1,3,31,37)
        mask = torch.zeros_like(rgb[:,:1])
        mask[:,:,4:18,9:24] = 1
        net = NemesisUNet(width=8)
        first = net(rgb,mask)
        second = net(torch.where(mask.bool(),torch.randn_like(rgb),rgb),mask)
        self.assertTrue(torch.equal(first,second))
        self.assertTrue(torch.equal(first*(1-mask),rgb*(1-mask)))
        objective(first,rgb,mask).backward()
        self.assertGreater(net.head.weight.grad.abs().sum(),0)

    def test_empty_and_full_masks(self):
        rgb = torch.rand(1,3,1,13)
        empty = torch.zeros_like(rgb[:,:1])
        self.assertTrue(torch.equal(context_fill(rgb,empty),rgb))
        self.assertTrue(torch.allclose(context_fill(rgb,empty+1),torch.full_like(rgb,0.5)))

    def test_public_pair_import_and_duplicate_grouping(self):
        def encode(image):
            buf=io.BytesIO()
            image.save(buf,format="PNG")
            return {"bytes":buf.getvalue(),"path":None}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = Image.new("RGB",(64,90),(20,90,230))
            dirty = clean.copy()
            dirty.paste("red",(35,40,50,60))
            mask = Image.new("L",clean.size)
            mask.paste(255,(35,40,50,60))
            row=dict(input_image=encode(dirty),output_image=encode(clean),mask_image=encode(mask))
            shard=root/"fixture.parquet"
            pq.write_table(pa.Table.from_pylist([row,row]),shard)
            rows=hf_pairs(root, SimpleNamespace(parquet=[shard],size=32))
            self.assertEqual(len(rows),1)
            image, area, target=Patches(root,rows[0]["split"],rows)[0]
            self.assertGreater((image-target).abs().sum(),0)
            self.assertGreater(area.sum(),0)
            self.assertGreater(target[2].mean(),0.8)

    def test_train_resume_and_lossless_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[]
            img = Image.fromarray(np.random.default_rng(8).integers(0,256,(32,32,3),dtype=np.uint8))
            mask=Image.new("L",img.size)
            mask.paste(255,(8,12,19,22))
            for split in ("train","val","test"):
                folder=root/split
                folder.mkdir()
                img.save(folder/"a.png")
                mask.save(folder/"a-mask.png")
                rows.append(dict(split=split,stem="a",group=split,paired=False))
            (root/"manifest.json").write_text(json.dumps(rows))
            args=SimpleNamespace(data=root,run=root/"run",epochs=1,batch=1,lr=.0005,seed=1,resume=False,device="cpu")
            train(args)
            args.resume=True
            args.epochs=2
            train(args)
            self.assertEqual(len(json.loads((args.run/"metrics.json").read_text())["history"]),2)
            restore(args.run/"best.pt",root/"val/a.png",root/"val/a-mask.png",root/"out.png","cpu")
            with Image.open(root/"out.png") as out:
                keep=np.array(mask)==0
                np.testing.assert_array_equal(np.array(img)[keep],np.array(out)[keep])


if __name__=="__main__":
    unittest.main()


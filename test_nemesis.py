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
from model.model_hybrid import (
    NemesisUNet,
    context_fill,
    objective
)
from model.model_baseline import manhwa_objective
from model.registry import build_model, initialize_model
from page import restore_page
from prepare import hf_pairs, paired_pages
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

    def test_manhwa_transfer_and_masked_contours(self):
        base_config = {"architecture": "baseline", "width": 8}
        manhwa_config = {**base_config, "architecture": "manhwa"}
        baseline, manhwa = build_model(base_config), build_model(manhwa_config)
        with torch.no_grad():
            baseline.head.weight.normal_(0, 0.01)
        initialize_model(manhwa, manhwa_config, {"model_config": base_config, "model": baseline.state_dict()})
        rgb = torch.rand(1, 3, 33, 37)
        mask = torch.zeros_like(rgb[:, :1], dtype=torch.bool)
        mask[..., 9:24, 3:25] = True
        prediction = manhwa(rgb, mask)
        torch.testing.assert_close(prediction, baseline(rgb, mask))
        loss = manhwa_objective(prediction, rgb, mask)
        loss.backward()
        contour_grad = manhwa.entry[0].weight.grad[:, 7:]
        self.assertTrue(contour_grad.isfinite().all())
        self.assertGreater(contour_grad.abs().sum(), 0)
        with torch.no_grad():
            manhwa.entry[0].weight[:, 7:].normal_(0, 0.01)
            dirty = torch.where(mask, 1 - rgb, rgb)
            self.assertTrue(torch.equal(manhwa(dirty, mask), manhwa(rgb, mask)))

    def test_page_tiles_cover_long_and_disconnected_masks(self):
        pixels = np.random.default_rng(12).integers(0, 256, (671, 143, 3), dtype=np.uint8)
        area = np.zeros(pixels.shape[:2], dtype=np.uint8)
        area[24:490, 61:104] = 255
        area[-5:, :10] = 128
        area[0, -1] = 255
        area[2, 1] = 127
        tiles = []

        def predict(crop, mask):
            self.assertIsNotNone(mask.getbbox())
            tiles.append(crop.size)
            return np.full((crop.height, crop.width, 3), .75, dtype=np.float32)

        image = Image.fromarray(pixels)
        pred = np.asarray(restore_page(image, Image.fromarray(area), predict, tile_size=64))
        self.assertGreater(len(tiles), 3)
        self.assertTrue(all(max(size) <= 64 for size in tiles))
        np.testing.assert_array_equal(pred[area < 128], pixels[area < 128])
        self.assertTrue(np.all(pred[area >= 128] == 191))
        with self.assertRaisesRegex(ValueError, "same dimensions"):
            restore_page(image, Image.new("L", (1, 1)), predict)

    def test_page_tiles_blend_overlapping_predictions(self):
        image = Image.new("RGB", (45, 32))
        mask = Image.new("L", image.size, 255)
        colors = iter((0.0, 1.0))

        def predict(crop, holes):
            return np.full((crop.height, crop.width, 3), next(colors), dtype=np.float32)

        row = np.asarray(restore_page(image, mask, predict, tile_size=32))[10, :, 0]
        self.assertEqual((row[0], row[-1]), (0, 255))
        self.assertTrue(0 < row[22] < 255)
        self.assertTrue(np.all(row[1:] >= row[:-1]))

    def test_prepare_aligned_pages_at_native_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pixels = np.random.default_rng(3).integers(0, 256, (105, 19, 3), dtype=np.uint8)
            target = Image.fromarray(pixels)
            dirty = target.copy()
            dirty.paste("white", (3, 3, 16, 89))
            mask = Image.new("L", target.size)
            mask.paste(255, (3, 3, 16, 89))
            for image, name in ((target, "target"), (dirty, "input"), (mask, "mask")):
                image.save(root / (name + ".png"))
            manifest = root / "pairs.jsonl"
            row = dict(series="work", split="train", group="sfx", input="input.png", target="target.png", mask="mask.png")
            manifest.write_text(json.dumps(row), encoding="utf-8")
            out = root / "out"
            out.mkdir()
            args = SimpleNamespace(manifest=manifest, size=32)
            rows = paired_pages(out, args)
            self.assertGreater(len(rows), 1)
            dataset = Patches(out, "train", rows)
            for (_, holes, patch), record in zip(dataset, rows):
                self.assertEqual(record["series"], "work")
                self.assertEqual(record["mask_type"], "sfx")
                self.assertEqual(tuple(patch.shape), (3, 32, 32))
                self.assertFalse(holes[..., 19:].any())
                left, top, right, bottom = record["box"]
                restored = (patch[:, :, :19] * 255).round().byte().permute(1, 2, 0).numpy()
                np.testing.assert_array_equal(restored, pixels[top:bottom, left:right])
            manifest.write_text(json.dumps(row) + "\n" + json.dumps({**row, "split": "test"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Series appears in multiple splits"):
                paired_pages(out, args)

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
    
    def test_hybrid_architectures(self):

        for attention in (False, True):
            for ssm in (False, True):

                with self.subTest(
                    attention=attention,
                    ssm=ssm
                ):

                    net = NemesisUNet(
                        width=8,
                        context_dim=32,
                        heads=4,
                        kv_size=4,
                        use_attention=attention,
                        use_ssm=ssm
                    )

                    # Avoid trivial identity of the zero-init head
                    with torch.no_grad():
                        net.head.weight.normal_(0, 0.001)

                    rgb = torch.rand(1, 3, 63, 79)

                    for kind in ("empty", "partial", "full"):

                        mask = torch.zeros(1, 1, 63, 79)

                        if kind == "partial":
                            mask[:, :, 15:40, 20:50] = 1

                        elif kind == "full":
                            mask.fill_(1)

                        output = net(rgb, mask)

                        self.assertEqual(
                            output.shape, rgb.shape
                        )

                        self.assertTrue(
                            torch.isfinite(output).all()
                        )

                        visible = ~mask.bool().expand_as(rgb)

                        self.assertTrue(
                            torch.equal(
                                output[visible],
                                rgb[visible]
                            )
                        )

                        changed = torch.where(
                            mask.bool(),
                            torch.rand_like(rgb),
                            rgb
                        )

                        second = net(changed, mask)

                        self.assertTrue(
                            torch.allclose(output, second)
                        )

                        net.zero_grad(set_to_none=True)

                        loss = objective(output, rgb, mask)

                        loss.backward()

                        for parameter in net.parameters():
                            if parameter.grad is not None:
                                self.assertTrue(
                                    torch.isfinite(
                                        parameter.grad
                                    ).all()
                                )



if __name__=="__main__":
    unittest.main()


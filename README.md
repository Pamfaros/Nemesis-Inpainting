# NemesisInpainting

NemesisInpainting is a PyTorch inpainting module for TypeR-Studio Nemesis Edition. It restores artwork underneath masked lettering in color manhwa, including dialogue and stylized sound effects (SFX). The main checkpoint, `manhwa`, uses a deterministic residual U-Net and processes long webtoon pages at their original resolution.

An editor or a text detector supplies the mask. Include outlines, shadows, and antialiased lettering in the mask. For dialogue inside a speech balloon, keep the balloon border visible so the model can use it as context.

## Quick start

Run these commands from the repository root. The training run was verified on Windows with Python 3.10, PyTorch 2.5.1 with CUDA 12.1, and a GeForce RTX 4080 Laptop GPU.

```powershell
python -m pip install -r requirements.txt
python inpaint.py runs/manhwa_native/best.pt page.png mask.png restored.png --device cpu
```

Place the trained checkpoint at `runs/manhwa_native/best.pt` or pass its actual path. Use `--device cuda` with a CUDA-enabled PyTorch installation. Training data and run outputs are local artifacts under `data/` and `runs/`.

| Argument | Format and behavior |
|---|---|
| `page.png` | An image of the entire page or a crop. The CLI converts it to 8-bit RGB. |
| `mask.png` | An image with the same width and height as the page. After grayscale conversion, values 128–255 mark pixels to restore. Values 0–127 mark pixels to keep. |
| `restored.png` | An RGB PNG at the original width and height. The CLI writes RGB pixels and leaves alpha and source metadata out of the output. |
| `--tile-size` | Maximum tile side in native pixels. Default 256, minimum 32. |
| `--overlap` | Minimum overlap between adjacent tiles, in pixels. Defaults to one quarter of the tile size. Valid values range from zero to one less than the tile size. |

## How the main model works

The `manhwa` checkpoint contains 383,475 parameters at the default width of 24. On each tile, the network takes floating-point RGB in `[0, 1]` with shape `B × 3 × H × W` and a binary mask with shape `B × 1 × H × W`. A mask value of `1` marks a hole. The network performs one forward pass per tile.

### 1. Build a fill from the visible artwork

`context_fill` builds two pyramids by repeatedly applying 2 × 2 average pooling to the visible RGB values and their visibility mask. At each scale, it divides the pooled color by the pooled visibility to compute the average of the available pixels. It then works back to the original resolution with bilinear interpolation. A location uses its local normalized average when the current scale contains visible pixels and the coarser fill otherwise. A fully masked tile starts from mid-gray, with each RGB channel set to 0.5.

### 2. Encode color contours

The first convolution receives 13 channels. Color differences are computed only across pairs of visible neighbors and are zero when either pixel falls inside the mask. This lets the model see color boundaries even where adjacent tones have similar luminance. The lettering hidden by the mask contributes neither color nor a contour to the input.

| Channels | Input feature |
|---:|---|
| 3 | Original RGB with the hole set to zero |
| 1 | Binary hole mask |
| 3 | Multiscale RGB context fill |
| 3 | Signed horizontal differences in red, green, and blue |
| 3 | Signed vertical differences in red, green, and blue |

### 3. Predict an RGB correction

The encoder runs at full, half, and quarter resolution with 24, 48, and 96 channels. Its blocks use 3 × 3 convolutions, four-group GroupNorm, and SiLU. Two stride-2 convolutions reduce the spatial size. Three bottleneck blocks with dilation rates 1, 2, and 4 widen the receptive field, and their output is added to the bottleneck input. The decoder uses nearest-neighbor upsampling and concatenates the matching encoder features before each decoding block. A final 3 × 3 convolution emits a three-channel correction.

Inside the mask, the network adds the correction to the context fill and clamps the result to `[0, 1]`. Outside the mask, it copies the original RGB. The model can continue edges that enter a hole or recover broad color transitions from surrounding pixels. Fine details hidden by a large mask can still appear blurred or be reconstructed incorrectly.

### 4. Restore a long page in tiles

`page.py` builds overlapping windows around the mask bounds and skips windows with empty masks. The model receives native-resolution crops and uses the known pixels within each crop as context. `restore_page` linearly tapers contributions toward internal tile edges and averages the weighted predictions where tiles overlap. The floating-point accumulation buffer holds only the current horizontal band.

The CLI then pastes the blended result only at masked pixels and writes an RGB PNG. An empty mask returns the original RGB pixels. This path handles tall webtoon strips and narrow or odd-sized crops. It uses the network for every active tile. The flat-background shortcut belongs to the experimental `run.py` path described below.

## Training the main checkpoint

### Synthetic lettering masks

`prepare.py pages --hangul-font` expects archives with page images under `images/` and matching YOLO label files under `labels/`. It samples native-size crops outside annotation bounds and filters low-variation crops. It cycles through Hangul dialogue, large rotated SFX, and rectangular detector-style masks. The original crop is the target. A source page stays in one split, and exact duplicates are filtered by page fingerprint. The training archive supplies train and validation crops. The test archive supplies held-out crops.

Use a TTF or OTF font that includes Hangul. The verified run used Malgun Gothic on Windows. Replace the archive paths with your own and use fresh output directories. The commands below show the settings used for the eight-epoch run.

```powershell
python prepare.py --out data/manhwa_native --size 256 pages --train-zip 'path/to/train.zip' --test-zip 'path/to/test.zip' --per-page 15 --hangul-font "$env:WINDIR\Fonts\malgun.ttf"
python train.py --data data/manhwa_native --run runs/manhwa_native --architecture manhwa --init-from runs/baseline_v01/best.pt --epochs 8 --batch 6 --lr 0.00015 --seed 27 --device cuda
```

`--architecture manhwa` selects the color-contour network. `--init-from` loads a compatible checkpoint. When initializing from the width-matched seven-channel `baseline`, the loader copies the original weights and initializes the six new contour-channel weights to zero. For a fresh initialization, omit `--init-from`.

### Aligned pairs with real lettering

Provide a JSONL manifest with one record per page. Paths are relative to the manifest directory. Each record points to the lettered image, its aligned clean reference, and its mask. The optional `group` field can describe the content or mask category. Assign each series to one split before cropping and use the same series ID for editions of the same work. `train.py` requires examples in all three splits.

```json
{"series":"work_a","split":"train","group":"hair","input":"a/lettered.png","target":"a/clean.png","mask":"a/text_mask.png"}
{"series":"work_b","split":"val","group":"sfx","input":"b/lettered.png","target":"b/clean.png","mask":"b/text_mask.png"}
{"series":"work_c","split":"test","group":"dialogue","input":"c/lettered.png","target":"c/clean.png","mask":"c/text_mask.png"}
```

`prepare.py pairs` crops the aligned images and mask together at native scale. It pads short crops to the selected size using edge padding for RGB and zero padding for the mask. It writes `manifest.json` for `train.py` and `pairs.jsonl` for `run.py`. The importer rejects a series or exact duplicate clean page assigned to multiple splits.

```powershell
python prepare.py --out data/manhwa_pairs --size 256 pairs --manifest data/pages.jsonl
python train.py --data data/manhwa_pairs --run runs/manhwa_pairs --architecture manhwa --init-from runs/manhwa_native/best.pt --device cuda
```

### Loss and checkpoint selection

Training uses AdamW and clips the gradient norm to 1.0. The loss combines four terms.

| Weight | Term | What it measures |
|---:|---|---|
| 1.00 | Masked RGB L1 | Absolute color error over masked pixels |
| 0.50 | Contrast-weighted gradient L1 | Horizontal and vertical RGB difference errors for neighbor pairs that touch the mask. Each pair is weighted by `1 + 3 × contrast`, where contrast is the largest absolute clean RGB difference across its channels. |
| 0.20 | Multiscale RGB L1 | Error after average pooling at factors 2, 4, and 8, evaluated in pooled cells that intersect the mask |
| 0.25 | Boundary RGB L1 | Color error on the inner mask rim, obtained by subtracting a 5 × 5 erosion from the mask |

`best.pt` holds the checkpoint with the lowest validation masked MAE, and `last.pt` holds the most recent model, optimizer state, settings, and training history. `metrics.json` records the validation history and test scores from `best.pt`. To continue training, reuse the original data, architecture, batch size, learning rate, seed, and run directory. Use `--resume` in place of `--init-from` and raise `--epochs` to the desired total.

## Verified results

The main checkpoint was fine-tuned for eight epochs on 855 crops from 57 pages. Epoch 5 had the lowest validation MAE on 270 crops from 18 separate pages. Both models below were evaluated on the same crops and masks. MAE and PSNR use RGB in `[0, 1]` and aggregate only the masked pixels. Lower MAE and higher PSNR are better.

| Evaluation set | Crops | Baseline MAE | Manhwa MAE | Baseline PSNR | Manhwa PSNR |
|---|---:|---:|---:|---:|---:|
| 11 held-out pages | 165 | 0.05683 | 0.05219 | 18.30 dB | 18.81 dB |
| Hangul dialogue subset | 55 | 0.02074 | 0.01886 | 24.18 dB | 24.48 dB |
| SFX subset | 55 | 0.05254 | 0.04941 | 18.87 dB | 18.84 dB |
| Detector mask subset | 55 | 0.06935 | 0.06300 | 17.23 dB | 18.01 dB |
| 45 external images | 135 | 0.09197 | 0.08659 | 14.67 dB | 15.09 dB |

The held-out result is an 8.2% relative reduction in masked MAE compared with `baseline_v01`. A whole-page test at 720 × 16,000 pixels preserved every RGB pixel outside the mask. These quality scores come from real artwork with synthetic masks. Measuring quality on real lettering and missed text from a detector requires aligned lettered and clean pages with the masks used in practice.

### Mobile feasibility

A fixed-shape FP32 export for 256 × 256 tiles was verified with the PyTorch Lite interpreter on Windows. The export was 1.49 MiB. Across all 165 held-out crops it differed from the original PyTorch model by at most 0.0000014 on the `[0, 1]` RGB scale.

On an Intel Core i9-13900HX, median inference time was 164 ms per tile with one CPU thread and 64 ms with four threads. Processing a 720 × 16,000 page with 51 active tiles took about 3.6 seconds with four threads, excluding file I/O. Peak Windows process memory was 676 MiB, including Python, PyTorch, and image buffers. These are desktop measurements. ExecuTorch and ONNX Runtime exports for these weights and performance on physical Android or iPadOS devices remain unverified.

## Experimental RGB and DDIM modes

`run.py` trains and runs a separate U-Net with widths 24, 48, 72, and 96. It downsamples three times, uses dilated residual blocks at the bottleneck, and conditions its blocks with FiLM from a pooled context vector and sinusoidal timestep features. Both modes see visible RGB, the binary mask, and horizontal and vertical luminance differences between visible neighbors.

| Mode | Training objective | Inference |
|---|---|---|
| `rgb` | L1 loss on predicted RGB in `[-1, 1]` within the mask | One pass |
| `diffusion` | MSE on Gaussian noise added only inside the mask, with 1,000 timesteps and a linear beta schedule from 0.0001 to 0.02 | Seeded deterministic DDIM with 4, 8, or 16 steps. Known pixels are fixed at each step. |

By default, `run.py infer` and `run.py eval` try a flat fill before running the network. The check uses the visible three-pixel ring around the mask. If the ring contains at least 16 pixels and each RGB value stays within `6 / 255` of its channel median, it fills the hole with that median. Use `--no-router` to measure the network alone. A uniform ring can also surround artwork hidden by a mask, so validate routing on examples that include thin lines and edges.

Use the `pairs.jsonl` generated by `prepare.py pairs` with `--size 256`. Each mode saves its own checkpoint format. Use `run.py` for those checkpoints and `inpaint.py` for checkpoints produced by `train.py`.

```powershell
python run.py train --manifest data/manhwa_pairs/pairs.jsonl --mode rgb --output runs/manhwa_rgb.pt --seed 27 --device cuda
python run.py train --manifest data/manhwa_pairs/pairs.jsonl --mode diffusion --output runs/manhwa_diffusion.pt --seed 27 --device cuda
python run.py infer --weights runs/manhwa_diffusion.pt --input page.png --mask mask.png --output restored.png --steps 8 --seed 27 --device cuda
python run.py eval --manifest data/manhwa_pairs/pairs.jsonl --split val --weights runs/manhwa_diffusion.pt --steps 8 --seed 27 --no-router --device cuda
```

`run.py infer` uses the shared page tiler. `run.py eval` reports a JSON line per crop with its route, seed, masked MAE, MSE, PSNR, and the number of modified pixels outside the mask. A perfect match has infinite PSNR and is reported as `null`.

## Verification and source files

```powershell
python -m unittest -v
```

The tests cover baseline weight transfer, gradients on the new contour channels, aligned native-resolution crops, blending on long pages, isolation of hidden RGB, and exact preservation of visible pixels.

| File | Role |
|---|---|
| [model/model_baseline.py](model/model_baseline.py) | Main residual U-Net and contour-weighted loss |
| [model/model_hybrid.py](model/model_hybrid.py) | Shared context fill and loss helpers, plus an alternate hybrid architecture |
| [model/registry.py](model/registry.py) | Architecture selection, checkpoint loading, and weight transfer |
| [inpaint.py](inpaint.py) and [page.py](page.py) | Page inference, tile selection, and blending |
| [prepare.py](prepare.py) and [manhwa_masks.py](manhwa_masks.py) | Data and mask preparation |
| [train.py](train.py) | Training, checkpointing, and masked evaluation |
| [run.py](run.py) and [model/model.py](model/model.py) | Experimental RGB and DDIM modes |

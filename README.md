# tb-classifier

A custom 3-class chest X-ray classifier for TB detection, built for the ALIVE implementation.
The architecture is a **3-stage-ablated ResNet18** with a **FlipR** block inserted after `layer2`, 
a learned left–right asymmetry gate motivated by the unilateral nature of many TB findings
on CXR. Trained from scratch.

Classes: `healthy` · `sick-non-tb` · `tb`. Dataset: TBX11K, folder-layout.

## Headline result

`best.ckpt` on the TBX11K test split (1260 images: 570 healthy, 570 sick-non-TB, 120 TB):

| Class       | Sensitivity | Specificity | AUROC  | Support |
| ----------- | :---------: | :---------: | :----: | :-----: |
| healthy     |   0.9965    |   0.9957    | 0.9993 |   570   |
| sick-non-tb |   0.9930    |   0.9870    | 0.9987 |   570   |
| tb          |   0.9417    |   0.9991    | 0.9984 |   120   |
| **macro**   | **0.9770**  | **0.9939**  | **0.9988** | 1260 |

Accuracy 0.9897 · macro F1 0.9834.

## Reproduction

Everything below uses the canonical run `experiments/configs/default.yaml`
(seed 42, 100 epochs, cosine LR, bf16-mixed, balanced class weights, augmentations off).

### 1. Install

```bash
uv sync                       # preferred
# or: pip install -e .
```

### 2. Get the data

TBX11K, folder-layout, in `data/tbx11k/`:

```
data/tbx11k/
  train/{healthy,sick-non-tb,tb}/*.png
  val/  {healthy,sick-non-tb,tb}/*.png
  test/ {healthy,sick-non-tb,tb}/*.png
```

Source: <!-- TODO: fill in the exact download URL / instructions you used -->

Expected counts after splitting: train 5880, val 1260, test 1260 (570 / 570 / 120 per class
in val and test; train is upsampled to 2660 / 2660 / 560 by the splitter). Sanity-check
with `uv run python scripts/check_data.py` — it loads each split and prints class counts
plus a sample batch.

### 3. Train

```bash
uv run python scripts/train.py --config experiments/configs/default.yaml --fast-dev-run  # one batch, no W&B
uv run python scripts/train.py --config experiments/configs/default.yaml                # real run
```

Outputs land in `experiments/results/flipr_resnet18_k3_npt/`:
- `checkpoints/epoch=<NN>-auroc=<X.XXX>.ckpt` — top-2 by `val/auroc_macro`
- `checkpoints/last.ckpt` — most recent epoch
- W&B run dir (logger writes here too)

The run logs to W&B project `tb-classifier`. Set `WANDB_MODE=offline` if you don't want
to push, or `--config <yaml-with-logger-disabled>` if you want to wire that up.

To resume from a checkpoint:

```bash
uv run python scripts/train.py --config experiments/configs/default.yaml --resume path/to/last.ckpt
```

### 4. Evaluate

```bash
uv run python scripts/evaluate.py                                                    # defaults to default.yaml + best.ckpt
uv run python scripts/evaluate.py --ckpt experiments/results/flipr_resnet18_k3_npt/checkpoints/last.ckpt
uv run python scripts/evaluate.py --device cpu                                       # force CPU
```

Prints the per-class table (sensitivity / specificity / AUROC / support), the macro
row, accuracy, the sklearn classification report, and the confusion matrix — the same
numbers shown in the headline table above.

### Reproducibility notes

- **Seed.** `L.seed_everything(cfg.training.seed, workers=True)` is called in
  `scripts/train.py` before any dataloader or model is constructed. This seeds Python,
  NumPy, PyTorch (CPU + CUDA), and Lightning worker processes.
- **Hardware.** The reported best run was trained on a single Colab T4 in bf16-mixed.
  Results on a different GPU / different cuDNN / different PyTorch version will land
  *close to* but not bit-exactly equal to the headline numbers — Lightning's
  `seed_everything` does not enable cuDNN deterministic mode (it slows training enough
  that we left it off). Expect runs to land within a few tenths of a point of the
  reported macro AUROC.
- **Augmentations are off** (`data.augment: false`). The full augmentation pipeline
  in `src/tb_classifier/data/transforms.py` is wired but disabled for the canonical
  run — turning it on is a separate ablation, not the reproduction path.
- **Colab.** Clone the repo, `uv sync` (or `pip install -e .`), mount Drive, then
  edit `experiments/configs/default.yaml` to set `training.log_dir` to a Drive path
  (e.g. `/content/drive/MyDrive/tb-classifier-runs`) before running
  `python scripts/train.py --config experiments/configs/default.yaml`. This keeps
  checkpoints alive across runtime disconnects.

## Model

Top-level model (`src/tb_classifier/models/classifier.py`) = backbone + linear head.
There are no architecture knobs — the backbone is the project.

The backbone (`src/tb_classifier/models/backbone.py`) is a torchvision **ResNet18**
truncated to three stages, with a `FlipRBlock` between `layer2` and `layer3`:

```
conv1 → bn1 → relu → maxpool → layer1 → layer2 → flipr → layer3 → avgpool → flatten
```

`feat_dim = 256` (resnet18 `layer3` output channels). The linear head maps that to
`num_classes`. Default is random init (`pretrained=false`); flip `model.pretrained`
in the YAML to start from ImageNet weights instead.

### FlipRBlock

`src/tb_classifier/models/blocks/flipr.py`. Given features `x`:

```
asym = avgpool3x3(x - flip_horizontal(x))
gate = sigmoid(conv1x1(asym))          # per-pixel scalar in (0, 1)
out  = x * (1 + gate)                  # residual-friendly amplification
```

Adds `C + 1` parameters total (one 1×1 conv from C to 1) and degrades to identity when
the gate saturates at 0, so it can only help or no-op.

## Configs

`experiments/configs/default.yaml` is the canonical run. The model section has
two fields (`num_classes`, `pretrained`); the rest of the YAML is data and
training hyperparams. Copy it to add new experiments.

## Dev

```bash
uv run python scripts/check_data.py                  # dataloader smoke test
uv run ruff check . && uv run ruff format .          # lint / format
```

## Layout

```
src/tb_classifier/
├── models/
│   ├── blocks/flipr.py      # FlipR asymmetry gate
│   ├── backbone.py          # 3-stage ResNet18 + FlipR
│   └── classifier.py        # backbone + linear head
├── data/
│   ├── dataset.py           # TBX11K folder dataset
│   ├── transforms.py        # albumentations pipelines
│   └── loaders.py
├── training/
│   └── lightning_module.py  # TBLitModule
├── utils/metrics.py         # AUROC, sensitivity, specificity
└── config.py                # YAML → dataclasses

experiments/
├── configs/                 # per-run YAMLs
└── results/                 # gitignored (checkpoints, W&B run dirs)

scripts/{train,evaluate,check_data}.py
data/                        # gitignored
```

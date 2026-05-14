# tb-classifier

A custom 3-class chest X-ray classifier for TB detection, built for the ALIVE implementation.
The architecture is a depth-truncated ResNet with an optional **FlipR** block — a learned
left–right asymmetry gate motivated by the unilateral nature of many TB findings on CXR.

Classes: `healthy` · `sick-non-tb` · `tb`. Dataset: TBX11K, folder-layout.

## Install

```bash
uv sync                    # preferred
# or: pip install -e .
```

## Run

```bash
uv run python scripts/train.py --config experiments/configs/flipr_resnet18_colab_k3.yaml
uv run python scripts/train.py --config <cfg> --fast-dev-run   # wiring sanity check
uv run pytest                                                  # tests
uv run ruff check . && uv run ruff format .                    # lint / format
```

Training is Lightning-based with W&B logging, bf16-mixed precision, cosine LR, and balanced
class weights when configured. Checkpoints are saved on best `val/auroc_macro`.

## Data layout

```
<data.root>/
  train/{healthy,sick-non-tb,tb}/*.png
  val/  {healthy,sick-non-tb,tb}/*.png
  test/ {healthy,sick-non-tb,tb}/*.png
```

Labels: `0=healthy`, `1=sick-non-tb`, `2=tb` (see `src/tb_classifier/data/dataset.py`).

## Model

Top-level model (`src/tb_classifier/models/classifier.py`) = backbone + linear head.

The backbone (`backbone.py`) wraps a torchvision **ResNet18/50** and exposes two knobs:

| Knob           | What it does                                                                 |
| -------------- | ---------------------------------------------------------------------------- |
| `keep_stages`  | 1–4. Drops `layer4`, then `layer3`, then `layer2` to retain spatial detail. |
| `use_flipr`    | Inserts a `FlipRBlock` after `layer2` (requires `keep_stages >= 2`).        |

`feat_dim` is read from the last kept stage; the linear head adapts automatically.
When `keep_stages == 4` and `use_flipr` is false, the plain torchvision ResNet is used
(with `fc → Identity`) so existing checkpoints keep loading.

### FlipRBlock

`src/tb_classifier/models/blocks/flipr.py`. Given features `x`:

```
asym = avgpool3x3(x - flip_horizontal(x))
gate = sigmoid(conv1x1(asym))          # per-pixel scalar in (0, 1)
out  = x * (1 + gate)                  # residual-friendly amplification
```

Adds essentially no parameters and degrades to identity at worst.

## Configs

Each run owns a YAML in `experiments/configs/`:

| Config                           | Backbone   | `keep_stages` | FlipR | Pretrained |
| -------------------------------- | ---------- | :-----------: | :---: | :--------: |
| `resnet18_colab.yaml`            | resnet18   | 4             |  no   |    yes     |
| `flipr_resnet18_colab.yaml`      | resnet18   | 4             |  yes  |    yes     |
| `flipr_resnet18_colab_k2.yaml`   | resnet18   | 2             |  yes  |    yes     |
| `flipr_resnet18_colab_k3.yaml`   | resnet18   | 3             |  yes  |    yes     |
| `flipr_resnet18_colab_k3_npt.yaml` | resnet18 | 3             |  yes  |     no     |
| `flipr_resnet50.yaml`            | resnet50   | —             |  yes  |    yes     |

The `_k*` variants ablate depth; `_npt` ablates ImageNet initialization.

## Layout

```
src/tb_classifier/
├── models/
│   ├── blocks/flipr.py      # FlipR asymmetry gate
│   ├── backbone.py          # FlipRResNet wrapper, stage stripping
│   └── classifier.py        # backbone + linear head
├── data/
│   ├── dataset.py           # TBX11K folder dataset
│   ├── transforms.py        # albumentations pipelines
│   └── loaders.py
├── training/
│   ├── lightning_module.py  # TBLitModule
│   ├── losses.py
│   └── trainer.py
├── utils/metrics.py         # AUROC, sensitivity, specificity
└── config.py                # YAML → dataclasses

experiments/
├── configs/                 # per-run YAMLs
└── results/                 # gitignored

scripts/{train,evaluate,check_data}.py
notebooks/                   # EDA, eval plots (e.g. PR curves)
data/, outputs/              # gitignored
```

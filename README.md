# SAM3-I: Segment Anything with Instructions

Official repository for the paper "SAM3-I: Segment Anything with Instructions".

## Directory Structure

```
SAM3-I/
├── run.sh                          # Unified entry (install / train / eval)
├── scripts/
│   ├── train.sh                    # Training launcher
│   ├── eval.sh                     # One-click evaluation (reproduces paper results)
│   ├── inference.py                # Multi-GPU batch inference
│   └── evaluate.py                 # gIoU / P@0.50 metric computation
└── sam3/
    ├── pyproject.toml
    └── sam3/
        ├── model_builder.py        # Model construction + weight loading
        ├── model/                  # Model components
        ├── train/configs/sam3i/    # Hydra configs
        │   ├── base.yaml           # ★ Primary config (modify paths here)
        │   ├── sam3i_1-1.yaml      # Stage 1-1
        │   ├── sam3i_1-2.yaml      # Stage 1-2
        │   └── sam3i_3_all.yaml    # Stage 3
        └── eval/                   # Evaluation logic
```

---

## Installation

Use Linux / WSL with `uv` and an NVIDIA driver compatible with CUDA 12.6.
The root `pyproject.toml` manages Python 3.12, the local editable `sam3`
workspace member, and the CUDA-enabled PyTorch index. Exact dependency
versions are recorded in `uv.lock`.

```bash
cd /home/kmg/SAM3-I
uv sync --locked
# Equivalent installation entry point:
# bash run.sh install
```

`uv sync` creates or updates `.venv` and installs the inference, evaluation,
training, and development dependencies. No manual environment activation is
needed when using `uv run`:

```bash
uv run python scripts/inference.py --help
uv run python scripts/evaluate.py --help

# Add/remove dependencies through uv; commit pyproject.toml and uv.lock together.
uv add <package>
uv remove <package>

# Optional notebook dependencies:
uv sync --locked --group notebooks
```

The default environment covers image training and evaluation. Video decoding
is optional (`uv sync --locked --group video`) and is also included in the
notebook group. The upstream Decord 0.6.0 Linux wheel has
[incorrect platform metadata](https://github.com/dmlc/decord/issues/356), which
can cause repeated reinstalls and a `uv pip check` warning in those optional
environments.

Run these commands from the repository root. `bash run.sh train`, `eval`, and
`inference` also use the uv-managed environment. For a smaller environment,
use `uv sync --locked --no-dev` and run commands with `uv run --no-dev`.
To update locked versions within the declared compatibility ranges, use
`uv lock --upgrade` followed by `uv sync --locked`.

Keep datasets, checkpoints, experiment logs, and evaluation outputs under
`/mnt/d/heihe/Documents/ChatGPT/sam3i试验区`; pass their paths explicitly to
the training or evaluation commands.

---

## Pretrained Checkpoints

| Stage | Description | Download |
|-------|-------------|----------|
| **Stage 3** | Joint training — reproduces paper results | [Google Drive](https://drive.google.com/drive/folders/1aCm6M_ckjd2l1UJq7g5cDm4965m8Dk3c?usp=sharing) |

---

## Data Preparation

Annotation JSONs ([Google Drive](https://drive.google.com/drive/folders/1aCm6M_ckjd2l1UJq7g5cDm4965m8Dk3c?usp=sharing)) and image folders can live in **separate locations**. Configure both in `base.yaml`:

```yaml
# sam3/sam3/train/configs/sam3i/base.yaml
paths:
  sam3i_datasets_root: ./sam3i_datasets   # annotation JSON root
  sam3i_image_root: /data/my_images       # image folder root (defaults to sam3i_datasets_root)
```

The two roots have parallel per-dataset sub-directories:

```
<sam3i_datasets_root>/              <sam3i_image_root>/
├── RefCOCO/                        ├── coco2014/          ← shared by 5 datasets
│   ├── sam3i_train.json            │   ├── train2014/
│   └── sam3i_val.json              │   └── val2014/
├── RefCOCOplus/                    ├── coco2017/          ← shared by 4 datasets
├── RefCOCOg/                       │   ├── train2017/
├── gRefCOCO/                       │   └── val2017/
├── Ref-ZOM/                        ├── ReasonSeg/         ← standalone
├── HMPL-Instruct_1to1/             └── ...
├── HMPL-Instruct_1toN/
├── HMPL-Instruct_1toAll/
├── ReasonSeg/
└── MMR/
```

### Image Sources

Many datasets share the same underlying images. Only **three** image sources are needed:

| Image source | Datasets | Download |
|---|---|---|
| **COCO train/val 2014** | RefCOCO, RefCOCOplus, RefCOCOg, gRefCOCO, Ref-ZOM | train2014.zip (http://images.cocodataset.org/zips/train2014.zip) <br> val2014.zip (http://images.cocodataset.org/zips/val2014.zip) |
| **COCO train/val 2017** | HMPL-Instruct_1to1, HMPL-Instruct_1toN, HMPL-Instruct_1toAll, MMR | train2017.zip (http://images.cocodataset.org/zips/train2017.zip) <br> val2017.zip (http://images.cocodataset.org/zips/val2017.zip) |
| **ReasonSeg** | ReasonSeg | [Google Drive](https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy) |

### Configuring Shared Image Folders

Since multiple datasets point to the same images, use the per-dataset `img_folder_*` variables in `base.yaml` to avoid duplicating data:

**Training** — edit `base.yaml`:
```yaml
paths:
  # COCO 2014 (shared by RefCOCO family + Ref-ZOM)
  img_folder_RefCOCO:     /data/coco2014/
  img_folder_RefCOCOplus: /data/coco2014/
  img_folder_RefCOCOg:    /data/coco2014/
  img_folder_gRefCOCO:    /data/coco2014/
  img_folder_RefZOM:      /data/coco2014/

  # COCO 2017 (shared by HMPL family + MMR)
  img_folder_HMPL_1to1:   /data/coco2017/
  img_folder_HMPL_1toN:   /data/coco2017/
  img_folder_HMPL_1toAll: /data/coco2017/
  img_folder_MMR:         /data/coco2017/

  # Standalone
  img_folder_ReasonSeg:   /data/ReasonSeg/
```

**Evaluation** — pass `IMAGE_FOLDER_MAP`:
```bash
IMAGE_FOLDER_MAP="RefCOCO:coco2014,RefCOCOplus:coco2014,RefCOCOg:coco2014,gRefCOCO:coco2014,Ref-ZOM:coco2014,HMPL-Instruct_1to1:coco2017,HMPL-Instruct_1toN:coco2017,HMPL-Instruct_1toAll:coco2017,MMR:coco2017" \
  bash scripts/eval.sh
```

Unmapped datasets default to `IMAGE_ROOT_BASE/<DATASET_NAME>/`.

---

## Training

### Stages

| Stage | Config | Description | Initial Weights |
|-------|--------|-------------|-----------------|
| 1 | `sam3i_1-1` | Simple query | SAM3 base (HuggingFace) |
| 2 | `sam3i_1-2` | Complex query | Stage 1 checkpoint |
| 3 | `sam3i_3_all` | Joint (all losses) | Stage 2 checkpoint |

### Quick Start

```bash
# 1. Edit paths in base.yaml first

# Stage 1
CONFIG=configs/sam3i/sam3i_1-1 bash run.sh train

# Stage 2 (set checkpoint_path in sam3i_1-2.yaml)
CONFIG=configs/sam3i/sam3i_1-2 bash run.sh train

# Stage 3 (set checkpoint_path in sam3i_3_all.yaml)
CONFIG=configs/sam3i/sam3i_3_all bash run.sh train
```

For Stage 2/3, uncomment `checkpoint_path` in the corresponding YAML and point to the previous stage's output.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG` | `configs/sam3i/sam3i_1-1` | Config name |
| `NPROC` | `8` | GPUs per node |
| `NNODES` | `1` | Number of nodes |
| `NODE_RANK` | `0` | Current node rank |
| `MASTER_ADDR` | `127.0.0.1` | Master address |
| `MASTER_PORT` | `29501` | Communication port |

---

## Evaluation

One command reproduces all paper results:

```bash
CHECKPOINT=/path/to/checkpoint.pt \
  DATASET_JSON_ROOT=/path/to/json/root \
  IMAGE_ROOT_BASE=/path/to/image/root \
  bash scripts/eval.sh
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHECKPOINT` | *(required)* | Model checkpoint path |
| `DATASET_JSON_ROOT` | *(required)* | Annotation JSON root |
| `IMAGE_ROOT_BASE` | *(required)* | Image folder root |
| `IMAGE_FOLDER_MAP` | *(optional)* | Shared folder mapping, format: `"DATASET:FOLDER,..."` |
| `OUTPUT_DIR` | `./outputs/eval_results` | Output directory |
| `GPUS` | `0,1,2,3,4,5,6,7` | GPU list |
| `BATCH_SIZE` | `32` | Inference batch size |
| `DET_THRESHOLD` | `0.5` | Detection threshold |

---

## Dual-Panel Instruction Annotation Tool

We release our [Dual-Panel Instruction Annotation Tool](https://github.com/debby-0527/SAM3-I/blob/main/web_annotation_tool/README.md), a dedicated web-based interface used to support challenging instruction grounding scenarios, in particular the one-to-many setting.


---

## Citation

```bibtex
@inproceedings{li2026sam3i,
title={SAM3-I: Segment Anything with Instructions},
author={Li, Jingjing and Feng, Yue and Guo, Yuchen and Huang, Jincai and Ji, Wei and Bi, Qi and Piao, Yongri and Zhang, Miao and Zhao, Xiaoqi and Chen, Qiang and Zou, Shihao and Lu, Huchuan and Cheng, Li},
booktitle={The 64th Annual Meeting of the Association for Computational Linguistics},
year={2026}
}
```

# ASL hand gesture classification (A–Z, 0–9)

Train an image classifier on **`data/asl_processed/`** (processed train/test splits) and run **webcam inference** via `src/train_asl.py`.

With **`venv` activated**, use `python` instead of `uv run python` in the commands below.

## Setup

### uv (recommended)

Install [uv](https://docs.astral.sh/uv/) (`brew install uv` or the curl installer from their docs). The repo pins **Python 3.12** in `.python-version`; `uv` can install that runtime for you and avoids many Homebrew **`pyexpat` / `ensurepip`** issues.

```bash
cd asl_detection
uv python install 3.12
uv sync
# Apple Silicon GPU (optional):
uv sync --extra metal
```

If `.venv` is in a bad state: `rm -rf .venv` then `uv sync` again.

### pip + venv

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

Use **Python 3.12** on macOS if you want **`tensorflow-metal`** (no wheel for 3.13). Prefer **`uv`** if Homebrew Python breaks `venv` / `pyexpat`.

## Training commands

```bash
uv run python src/train_asl.py train --arch efficientnetb0 --data-dir data/asl_processed --model-dir models/asl_model
```

```bash
uv run python src/train_asl.py train --arch mobilenetv2 --data-dir data/asl_processed --model-dir models/asl_mobilenetv2
```

**Processed → raw (Option A):** train on processed, then fine-tune on `data/asl_dataset/`:

```bash
uv run python src/train_asl.py train --arch mobilenetv2 --model-dir models/asl_mobilenetv2 --raw-finetune --raw-dir data/asl_dataset
```

**Optional:** `--fine-tune --fine-tune-at 200`, `--val-split 0.2` (validation carved from `train/`; `test/` is only for the final printed test score and `evaluate`), `--mixed-float16` (GPU), `--cache-in-memory` (needs RAM), `--batch-size` (default 64). **`ReduceLROnPlateau`** is enabled on `val_acc`; tune with `--lr`, `--reduce-lr-patience`, etc.

Trained weights are written to **`{model-dir}/model.keras`** (plus `class_names.json` and `meta.json`). Older checkpoints without `model.keras` may still load if `load_model(model_dir)` works for that layout.

## macOS: TensorFlow and Metal

| Topic | Notes |
|--------|--------|
| **Metal** | `uv sync --extra metal`. Do not use **`--cpu-only`** if you want the GPU. |
| **TensorFlow version** | Pinned **`tensorflow<2.19`** so **`tensorflow-metal` 1.2.x** can load (TF 2.19+ breaks that plugin). After dependency changes: `rm -rf .venv && uv sync --extra metal`. |
| **`DYLD_LIBRARY_PATH`** | Do **not** point it at Homebrew Expat while using Metal; it breaks resolution of TensorFlow’s own `.so` files. `unset DYLD_LIBRARY_PATH` before training. |
| **Import crash** (`mutex` / protobuf) | Use a **clean env** (`uv sync` or a small venv). Huge Conda stacks with **PyArrow** often clash with TF 2.20+; try `pip uninstall -y pyarrow` if safe, or see `pyproject.toml` constraints. |
| **Stability** | The training script re-execs once on macOS with `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`. For the old single-thread workaround (much slower): **`--cpu-only`** or `ASL_TF_STABLE=1`. |

## Evaluate (CLI)

```bash
uv run python src/train_asl.py evaluate --data-dir data/asl_processed --model-dir models/asl_model
```

## Realtime (webcam)

```bash
uv run python src/train_asl.py realtime --model-dir models/asl_mobilenetv2
```

Hand crop (often better on messy backgrounds):

```bash
uv run python src/train_asl.py realtime --model-dir models/asl_mobilenetv2 --use-mediapipe
```

## Notebook

Open **`notebooks/ASL_Evaluation.ipynb`** and run all cells. Point **`MODEL_DIR`** at your saved model (defaults assume `../models/asl_model/`). Test images: **`data/asl_processed/test/`**.

## Built on 
| Item | Specification |
|------|----------------|
| **Model** | MacBook Pro |
| **SoC** | Apple M4 |
| **CPU** | 10 cores (4 performance, 6 efficiency) |
| **Memory** | 16 GB unified |
| **GPU** | Apple M4 integrated, 10 GPU cores, **Metal 4** |
| **OS** | macOS 26.2 (build 25C56) |

# SnakeGeneration

SnakeGeneration is a reinforcement learning project for training and comparing Snake agents built with **PyTorch**. The repository includes an optimized vectorized Snake environment, multiple Q-network architectures, training utilities, benchmarking helpers, and a Pygame-based viewer for visualizing trained models and exporting comparison videos.

Almost all of the code in this repository was written by me. I used Copilot mainly as support for parts of the project such as the environment vectorization, the viewer/export code, and sections of the `README.md`—especially in areas that were new to me, such as MP4 export.

The project was created mainly as a fun challenge and as an excuse to make use of a spare GPU I had lying around. All models were trained on a 5 GB GPU.

---

## Demo

### Static seed comparison

https://github.com/user-attachments/assets/7ce6f687-500c-41a9-8d47-eb5c8ae909eb


### Single run comparison

https://github.com/user-attachments/assets/5a9ced84-00ad-4118-a5b4-5b3f104cdf1a


### 10 minutes comparison - best score

https://github.com/user-attachments/assets/5b0cf737-a5e7-428d-b547-c343cfd68980

### 1 hour comparison - best score

https://github.com/user-attachments/assets/cb19d2ac-20f8-4fab-992b-16aca5f72090

### Note on variation between runs

Some of the visible differences between models come from simple run-to-run randomness rather than from the model architecture alone. Because the environment uses randomized starts and fruit placement, a model can sometimes look better or worse just because it got a more favorable or less favorable seed in a particular episode.

For that reason, the static-seed comparison is useful as a fairer side-by-side reference point. It does not make every trajectory perfectly identical for the entire run, because different actions still cause the games to diverge, but it reduces a lot of the randomness that appears in fully random comparisons.

---

## Features

- **Multiple model types**
  - Dueling **MLP**
  - Dueling **CNN**
  - Dueling **C51-CNN**
  - **EA-based** runs seeded from trained models
- **Fast vectorized environment** implemented in `env_fast.py`
- **Training utilities** in `trainer.py`
- **Benchmark support** in `benchmark.py`
- **Training pipeline** in `main.py`
- **Model comparison viewer** in `app.py`
- **Saved model loading** from `saved_models/`
- **Custom Snake graphics** from `Graphics/` / `graphics/`

---

## Training pipeline overview

The current workflow is organized as a multi-stage training pipeline:

1. Benchmark candidate configurations for **10x10 MLP** and **10x10 CNN** models.
2. Train final **10x10 MLP**, **10x10 CNN**, and **10x10 C51-CNN** models.
3. Train **20x20 seeded models** from the 10x10 checkpoints.
4. Launch **EA jobs** from selected saved checkpoints.

This produces a mix of:

- base 10x10 models,
- seeded 20x20 models,
- and EA-derived descendants.

---

## Current saved models

The repository currently contains the following saved checkpoints:

- `c51_cnn_10x10.pt`
- `c51_cnn_20x20_seed_c51_cnn_10x10.pt`
- `cnn_10x10.pt`
- `cnn_20x20_seed_cnn_10x10.pt`
- `ea_g10_from_c51_cnn_10x10_c51.pt`
- `ea_g10_from_cnn_10x10_cnn.pt`
- `ea_g10_from_mlp_10x10_mlp.pt`
- `ea_g20_from_c51_cnn_20x20_seed_c51_cnn_10x10_c51.pt`
- `ea_g20_from_cnn_20x20_seed_cnn_10x10_cnn.pt`
- `ea_g20_from_mlp_20x20_seed_mlp_10x10_mlp.pt`

The comparison viewer formats these filenames into more readable labels for the grid view.

---

## Project structure

```text
.
├── .gitignore
├── assets/                            # exported comparison media
├── benchmarks/                        # benchmark outputs created by main.py
├── checkpoints/                       # intermediate and final training checkpoints
├── Font/
│   └── PoetsenOne-Regular.ttf
├── Graphics/
│   ├── apple.png
│   ├── body_bl.png
│   ├── body_br.png
│   ├── body_horizontal.png
│   ├── body_tl.png
│   ├── body_tr.png
│   ├── body_vertical.png
│   ├── head_down.png
│   ├── head_left.png
│   ├── head_right.png
│   ├── head_up.png
│   ├── tail_down.png
│   ├── tail_left.png
│   ├── tail_right.png
│   └── tail_up.png
├── saved_models/
│   └── *.pt
├── app.py
├── benchmark.py
├── env_fast.py
├── main.py
├── models.py
└── trainer.py
```

---

## Requirements

Recommended:

- Python **3.10+**
- `torch`
- `numpy`
- `pygame`
- `imageio`
- `pillow`

If you have a `requirements.txt`, install everything with:

```bash
pip install -r requirements.txt
```

Otherwise, install the core packages manually:

```bash
pip install torch numpy pygame imageio pillow
```

---

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/Pawel-Sokolowski/SnakeGeneration.git
cd SnakeGeneration
```

Create and activate a virtual environment (recommended):

### Windows

```bash
python -m venv venv
.\venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Then install dependencies.

---

## Usage

### Run the training pipeline

The main entry point for training is:

```bash
python main.py
```

`main.py` supports command-line options including:

- `--train-envs`
- `--c51-envs`
- `--train-batches`
- `--train-lrs`
- `--full-steps-10`
- `--full-steps-20`
- `--log-every`
- `--curiosity-every`
- `--ea-parallel`

Example:

```bash
python main.py --log-every 1000 --curiosity-every 1 --ea-parallel
```

### View trained models

Use the viewer to load every `.pt` file from `saved_models/` and compare them side by side:

```bash
python app.py
```

Depending on the current version of `app.py`, the viewer can:

- visualize multiple trained models in a grid,
- render custom Snake graphics,
- record a single-run comparison,
- record a “best run during 10 minutes” comparison,
- record a “best run during 1 hour” comparison,
- record a static-seed comparison.

### Benchmarking

Benchmark selection is integrated into the main pipeline and helper utilities:

```bash
python benchmark.py
```

or, depending on your workflow:

```bash
python main.py
```

---

## Model types

The repository includes the following model classes in `models.py`:

- `DuelingMLP`
- `DuelingCNN`
- `DuelingC51CNN`

These are used by the training pipeline and by the visualization app when loading `.pt` checkpoints.

---

## Notes

- The app expects Snake sprites in either `Graphics/` or `graphics/`.
- Demo media and exported outputs are stored in `assets/`.
- If a model file is missing or incompatible, the corresponding load step will fail.
- The environment supports both vector observations and CNN-style observations, which is why the repository supports both MLP and CNN-based agents.

---

## Contributing

Contributions are welcome. If you want to improve training, visualization, model architectures, or documentation:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Open a pull request

---

## License

This repository currently does not declare a license in the project structure shown here. If you plan to make the project open source for broader reuse, adding a `LICENSE` file is strongly recommended.

# SnakeGeneration

SnakeGeneration is a reinforcement learning project for training and comparing Snake agents built with **PyTorch**. The repository includes an optimized vectorized Snake environment, multiple Q-network architectures, training utilities, benchmarking helpers, and a Pygame viewer for visualizing trained models and exporting comparison GIFs.

---

## Demo

### Single run comparison

This GIF shows a live-style multi-model comparison in a single run.

![Single run comparison](assets/all_models_until_all_dead.gif)

### Best run during 10 minutes

This GIF shows the best recorded run for each model during a 10-minute collection window.

![Best run during 10 minutes](assets/all_models_best_10min_grid.gif)

---

## Features

- **Multiple model types**
  - Dueling **MLP**
  - Dueling **CNN**
  - Dueling **C51-CNN**
- **Fast vectorized environment** implemented in `env_fast.py`
- **Training utilities** in `trainer.py`
- **Model comparison viewer** in `app.py`
- **Benchmark support** in `benchmark.py`
- **Saved model loading** from `saved_models/`
- **Custom Snake graphics** from `Graphics/` / `graphics/`

---

## Project Structure

```text
.
├── .gitignore
├── assets/
│   ├── all_models.gif
│   └── all_models_best_runs.gif
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

If you have a `requirements.txt`, install everything with:

```bash
pip install -r requirements.txt
```

Otherwise, install the core packages manually:

```bash
pip install torch numpy pygame imageio
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

### Train models

Use `main.py` to train agents.

```bash
python main.py
```

If your training script supports command-line arguments, you can run the relevant mode or configuration for:

- MLP training
- CNN training
- C51-CNN training
- evolutionary / benchmark workflows

---

### View trained models

Use the viewer to load every `.pt` file from `saved_models/` and compare them side by side:

```bash
python app.py
```

Depending on your current `app.py` version, it can:

- visualize multiple trained models in a grid,
- render custom snake graphics,
- export a single-run comparison GIF,
- export a “best run during 10 minutes” comparison GIF.

---

### Benchmark

If you use the benchmark helper:

```bash
python benchmark.py
```

or, depending on your workflow:

```bash
python main.py benchmark
```

---

## Model Types

The repository currently includes the following model classes in `models.py`:

- `DuelingMLP`
- `DuelingCNN`
- `DuelingC51CNN`

These are used by the training pipeline and by the visualization app when loading `.pt` checkpoints.

---

## Saved Models

Trained checkpoints are expected in:

```text
saved_models/
```

Example filenames:

- `cnn_10x10.pt`
- `c51_cnn_10x10.pt`
- `ea_g20_from_mlp_20x20_seed_mlp_10x10_mlp.pt`

The viewer formats these names automatically for display in the comparison grid.

---

## Notes

- The app expects Snake sprites in either `Graphics/` or `graphics/`.
- The generated demo GIFs are stored in `assets/`.
- If a model file is missing or incompatible, the corresponding load step will fail.

---

## Contributing

Contributions are welcome. If you want to improve training, visualization, model architectures, or documentation:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Open a pull request

---

## License

This repository currently does not declare a license in the provided project structure.
If you plan to make the project open source for broader reuse, adding a license file is strongly recommended.

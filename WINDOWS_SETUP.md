# Snake AI Trainer - Windows App

A simple Windows application for training and testing Snake AI models using Deep Q-Learning.

## Features

- 🚀 **Train Model** - Easy one-click training with customizable parameters
- 🎮 **Test Model** - Watch your trained AI play Snake and see performance stats
- 📊 **Real-time Progress** - See training progress with live updates
- 💾 **Model Management** - Automatically saves and loads models
- 🎯 **GPU Support** - Automatically uses CUDA if available

## Quick Start for Windows

### Method 1: Easy Setup (Recommended)

1. **Download the repository** as a ZIP file and extract it
2. **Double-click `setup_windows.bat`** to install dependencies
3. **Double-click `run_app.bat`** to start the application

### Method 2: Manual Setup

1. Make sure you have Python 3.8+ installed
2. Open Command Prompt in the project folder
3. Run: `pip install -r requirements_windows.txt`
4. Run: `python snake_trainer_app.py`

## How to Use

### Training a Model

1. Open the application
2. Go to the **"Train Model"** tab
3. Adjust training parameters if desired:
   - **Learning Rate**: How fast the AI learns (0.001 is good default)
   - **Episodes**: How long to train (1000 episodes ≈ 10-30 minutes)
   - **Parallel Envs**: More environments = faster training (8 is good default)
   - **Grid Size**: Smaller grids train faster (15 is good default)
4. Click **"🚀 Train Model"**
5. Watch the progress and log messages
6. The trained model will be automatically saved when complete

### Testing a Model

1. Go to the **"Test Model"** tab
2. Select a trained model from the dropdown
3. Click **"🎮 Test Model"**
4. Watch the AI play Snake and see the performance statistics

## System Requirements

- **Operating System**: Windows 10/11
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **GPU**: Optional (NVIDIA GPU with CUDA for faster training)

## Training Time Estimates

- **CPU Only**: 1000 episodes ≈ 30-60 minutes
- **GPU (GTX 1070+)**: 1000 episodes ≈ 10-20 minutes

## Model Files

Trained models are saved in the `saved_models/` folder with timestamp names.
Each model file (`.pt`) contains:
- The neural network weights
- Training parameters used
- Performance metrics

## Troubleshooting

### "No module named 'tkinter'"
- Install tkinter: `pip install tk` or use Python from python.org

### "CUDA not available"
- The app will work on CPU (just slower)
- For GPU support, install CUDA from NVIDIA

### "Permission denied" on .bat files
- Right-click the .bat file → "Run as administrator"
- Or run the Python commands manually

### Poor AI Performance
- Try training for more episodes (2000+)
- Experiment with different learning rates
- Smaller grid sizes are easier to learn

## Advanced Usage

### Custom Training Parameters

You can modify the training parameters in the GUI:

- **Learning Rate**: Controls how much the AI updates its knowledge each step
  - Lower (0.0005): More stable but slower learning
  - Higher (0.002): Faster learning but potentially unstable

- **Episodes**: Total number of games the AI will play during training
  - 500: Quick test (15-30 minutes)
  - 1000: Standard training (30-60 minutes)
  - 2000+: Extended training for better performance

- **Parallel Environments**: Number of Snake games running simultaneously
  - 4: Lower memory usage
  - 8: Balanced (recommended)
  - 16: Faster training but more memory usage

- **Grid Size**: Size of the Snake playing field
  - 10: Very small, learns quickly but limited complexity
  - 15: Good balance (recommended)
  - 20: More complex, takes longer to learn

### Creating an Executable

To create a standalone .exe file that doesn't require Python:

1. Install PyInstaller: `pip install pyinstaller`
2. Run: `pyinstaller --onefile --windowed snake_trainer_app.py`
3. The .exe will be in the `dist/` folder

## Technical Details

- **AI Algorithm**: Deep Q-Learning with Experience Replay
- **Neural Network**: Multi-layer Perceptron (MLP) with 256→128 hidden units
- **Framework**: PyTorch for AI, Tkinter for GUI
- **Game Engine**: Custom Snake environment using Gymnasium and Pygame

## Tips for Best Results

1. **Start Small**: Begin with default settings (1000 episodes, grid size 15)
2. **Be Patient**: Training takes time - watch the progress logs
3. **Multiple Models**: Try training several models with different parameters
4. **Test Performance**: Use the test tab to evaluate different models
5. **Hardware**: GPU training is much faster if you have NVIDIA graphics

---

Enjoy training your Snake AI! 🐍🤖
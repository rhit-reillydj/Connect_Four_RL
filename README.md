# AlphaFour - Connect Four RL

AlphaFour is a Connect Four game where a human can play against an AI powered by Deep Reinforcement Learning. The project is inspired by DeepMind's AlphaGo and AlphaZero, adapted for the game of Connect Four. The primary interface is a web application built with Streamlit.

This project was developed by Dominic Reilly.

## Features

*   Play Connect Four against an AI opponent in your browser.
*   The AI uses a Convolutional Neural Network (CNN) to evaluate board positions and a Monte Carlo Tree Search (MCTS) algorithm to decide its moves.
*   **NEW**: Full multithreaded training pipeline for faster AI development
*   **NEW**: Multiple training strategies including TensorFlow distributed training and custom data parallelism
*   The AI model (`best.weights.h5`) is pre-trained.
*   Interactive and responsive Streamlit GUI.
*   Includes a "How It Works" page explaining the underlying AI components (CNN, DRL, MCTS, Self-Play).
*   Comprehensive multithreading across all training phases (self-play, training, arena)

## Technologies Used

*   Python 3
*   Streamlit (for the web GUI)
*   TensorFlow (Keras API for the Neural Network with distributed training support)
*   NumPy (for numerical operations)
*   streamlit-bridge (for enhanced HTML interaction in Streamlit)
*   Multiprocessing (for parallel self-play, arena, and training phases)

## Multithreaded Training Capabilities

This project now features **comprehensive multithreading** across all training phases:

### 🚀 Performance Improvements
- **Self-Play**: Parallelized across multiple CPU cores
- **Training**: Multiple strategies available (TensorFlow distributed, custom data parallel)
- **Arena**: Parallelized model evaluation games

### 🎯 Training Methods Available
1. **TensorFlow Distributed Training** (Recommended)
   - Automatic GPU/CPU device detection
   - Industry-standard gradient synchronization
   - Best for multi-GPU setups

2. **Data Parallel Training** (Custom)
   - Splits training data across workers
   - Averages model weights after training
   - Great for CPU-heavy systems

3. **Single-threaded Training** (Fallback)
   - Traditional sequential training
   - Best for debugging and stability

### ⚙️ Quick Configuration
Edit `src/main.py` to configure training method:
```python
# For TensorFlow distributed training (default)
'training_method': 'distributed'

# For custom data parallel training  
'training_method': 'data_parallel'
'num_training_workers': 4

# For single-threaded training
'training_method': 'single'
```

📖 **See [MULTITHREADED_TRAINING.md](MULTITHREADED_TRAINING.md) for detailed documentation**

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `requirements.txt` file lists the core libraries. For exact versions, you can update it by running `pip freeze > requirements.txt` in your activated environment after installation.*

4.  **Model:**
    The pre-trained model `best.weights.h5` is expected to be in the `./temp_connect_four/` directory. Ensure this file is present.

## Running the Application

### Playing the Game
To start the Streamlit web application, run the following command from the root directory of the project:

```bash
streamlit run 🚀_AlphaFour_Game.py
```

This will open the game in your default web browser.

### Training the AI (Advanced)
To train the AI model from scratch with multithreaded training:

```bash
cd src
python main.py
```

### Testing Multithreaded Training
To test and benchmark the different training methods:

```bash
python test_multithreaded_training.py
```

## Project Structure

```
Connect Four RL/
├── src/                     # Core AI training and game logic
│   ├── main.py             # Main training script with multithreading support
│   ├── coach.py            # AI training orchestrator with parallel phases
│   ├── model.py            # Neural Network with distributed training support
│   ├── connect_four.py     # Core Connect Four game logic
│   ├── mcts.py             # Monte Carlo Tree Search implementation
│   ├── arena.py            # Model evaluation with parallel games
│   ├── play_gui.py         # Alternative Pygame-based GUI
│   └── utils.py            # Utility functions and classes
├── pages/
│   └── 🧠_How_It_Works.py  # Streamlit page explaining AI concepts
├── temp_connect_four/
│   └── best.weights.h5      # Pre-trained neural network model
├── 🚀_AlphaFour_Game.py     # Main Streamlit application file for playing the game
├── test_multithreaded_training.py  # Performance testing script
├── MULTITHREADED_TRAINING.md        # Detailed multithreading documentation
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Performance Benchmarks

Example performance improvements on a typical development machine:

| Training Method | Time | Speedup | Best For |
|----------------|------|---------|----------|
| Single-threaded | 45.2s | 1.0x (baseline) | Debugging, compatibility |
| TensorFlow Distributed | 28.7s | 1.57x | GPU acceleration, stability |
| Data Parallel (4 workers) | 19.3s | 2.34x | CPU-heavy systems |

*Results vary based on hardware, dataset size, and model complexity*

## Further Development / Training

The project includes a complete AlphaZero-style training pipeline with:

- **Self-play data generation** (multithreaded)
- **Neural network training** (multiple multithreaded strategies) 
- **Model evaluation in arena** (multithreaded)
- **Automatic model selection** based on performance

Files like `coach.py`, `main.py`, and `arena.py` provide the infrastructure for training the AI model from scratch using a self-play loop, similar to AlphaZero. The multithreaded implementation significantly reduces training time compared to traditional sequential approaches.

To get started with training:
1. Configure your preferred training method in `src/main.py`
2. Run `python src/main.py` to start the training loop
3. Monitor progress and adjust hyperparameters as needed
4. Use `python test_multithreaded_training.py` to benchmark performance 

## Quick Start

Run the following command to start training:

```bash
python src/main.py
```

The application will detect your system's hardware capabilities and automatically configure:
- CPU usage (75% of cores by default to prevent overheating)
- GPU acceleration (if available)
- Multithreaded training based on your hardware

### 🛑 Graceful Shutdown

The training process now supports **graceful shutdown** with Ctrl+C:

- **During Self-Play**: Ctrl+C will stop after completing the current game
- **During Arena**: Ctrl+C will stop after completing the current evaluation
- **During Training**: Ctrl+C will stop after completing the current batch/epoch
- **During All Phases**: Training progress is saved before shutdown

This allows you to safely interrupt long-running training sessions without losing progress.

## Configuration

You can easily modify the training parameters by editing the `args` dictionary in `src/main.py`: 
# AlphaFour - Connect Four RL

AlphaFour is a Connect Four game where a human can play against an AI powered by Deep Reinforcement Learning. The project is inspired by DeepMind's AlphaGo and AlphaZero, adapted for the game of Connect Four. The primary interface is a web application built with Streamlit.

This project was developed by Dominic Reilly.

## Features

*   Play Connect Four against an AI opponent in your browser.
*   The AI uses a Convolutional Neural Network (CNN) to evaluate board positions and a Monte Carlo Tree Search (MCTS) algorithm to decide its moves.
*   The AI model (`best.weights.h5`) is pre-trained.
*   Interactive and responsive Streamlit GUI.
*   Includes a "How It Works" page explaining the underlying AI components (CNN, DRL, MCTS, Self-Play).

## Technologies Used

*   Python 3
*   Streamlit (for the web GUI)
*   TensorFlow (Keras API for the Neural Network)
*   NumPy (for numerical operations)
*   streamlit-bridge (for enhanced HTML interaction in Streamlit)

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

To start the Streamlit web application, run the following command from the root directory of the project:

```bash
streamlit run 🚀_AlphaFour_Game.py
```

This will open the game in your default web browser.

## Project Structure

```
Connect Four RL/
├── pages/
│   └── 🧠_How_It_Works.py  # Streamlit page explaining AI concepts
├── temp_connect_four/
│   └── best.weights.h5      # Pre-trained neural network model
├── 🚀_AlphaFour_Game.py     # Main Streamlit application file for playing the game
├── arena.py                 # (Likely for pitting models against each other)
├── coach.py                 # (Likely for the AI training loop)
├── connect_four.py          # Core Connect Four game logic
├── main.py                  # (Likely main script to initiate training or other processes)
├── mcts.py                  # Monte Carlo Tree Search implementation
├── model.py                 # Neural Network (CNN) definition using Keras/TensorFlow
├── play_gui.py              # Original Pygame-based GUI (may still have interactive learning features)
├── utils.py                 # Utility functions and classes
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Further Development / Training

Files like `coach.py`, `main.py`, and `arena.py` suggest infrastructure for training the AI model from scratch using a self-play loop, similar to AlphaZero. If you wish to retrain or experiment with training, these would be the starting points. 
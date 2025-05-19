from argparse import Namespace
import os
import signal
import multiprocessing

from coach import Coach
from connect_four import ConnectFourGame
from model import ConnectFourNNet
from utils import dotdict # Assuming a utils.py for dotdict or use Namespace directly

# Default arguments (can be overridden by command line or a config file)
args = dotdict({
    'num_iters': 10,          # Number of training iterations.
    'num_eps': 60,            # Number of self-play games to generate per iteration.
    'temp_threshold': 5,     # Number of moves after which temperature becomes 0 for action selection in self-play.
    'update_threshold': 0.50,   # Win rate threshold to accept new model in Arena.
    'max_len_of_queue': 100000, # Maximum size of the training examples deque.
    'num_mcts_sims': 200,       # Number of MCTS simulations per move.
    'arena_compare': 20,      # Number of games to play in Arena for model comparison.
    'arena_verbose': False,   # Whether to print Arena game details.
    'cpuct': 1.5,             # Exploration constant for PUCT.
    
    'num_parallel_self_play_workers': os.cpu_count(), # Number of parallel workers for self-play, defaults to num CPUs.
    'num_parallel_arena_workers': os.cpu_count(),     # Number of parallel workers for arena games, defaults to num CPUs.

    'checkpoint': './temp_connect_four/', # Folder to save checkpoints and examples.
    'load_model': True,       # Whether to load a saved model on startup.
    'load_folder_file': ('./temp_connect_four/', 'best.weights.h5'), # MODIFIED - Tuple (folder, filename) for loading model.
    'save_examples_freq': 1,  # Save training examples every N iterations.

    # Neural Network specific args
    'lr': 0.001,              # Learning rate.
    'epochs': 15,             # Number of training epochs per iteration.
    'batch_size': 64,         # Training batch size.
    'num_res_blocks': 5,      # Number of residual blocks in NNet.
    'num_channels': 64,       # Number of channels in NNet conv layers.
    # 'dropout': 0.3,         # Dropout rate (if used in model)

    # MCTS specific args for exploration during self-play
    'add_dirichlet_noise': True, # Add Dirichlet noise at the root node in self-play.
    'dirichlet_alpha': 0.3,      # Alpha parameter for Dirichlet noise.
    'epsilon_noise': 0.25,       # Weight of Dirichlet noise in root policy.
})

shutdown_event = multiprocessing.Event() # Global event for signaling shutdown

def graceful_signal_handler(sig, frame):
    print(f'Graceful shutdown initiated by signal {sig}...')
    shutdown_event.set()

def main():
    # Setup signal handlers
    signal.signal(signal.SIGINT, graceful_signal_handler)
    signal.signal(signal.SIGTERM, graceful_signal_handler)

    # Suppress TensorFlow INFO and WARNING messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0 = all, 1 = no INFO, 2 = no INFO/WARNING, 3 = no INFO/WARNING/ERROR

    print("Initializing game, neural network, and coach...")
    print("Using arguments:", args)

    game = ConnectFourGame() # Using default 6x7 board
    nnet = ConnectFourNNet(game, args)

    if args.load_model:
        print(f"Attempting to load model from: {args.load_folder_file[0]}/{args.load_folder_file[1]}")
        # Loading is handled by Coach constructor if file exists
    
    c = Coach(game, nnet, args, shutdown_event) # Pass shutdown_event to Coach
    
    print("Starting the learning process...")
    try:
        c.learn()
    except KeyboardInterrupt:
        print("Main: KeyboardInterrupt caught. Ensuring shutdown event is set.")
        shutdown_event.set()
    finally:
        if shutdown_event.is_set():
            print("Main: Learning process concluded due to shutdown signal.")
        else:
            print("Main: Learning process completed normally.")

if __name__ == "__main__":
    # For simplicity, using direct dotdict. 
    # For more complex scenarios, consider using argparse for command line argument parsing.
    
    # Example: Override an arg
    # args.num_iters = 1
    # args.num_eps = 2
    # args.arena_compare = 4
    # args.num_mcts_sims = 10
    # args.load_model = False

    main() 
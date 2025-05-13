from argparse import Namespace

from coach import Coach
from connect_four import ConnectFourGame
from model import ConnectFourNNet
from utils import dotdict # Assuming a utils.py for dotdict or use Namespace directly

# Default arguments (can be overridden by command line or a config file)
args = dotdict({
    'num_iters': 25,          # Number of training iterations.
    'num_eps': 35,            # Number of self-play games to generate per iteration.
    'temp_threshold': 10,     # Number of moves after which temperature becomes 0 for action selection in self-play.
    'update_threshold': 0.55, # Win rate threshold to accept new model in Arena.
    'max_len_of_queue': 200000, # Maximum size of the training examples deque.
    'num_mcts_sims': 200,      # Number of MCTS simulations per move.
    'arena_compare': 10,      # Number of games to play in Arena for model comparison.
    'arena_verbose': False,   # Whether to print Arena game details.
    'cpuct': 1.0,             # Exploration constant for PUCT.
    
    'checkpoint': './temp_connect_four/', # Folder to save checkpoints and examples.
    'load_model': True,       # Whether to load a saved model on startup.
    'load_folder_file': ('./temp_connect_four/', 'best.weights.h5'), # MODIFIED - Tuple (folder, filename) for loading model.
    'save_examples_freq': 1,  # Save training examples every N iterations.

    # Neural Network specific args
    'lr': 0.001,              # Learning rate.
    'epochs': 10,             # Number of training epochs per iteration.
    'batch_size': 64,         # Training batch size.
    'num_res_blocks': 5,      # Number of residual blocks in NNet.
    'num_channels': 64,       # Number of channels in NNet conv layers.
    # 'dropout': 0.3,         # Dropout rate (if used in model)

    # MCTS specific args for exploration during self-play
    'add_dirichlet_noise': True, # Add Dirichlet noise at the root node in self-play.
    'dirichlet_alpha': 0.3,      # Alpha parameter for Dirichlet noise.
    'epsilon_noise': 0.25,       # Weight of Dirichlet noise in root policy.
})

def main():
    print("Initializing game, neural network, and coach...")
    print("Using arguments:", args)

    game = ConnectFourGame() # Using default 6x7 board
    nnet = ConnectFourNNet(game, args)

    if args.load_model:
        print(f"Attempting to load model from: {args.load_folder_file[0]}/{args.load_folder_file[1]}")
        # Loading is handled by Coach constructor if file exists
    
    c = Coach(game, nnet, args)
    
    if args.load_model and c.skip_first_self_play:
        print("Model and/or examples loaded. Skipping first self-play according to Coach logic.")
    
    print("Starting the learning process...")
    c.learn()

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
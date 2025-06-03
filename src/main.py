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
    'num_iters': 30,          # Number of training iterations.
    'num_eps': 60,            # Number of self-play games to generate per iteration.
    'temp_threshold': 1,     # MODIFIED: Number of moves after which temperature becomes 0 for action selection in self-play. (Was 10)
    'update_threshold': 0.52,   # Win rate threshold to accept new model in Arena.
    'max_len_of_queue': 200000, # Maximum size of the training examples deque.
    'num_mcts_sims': 300,       # Number of MCTS simulations per move.
    'arena_compare': 20,      # Number of games to play in Arena for model comparison.
    'arena_verbose': False,   # Whether to print Arena game details.
    'cpuct': 1.5,             # Exploration constant for PUCT.
    
    # CPU Usage Control
    'cpu_usage_fraction': 1.0,  # Fraction of CPU cores to use (0.0-1.0). 0.75 = 75% of cores to reduce heating
    'max_cpu_cores': None,       # Alternative: Set absolute max number of cores to use (overrides cpu_usage_fraction if set)
    
    'num_parallel_self_play_workers': None,  # Will be calculated based on CPU limits if None
    'num_parallel_arena_workers': None,      # Will be calculated based on CPU limits if None

    'checkpoint': './src/temp_connect_four/', # Folder to save checkpoints and examples.
    'load_model': True,       # Whether to load a saved model on startup.
    'load_folder_file': ('./src/temp_connect_four/', 'best.keras'), # MODIFIED - Tuple (folder, filename) for loading model.
    'save_examples_freq': 1,  # Save training examples every N iterations.

    # Neural Network specific args
    'lr': 0.001,              # Learning rate.
    'epochs': 15,             # Number of training epochs per iteration.
    'batch_size': 64,         # Training batch size.
    'num_res_blocks': 5,      # Number of residual blocks in NNet.
    'num_channels': 64,       # Number of channels in NNet conv layers.
    # 'dropout': 0.3,         # Dropout rate (if used in model)
    
    # Distributed Training specific args
    'use_distributed_training': True,  # Enable distributed/multithreaded training for faster model training.
    'training_method': 'distributed',   # Training method: 'single', 'distributed', or 'data_parallel'
    'num_training_workers': None,       # Number of parallel training workers for data_parallel method (None = auto)

    # MCTS specific args for exploration during self-play
    'add_dirichlet_noise': True, # Add Dirichlet noise at the root node in self-play.
    'dirichlet_alpha': 0.3,      # Alpha parameter for Dirichlet noise.
    'epsilon_noise': 0.25,       # Weight of Dirichlet noise in root policy.
})

shutdown_event = multiprocessing.Event() # Global event for signaling shutdown

def calculate_cpu_workers(args):
    """
    Calculate the number of CPU workers to use based on user preferences.
    This helps control CPU usage and reduce system heating.
    
    Returns:
        int: Number of CPU workers to use
    """
    total_cores = os.cpu_count() or 1
    
    # If max_cpu_cores is explicitly set, use that (with bounds checking)
    if args.get('max_cpu_cores') is not None:
        max_cores = args.get('max_cpu_cores')
        if max_cores <= 0:
            print(f"Warning: max_cpu_cores must be positive, using 1 core instead of {max_cores}")
            return 1
        elif max_cores > total_cores:
            print(f"Warning: max_cpu_cores ({max_cores}) exceeds available cores ({total_cores}), using {total_cores}")
            return total_cores
        else:
            return max_cores
    
    # Otherwise use cpu_usage_fraction
    cpu_fraction = args.get('cpu_usage_fraction', 0.75)
    if cpu_fraction <= 0:
        print(f"Warning: cpu_usage_fraction must be positive, using 0.25 instead of {cpu_fraction}")
        cpu_fraction = 0.25
    elif cpu_fraction > 1.0:
        print(f"Warning: cpu_usage_fraction cannot exceed 1.0, using 1.0 instead of {cpu_fraction}")
        cpu_fraction = 1.0
    
    calculated_workers = max(1, int(total_cores * cpu_fraction))
    print(f"CPU Usage Control: Using {calculated_workers}/{total_cores} cores ({cpu_fraction*100:.0f}% of available CPU)")
    
    return calculated_workers

def check_gpu_availability():
    """
    Check for GPU availability and report GPU information.
    This helps users understand their hardware setup.
    """
    try:
        import tensorflow as tf
        
        # Check if TensorFlow can see any GPUs
        physical_gpus = tf.config.list_physical_devices('GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        
        print("\n=== GPU Detection Results ===")
        print(f"Physical GPUs detected: {len(physical_gpus)}")
        print(f"Logical GPUs available: {len(logical_gpus)}")
        
        if physical_gpus:
            print("\n📊 GPU Details:")
            for i, gpu in enumerate(physical_gpus):
                print(f"  GPU {i}: {gpu.name}")
                try:
                    # Try to get more detailed GPU info
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    if gpu_details:
                        for key, value in gpu_details.items():
                            print(f"    {key}: {value}")
                except:
                    pass  # Some systems may not support detailed GPU info
            
            # Check if TensorFlow is actually using GPUs
            print(f"\n🔧 TensorFlow GPU Configuration:")
            print(f"  - Built with CUDA support: {tf.test.is_built_with_cuda()}")
            print(f"  - GPU device available for TensorFlow: {tf.test.is_gpu_available()}")
            
            # Test a simple operation to see which device TensorFlow chooses
            try:
                tf.debugging.set_log_device_placement(False)  # Disable verbose logging
                with tf.device('/GPU:0'):
                    test_tensor = tf.constant([1.0, 2.0, 3.0])
                    result = tf.reduce_sum(test_tensor)
                print(f"  - GPU operations test: ✅ SUCCESS (result: {result.numpy()})")
            except Exception as e:
                print(f"  - GPU operations test: ❌ FAILED ({str(e)})")
                
        else:
            print("❌ No GPUs detected. Training will use CPU only.")
            print("   Consider checking:")
            print("   - GPU drivers are installed")
            print("   - CUDA is properly configured") 
            print("   - TensorFlow-GPU is installed")
        
        return len(physical_gpus) > 0
        
    except ImportError:
        print("❌ TensorFlow not available for GPU detection")
        return False
    except Exception as e:
        print(f"❌ Error during GPU detection: {e}")
        return False
    finally:
        print("=" * 30)

def graceful_signal_handler(sig, frame):
    print(f'Graceful shutdown initiated by signal {sig}...')
    shutdown_event.set()

def main():
    # Setup signal handlers
    signal.signal(signal.SIGINT, graceful_signal_handler)
    signal.signal(signal.SIGTERM, graceful_signal_handler)

    # Suppress TensorFlow INFO and WARNING messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0 = all, 1 = no INFO, 2 = no INFO/WARNING, 3 = no INFO/WARNING/ERROR

    print("=" * 60)
    print("🎯 AlphaFour Connect Four RL Training")
    print("=" * 60)
    
    # Check GPU availability and report findings
    gpu_available = check_gpu_availability()
    
    # Calculate CPU workers based on user settings
    cpu_workers = calculate_cpu_workers(args)
    
    # Apply CPU limits to parallel worker settings if they weren't explicitly set
    if args.get('num_parallel_self_play_workers') is None:
        args['num_parallel_self_play_workers'] = cpu_workers
    
    if args.get('num_parallel_arena_workers') is None:
        args['num_parallel_arena_workers'] = cpu_workers
    
    if args.get('num_training_workers') is None:
        args['num_training_workers'] = cpu_workers
    
    print(f"\n🔧 Configuration Summary:")
    print(f"  - Self-play workers: {args['num_parallel_self_play_workers']}")
    print(f"  - Arena workers: {args['num_parallel_arena_workers']}")
    print(f"  - Training workers: {args['num_training_workers']} (for data_parallel method)")
    print(f"  - Training method: {args['training_method']}")
    if gpu_available:
        print(f"  - GPU acceleration: ✅ Enabled")
    else:
        print(f"  - GPU acceleration: ❌ CPU only")
    
    print("\nInitializing game, neural network, and coach...")
    # print("Using arguments:", args) # Removed for cleaner output

    game = ConnectFourGame() # Using default 6x7 board
    nnet = ConnectFourNNet(game, args, shutdown_event)  # Pass shutdown_event to enable graceful training interruption

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
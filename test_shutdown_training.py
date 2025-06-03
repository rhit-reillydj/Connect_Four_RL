#!/usr/bin/env python3
"""
Test script to verify that Ctrl+C works during model training phase.
This will start a short training session that you can interrupt with Ctrl+C.
"""

import sys
import os
import multiprocessing
import time
import signal

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Create a shutdown event for testing
shutdown_event = multiprocessing.Event()

def signal_handler(sig, frame):
    """Handle Ctrl+C by setting the shutdown event."""
    print(f"\n🛑 Received signal {sig}. Setting shutdown event...")
    shutdown_event.set()

try:
    from connect_four import ConnectFourGame
    from model import ConnectFourNNet
    from utils import dotdict
    import numpy as np
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create minimal args for testing
    test_args = dotdict({
        'lr': 0.001,
        'epochs': 20,  # Still high enough to test interruption
        'batch_size': 32,
        'num_res_blocks': 2,  # Reduced for faster setup
        'num_channels': 32,   # Reduced for faster setup
        'use_distributed_training': False,  # Keep simple for testing
        'training_method': 'single'
    })
    
    print("🧪 Testing Ctrl+C handling during training...")
    print("=" * 50)
    print("This test will start training for 20 epochs.")
    print("Press Ctrl+C at any time during training to test graceful shutdown.")
    print("You should see the training stop gracefully with a shutdown message.")
    print("=" * 50)
    
    # Create game and neural network
    game = ConnectFourGame()
    nnet = ConnectFourNNet(game, test_args, shutdown_event)
    
    # Generate some dummy training examples
    print("Generating test training examples...")
    board_x, board_y = game.get_board_size()
    action_size = game.get_action_size()
    
    num_examples = 500  # More examples for longer training time
    examples = []
    for i in range(num_examples):
        # Random board state
        board = np.random.randint(0, 3, size=(board_x, board_y))
        # Random policy (but normalized)
        policy = np.random.random(action_size)
        policy = policy / policy.sum()
        # Random value
        value = np.random.uniform(-1, 1)
        
        examples.append([board, policy, value])
    
    print(f"Generated {num_examples} training examples")
    print("\n🏁 Starting training...")
    print("Press Ctrl+C to test graceful shutdown!\n")
    
    # Start training
    history = nnet.train(examples)
    
    if shutdown_event.is_set():
        print("✅ Training was interrupted by shutdown signal!")
    else:
        print("✅ Training completed normally (no interruption)")
    
    print("\n🎯 Test completed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all required modules are available.")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 
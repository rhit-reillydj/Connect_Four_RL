import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, Dense, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
import numpy as np
import multiprocessing
import os
import tempfile
import time

def residual_block(x, filters, kernel_size=(3,3)):
    """Defines a residual block.
    Args:
        x: Input tensor.
        filters (int): Number of filters for the convolutional layers.
        kernel_size (tuple): Kernel size for the convolutional layers.
    Returns:
        Tensor output from the residual block.
    """
    shortcut = x

    # First component of main path
    y = Conv2D(filters, kernel_size, padding='same')(x)
    y = BatchNormalization(axis=3)(y)
    y = ReLU()(y)

    # Second component of main path
    y = Conv2D(filters, kernel_size, padding='same')(y)
    y = BatchNormalization(axis=3)(y)
    
    # Add shortcut to main path
    y = Add()([shortcut, y])
    y = ReLU()(y)
    
    return y

def _train_worker_data_parallel(worker_args):
    """
    Worker function for parallel data training.
    Must be at module level for multiprocessing.
    """
    board_x, board_y, action_size, args, examples, baseline_weights_path, output_weights_path, worker_id, shutdown_event = worker_args
    
    try:
        print(f"Worker {worker_id}: Starting training on {len(examples)} examples")
        
        # Suppress TensorFlow logging for workers
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        # Check if shutdown was requested before starting
        if shutdown_event and shutdown_event.is_set():
            print(f"Worker {worker_id}: Shutdown requested before training start")
            return None
        
        # Load the baseline model
        model = tf.keras.models.load_model(baseline_weights_path)
        
        # Prepare training data
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards).reshape((-1, board_x, board_y, 1))
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        
        # Train the model
        batch_size = args.get('batch_size', 64)
        epochs = args.get('epochs', 10)
        
        # Use fewer epochs for parallel workers to reduce training time
        worker_epochs = max(1, epochs // 2)
        
        # Create shutdown callback for worker if shutdown_event is provided
        callbacks = []
        if shutdown_event:
            shutdown_callback = ShutdownCallback(shutdown_event)
            callbacks.append(shutdown_callback)
        
        history = model.fit(
            input_boards,
            {'policy_head': target_pis, 'value_head': target_vs},
            batch_size=batch_size,
            epochs=worker_epochs,
            callbacks=callbacks,
            verbose=0  # Suppress output for cleaner logging
        )
        
        # Check if shutdown was requested after training
        if shutdown_event and shutdown_event.is_set():
            print(f"Worker {worker_id}: Shutdown requested after training")
            return None
        
        # Save the trained model
        model.save(output_weights_path)
        
        print(f"Worker {worker_id}: Training completed, final loss: {history.history['loss'][-1]:.4f}")
        return output_weights_path
        
    except Exception as e:
        print(f"Worker {worker_id}: Training failed with error: {e}")
        return None

class ShutdownCallback(Callback):
    """
    Custom callback that monitors the shutdown event and stops training when Ctrl+C is pressed.
    """
    def __init__(self, shutdown_event):
        super().__init__()
        self.shutdown_event = shutdown_event
        
    def on_batch_end(self, batch, logs=None):
        """Check for shutdown signal after each batch."""
        if self.shutdown_event.is_set():
            print("\n🛑 Shutdown signal detected during training. Stopping training gracefully...")
            self.model.stop_training = True
    
    def on_epoch_end(self, epoch, logs=None):
        """Check for shutdown signal after each epoch."""
        if self.shutdown_event.is_set():
            print(f"\n🛑 Shutdown signal detected after epoch {epoch + 1}. Stopping training gracefully...")
            self.model.stop_training = True

class ConnectFourNNet():
    def __init__(self, game, args, shutdown_event=None):
        """
        Initialize the Neural Network for Connect Four.
        Args:
            game: An instance of the ConnectFourGame class.
            args: A dictionary or an argparse object containing hyperparameters.
                  Expected keys in args:
                  lr (float): Learning rate.
                  dropout (float): Dropout rate (currently not used in this basic cnn).
                  epochs (int): Number of training epochs.
                  batch_size (int): Training batch size.
                  num_channels (int): Number of channels in conv layers (e.g. 512 in AlphaZero).
                                     For this simpler model, we define them explicitly.
                  use_distributed_training (bool): Whether to use distributed training strategy.
            shutdown_event: Multiprocessing event for graceful shutdown during training.
        """
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args # Store args for potential use in other methods
        self.shutdown_event = shutdown_event  # Store shutdown event for training interruption
        num_res_blocks = self.args.get('num_res_blocks', 5) # Number of residual blocks
        num_channels = self.args.get('num_channels', 64)   # Number of channels in conv layers

        # Configure GPU memory growth to prevent TensorFlow from consuming all GPU memory
        self._configure_gpu_memory()
        
        # Initialize distributed training strategy
        self.use_distributed = self.args.get('use_distributed_training', False)
        self.strategy = None
        
        if self.use_distributed and self.args.get('training_method', 'single') == 'distributed':
            try:
                # Create MirroredStrategy for distributed training
                self.strategy = tf.distribute.MirroredStrategy()
                print(f"Distributed training enabled with {self.strategy.num_replicas_in_sync} replica(s)")
                
                # Build model within strategy scope
                with self.strategy.scope():
                    self._create_model(num_channels, num_res_blocks)
            except Exception as e:
                print(f"Warning: Failed to initialize distributed strategy: {e}")
                print("Falling back to single-device training.")
                self.use_distributed = False
                self.strategy = None
                self._create_model(num_channels, num_res_blocks)
        else:
            # Standard single-device training
            self._create_model(num_channels, num_res_blocks)
        
        print(f"Neural network initialized. Use distributed: {self.use_distributed}")
    
    def _configure_gpu_memory(self):
        """
        Configure GPU memory settings to prevent TensorFlow from consuming all GPU memory.
        This allows other applications to use the GPU and prevents out-of-memory errors.
        """
        try:
            # List all physical GPUs
            gpus = tf.config.list_physical_devices('GPU')
            
            if gpus:
                try:
                    # Enable memory growth for all GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
                    
                    # Optionally set memory limit (uncomment if you want to limit GPU memory)
                    # memory_limit = 1024 * 4  # 4GB limit example
                    # for gpu in gpus:
                    #     tf.config.experimental.set_memory_limit(gpu, memory_limit)
                    #     print(f"GPU memory limit set to {memory_limit}MB")
                        
                except RuntimeError as e:
                    # Virtual devices must be set before GPUs have been initialized
                    print(f"⚠️  GPU memory configuration failed: {e}")
                    print("This is normal if TensorFlow has already been initialized.")
            else:
                print("No GPUs detected for memory configuration.")
                
        except ImportError:
            print("TensorFlow not available for GPU memory configuration.")
        except Exception as e:
            print(f"Error configuring GPU memory: {e}")

    def _create_model(self, num_channels, num_res_blocks):
        """Create the neural network model."""
        # Neural Net
        input_boards = Input(shape=(self.board_x, self.board_y, 1)) # s: batch_size x board_x x board_y x 1

        # Initial Convolutional Layer
        x = Conv2D(filters=num_channels, kernel_size=(3, 3), padding='same')(input_boards)
        x = BatchNormalization(axis=3)(x)
        x = ReLU()(x)

        # Residual Blocks
        for _ in range(num_res_blocks):
            x = residual_block(x, filters=num_channels)

        # Policy Head
        pi = Conv2D(filters=2, kernel_size=(1, 1), padding='same')(x) # Reduced filters based on AlphaGo Zero paper (page 13)
        pi = BatchNormalization(axis=3)(pi)
        pi = ReLU()(pi)
        pi = Flatten()(pi)
        self.pi = Dense(self.action_size, activation='softmax', name='policy_head')(pi)

        # Value Head
        v = Conv2D(filters=1, kernel_size=(1, 1), padding='same')(x) # Reduced filters based on AlphaGo Zero paper
        v = BatchNormalization(axis=3)(v)
        v = ReLU()(v)
        v = Flatten()(v)
        v = Dense(num_channels, activation='relu')(v) # Intermediate dense layer
        self.v = Dense(1, activation='tanh', name='value_head')(v)

        self.model = Model(inputs=input_boards, outputs=[self.pi, self.v])
        
        # Define losses for each head
        losses = {
            'policy_head': 'categorical_crossentropy',
            'value_head': 'mean_squared_error'
        }
        # Optional: Define loss weights if one loss is more important
        loss_weights = {
            'policy_head': 1.0,
            'value_head': 1.0
        }

        # Adjust learning rate for distributed training
        base_lr = self.args.get('lr', 0.001)
        if self.use_distributed and self.strategy:
            # Scale learning rate by number of replicas for distributed training
            effective_lr = base_lr * self.strategy.num_replicas_in_sync
            print(f"Scaling learning rate from {base_lr} to {effective_lr} for {self.strategy.num_replicas_in_sync} replicas")
        else:
            effective_lr = base_lr

        self.model.compile(
            optimizer=Adam(learning_rate=effective_lr), 
            loss=losses,
            loss_weights=loss_weights,
            jit_compile=True
        )

    def train(self, examples):
        """
        Train the neural network using TensorFlow's distributed strategy.
        Now includes GPU utilization reporting.
        
        Args:
            examples (list): List of training examples.
        """
        if not examples:
            print("No training examples provided, skipping training.")
            return
        
        # Check which devices are being used
        try:
            available_devices = tf.config.list_logical_devices()
            gpus = [d for d in available_devices if d.device_type == 'GPU']
            cpus = [d for d in available_devices if d.device_type == 'CPU']
            
            print(f"\n🔧 Training Device Info:")
            print(f"  - Available CPUs: {len(cpus)}")
            print(f"  - Available GPUs: {len(gpus)}")
            if gpus:
                print(f"  - GPU names: {[gpu.name for gpu in gpus]}")
                print(f"  - Training strategy: Distributed (Multi-GPU)" if len(gpus) > 1 else f"  - Training strategy: Single GPU")
            else:
                print(f"  - Training strategy: CPU only")
        except Exception as e:
            print(f"  - Device info unavailable: {e}")
        
        # Create distributed strategy if not already created
        if not hasattr(self, 'strategy'):
            if self.args.get('use_distributed_training', True) and self.args.get('training_method', 'distributed') == 'distributed':
                try:
                    # Try to use MirroredStrategy for multi-GPU training
                    self.strategy = tf.distribute.MirroredStrategy()
                    print(f"  - Using MirroredStrategy with {self.strategy.num_replicas_in_sync} replica(s)")
                except Exception as e:
                    print(f"  - Failed to create MirroredStrategy: {e}")
                    print(f"  - Falling back to single-device training")
                    self.strategy = None
            else:
                self.strategy = None
        
        print(f"Current LR before training: {tf.keras.backend.get_value(self.model.optimizer.learning_rate)}")
        input_boards = np.array([e[0] for e in examples], dtype=np.float32)
        target_pis = np.array([e[1] for e in examples], dtype=np.float32)
        target_vs = np.array([e[2] for e in examples], dtype=np.float32)

        # Create the model within the strategy scope if using distributed training
        if self.strategy:
            with self.strategy.scope():
                # Model should already be created within strategy scope
                if not hasattr(self, 'model') or self.model is None:
                    self._build_model()  # Rebuild model in strategy scope if needed
        
        # Adjust batch size for distributed training
        base_batch_size = self.args.get('batch_size', 64)
        if self.strategy:
            # Global batch size should be divisible by number of replicas
            global_batch_size = base_batch_size * self.strategy.num_replicas_in_sync
            print(f"Using global batch size {global_batch_size} ({base_batch_size} per replica x {self.strategy.num_replicas_in_sync} replicas)")
        else:
            global_batch_size = base_batch_size

        # Define callbacks
        reduce_lr = ReduceLROnPlateau(
            monitor='loss',      # Monitor the total training loss
            factor=0.2,          # Factor by which the learning rate will be reduced. new_lr = lr * factor
            patience=10,         # MODIFIED: Number of epochs with no improvement after which learning rate will be reduced (was 5).
            min_lr=0.00005,      # MODIFIED: Lower bound on the learning rate (was 0.00001).
            verbose=1
        )

        # Add shutdown callback if shutdown event is available
        callbacks = [reduce_lr]
        if self.shutdown_event is not None:
            shutdown_callback = ShutdownCallback(self.shutdown_event)
            callbacks.append(shutdown_callback)
            print("🛑 Shutdown monitoring enabled during training (Ctrl+C will gracefully stop training)")

        print(f"Training with distributed strategy: {self.use_distributed}")
        print(f"Training on {len(examples)} examples...")

        # Create tf.data.Dataset for better performance with distributed training
        if self.strategy:
            # Create dataset for distributed training
            dataset = tf.data.Dataset.from_tensor_slices((
                input_boards,
                {'policy_head': target_pis, 'value_head': target_vs}
            ))
            
            # Batch and distribute the dataset
            dataset = dataset.batch(global_batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            # Distribute the dataset
            dist_dataset = self.strategy.experimental_distribute_dataset(dataset)
            
            # Training with distributed dataset
            history = self.model.fit(
                dist_dataset,
                epochs=self.args.get('epochs', 10),
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Standard training for single device
            history = self.model.fit(
                input_boards, 
                {'policy_head': target_pis, 'value_head': target_vs},
                batch_size=global_batch_size,
                epochs=self.args.get('epochs', 10),
                callbacks=callbacks
            )
        
        return history

    def predict(self, canonical_board):
        """
        Predict policy and value for a given canonical board state.
        Args:
            canonical_board (np.ndarray): The board state from the current player's perspective.
        Returns:
            tuple: (policy, value)
                     policy (np.ndarray): Probability distribution over actions.
                     value (float): Estimated value of the board state for the current player.
        """
        board = canonical_board.reshape(-1, self.board_x, self.board_y, 1)
        pi, v = self.model.predict(board, verbose=0) # verbose=0 to suppress prediction logging
        
        return pi[0], v[0][0] # pi is (1, action_size), v is (1,1)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.keras'):
        """
        Save the current model (including architecture, weights, and optimizer state) to a file.
        Args:
            folder (str): The directory to save the checkpoint.
            filename (str): The name of the checkpoint file (e.g., .keras).
        """
        import os
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, filename)
        self.model.save(filepath)
        print(f"Model checkpoint saved to {filepath}")

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.keras'):
        """
        Load a model (including architecture, weights, and optimizer state) from a .keras file,
        or just weights from a .h5 file.
        Args:
            folder (str): The directory where the checkpoint is saved.
            filename (str): The name of the checkpoint file (e.g., .keras or .h5).
        """
        import os
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            if filename.endswith('.keras'):
                self.model = tf.keras.models.load_model(filepath)
                print(f"Full model loaded from {filepath} (includes optimizer state).")
                
                # Check and potentially reset learning rate if it's too low
                current_lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
                print(f"Loaded model's current LR: {current_lr:.7f}")
                
                escape_threshold_lr = 0.00004 # If LR is below this (e.g., the old 1e-5)
                reset_to_lr = 0.00005       # Reset to this value (our new min_lr for ReduceLROnPlateau)

                if current_lr < escape_threshold_lr:
                    print(f"Current LR {current_lr:.7f} is below escape threshold {escape_threshold_lr:.7f}. Setting LR to {reset_to_lr:.7f}.")
                    self.model.optimizer.learning_rate.assign(reset_to_lr)
                    print(f"New LR after reset: {tf.keras.backend.get_value(self.model.optimizer.learning_rate):.7f}")
                
            elif filename.endswith('.h5'): # For backward compatibility with .weights.h5
                # Ensure model is built before loading weights if it hasn't been used yet.
                # A simple predict call on a dummy input can build it.
                # However, self.model.load_weights() should work if the model architecture is already defined.
                try:
                    self.model.load_weights(filepath)
                    print(f"Model weights loaded from {filepath} (optimizer state not included).")
                except Exception as e:
                    print(f"Error loading weights from {filepath}: {e}")
                    print("This might be due to an architecture mismatch or if the model wasn't compiled/built.")
                    print("Ensure the model architecture matches the saved weights.")
            else:
                print(f"Unsupported file format: {filename}. Please use .keras or .h5")
        else:
            print(f"No model checkpoint found at {filepath}")

    def train_parallel_data(self, examples, num_workers=None):
        """
        Alternative multithreaded training using data parallelism.
        Splits training data across multiple processes, trains separate models,
        and averages the resulting weights.
        
        Args:
            examples (list): List of training examples.
            num_workers (int): Number of parallel training workers. If None, uses CPU count.
        """
        if num_workers is None:
            num_workers = min(os.cpu_count(), 4)  # Cap at 4 to avoid memory issues
        
        # If we have fewer examples than workers, fall back to single training
        if len(examples) < num_workers * 2:
            print(f"Too few examples ({len(examples)}) for {num_workers} workers. Using single-threaded training.")
            return self.train(examples)
        
        # Check if shutdown was requested before starting parallel training
        if self.shutdown_event and self.shutdown_event.is_set():
            print("🛑 Shutdown requested before parallel training start")
            return None
        
        print(f"Starting parallel data training with {num_workers} workers on {len(examples)} examples...")
        if self.shutdown_event:
            print("🛑 Shutdown monitoring enabled for parallel training workers")
        
        # Save current model weights as baseline
        temp_dir = tempfile.mkdtemp()
        baseline_weights_path = os.path.join(temp_dir, 'baseline_weights.keras')
        self.model.save(baseline_weights_path)
        
        # Split examples into chunks for each worker
        chunk_size = len(examples) // num_workers
        example_chunks = []
        for i in range(num_workers):
            start_idx = i * chunk_size
            if i == num_workers - 1:  # Last worker gets remaining examples
                end_idx = len(examples)
            else:
                end_idx = (i + 1) * chunk_size
            example_chunks.append(examples[start_idx:end_idx])
        
        # Prepare worker arguments
        worker_args = []
        for i, chunk in enumerate(example_chunks):
            worker_weights_path = os.path.join(temp_dir, f'worker_{i}_weights.keras')
            worker_args.append((
                self.board_x, self.board_y, self.action_size,
                self.args, chunk, baseline_weights_path, worker_weights_path, i, self.shutdown_event
            ))
        
        # Run parallel training workers
        start_time = time.time()
        try:
            with multiprocessing.Pool(processes=num_workers) as pool:
                results = pool.map(_train_worker_data_parallel, worker_args)
            
            # Check if shutdown was requested during training
            if self.shutdown_event and self.shutdown_event.is_set():
                print("🛑 Shutdown requested during parallel training")
                return None
            
            # Check if all workers completed successfully
            successful_workers = [path for path in results if path is not None]
            
            if len(successful_workers) == 0:
                print("No workers completed successfully. Falling back to single-threaded training.")
                return self.train(examples)
            
            print(f"Parallel training completed in {time.time() - start_time:.2f}s")
            print(f"Successfully trained {len(successful_workers)}/{num_workers} workers")
            
            # Average the weights from successful workers
            self._average_model_weights(successful_workers)
            
        except Exception as e:
            print(f"Error during parallel training: {e}")
            print("Falling back to single-threaded training.")
            return self.train(examples)
        finally:
            # Clean up temporary files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _average_model_weights(self, weight_file_paths):
        """Average weights from multiple trained models."""
        print(f"Averaging weights from {len(weight_file_paths)} models...")
        
        # Load weights from all models
        all_weights = []
        for weight_path in weight_file_paths:
            try:
                temp_model = tf.keras.models.load_model(weight_path)
                all_weights.append(temp_model.get_weights())
            except Exception as e:
                print(f"Error loading weights from {weight_path}: {e}")
        
        if not all_weights:
            print("No valid weight files to average!")
            return
        
        # Average the weights
        averaged_weights = []
        for layer_idx in range(len(all_weights[0])):
            layer_weights = [weights[layer_idx] for weights in all_weights]
            averaged_layer_weights = np.mean(layer_weights, axis=0)
            averaged_weights.append(averaged_layer_weights)
        
        # Set the averaged weights to our model
        self.model.set_weights(averaged_weights)
        print("Weight averaging completed.")

# Example usage (if you want to test the class directly)
if __name__ == '__main__':
    pass # Add a pass statement or example code here 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, Dense, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np

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

class ConnectFourNNet():
    def __init__(self, game, args):
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
        """
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args # Store args for potential use in other methods
        num_res_blocks = self.args.get('num_res_blocks', 5) # Number of residual blocks
        num_channels = self.args.get('num_channels', 64)   # Number of channels in conv layers

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

        self.model.compile(
            optimizer=Adam(learning_rate=self.args.get('lr', 0.001)), 
            loss=losses,
            loss_weights=loss_weights
        )

    def train(self, examples):
        """
        Train the neural network using a list of examples.
        Args:
            examples (list): List of training examples, where each example is a tuple
                             (canonical_board, mcts_policy, value).
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        
        # Reshape input_boards for the network: (num_examples, board_x, board_y, 1)
        input_boards = np.asarray(input_boards)
        # Ensure boards are in canonical form if not already
        # Assuming examples provide canonical_board directly as per AlphaZero standard practice
        input_boards = input_boards.reshape((-1, self.board_x, self.board_y, 1))
        
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)

        # Define callbacks
        reduce_lr = ReduceLROnPlateau(
            monitor='loss',      # Monitor the total training loss
            factor=0.2,          # Factor by which the learning rate will be reduced. new_lr = lr * factor
            patience=5,          # Number of epochs with no improvement after which learning rate will be reduced.
            min_lr=0.00001,      # Lower bound on the learning rate.
            verbose=1
        )

        history = self.model.fit(
            input_boards, 
            {'policy_head': target_pis, 'value_head': target_vs},
            batch_size=self.args.get('batch_size', 64),
            epochs=self.args.get('epochs', 10),
            callbacks=[reduce_lr] # Add callback here
        )
        # print(f"Training loss: {history.history['loss'][-1]}")

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

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.h5'):
        """
        Save the current model weights to a checkpoint file.
        Args:
            folder (str): The directory to save the checkpoint.
            filename (str): The name of the checkpoint file (e.g., .h5 for Keras).
        """
        import os
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, filename)
        self.model.save_weights(filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.h5'):
        """
        Load model weights from a checkpoint file.
        Args:
            folder (str): The directory where the checkpoint is saved.
            filename (str): The name of the checkpoint file.
        """
        import os
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            self.model.load_weights(filepath)
            print(f"Checkpoint loaded from {filepath}")
        else:
            print(f"No checkpoint found at {filepath}") 
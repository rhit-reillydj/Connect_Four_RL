import os
import tensorflow as tf
from model import ConnectFourNNet
from connect_four import ConnectFourGame
from utils import dotdict # Make sure utils.py with dotdict is in your Python path or same directory

# --- Configuration ---
# Args needed for ConnectFourNNet initialization, matching your main.py defaults
# Ensure these match the settings used when the .h5 weights were originally saved.
conversion_args = dotdict({
    'lr': 0.001,              # Learning rate (will be used to compile the new model)
    'num_res_blocks': 5,      # Number of residual blocks in NNet
    'num_channels': 64,       # Number of channels in NNet conv layers
    # Add any other args YOUR ConnectFourNNet specifically uses from 'args' 
    # during its __init__ that affect model structure or compilation,
    # if they differ from these common ones.
})

weights_folder = './temp_connect_four/' # Folder where your checkpoints are stored
old_weights_filename = 'best.weights.h5' 
new_model_filename = 'best.keras'       

# --- Conversion Process ---
def convert():
    print("Starting weight to model conversion...")

    # 1. Initialize the game and network architecture
    print("Initializing game and NNet architecture...")
    try:
        game = ConnectFourGame() # Assuming default game setup is sufficient
        nnet = ConnectFourNNet(game, conversion_args)
        print("NNet architecture created and compiled.")
    except Exception as e:
        print(f"Error initializing Game/NNet: {e}")
        print("Please ensure connect_four.py, model.py, and utils.py are accessible and correct.")
        return

    # 2. Load the old weights into the NNet instance
    weights_filepath = os.path.join(weights_folder, old_weights_filename)
    if os.path.exists(weights_filepath):
        print(f"Loading weights from: {weights_filepath}")
        try:
            # The model in nnet is already compiled. We load weights into it.
            nnet.model.load_weights(weights_filepath)
            print("Weights loaded successfully into the model architecture.")
        except Exception as e:
            print(f"Error loading weights from {weights_filepath}: {e}")
            print("Ensure the weights file is compatible with the current model architecture defined by conversion_args.")
            return
    else:
        print(f"Error: Weights file not found at {weights_filepath}")
        print(f"Please ensure '{old_weights_filename}' exists in the '{weights_folder}' directory.")
        return

    # 3. Save the full model (architecture + loaded weights + new optimizer state)
    # The nnet.model now has the architecture, the loaded weights, and a freshly initialized optimizer (from nnet.__init__)
    model_filepath = os.path.join(weights_folder, new_model_filename)
    print(f"Saving full model to: {model_filepath}")
    try:
        # Use the save_checkpoint method from ConnectFourNNet which now uses model.save()
        nnet.save_checkpoint(folder=weights_folder, filename=new_model_filename) 
        # This internally calls self.model.save()
        print(f"Successfully converted and saved model to {model_filepath}")
        print(f"You can now use '{new_model_filename}' with your updated training script, which expects .keras files.")
    except Exception as e:
        print(f"Error saving the full model to {model_filepath}: {e}")
        return

    print("Conversion process finished.")

if __name__ == '__main__':
    # Suppress TensorFlow INFO and WARNING messages for cleaner output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    tf.get_logger().setLevel('ERROR') # Suppress Keras/TF INFO logs too

    convert() 
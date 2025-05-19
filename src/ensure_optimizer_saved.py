import os
import tensorflow as tf
from model import ConnectFourNNet
from connect_four import ConnectFourGame
from utils import dotdict

# Args needed for ConnectFourNNet initialization
MODEL_ARGS = dotdict({
    'lr': 0.001,
    'num_res_blocks': 5,
    'num_channels': 64,
    'epochs': 15, # Default from main.py, used during model.compile implicitly
    'batch_size': 64, # Default from main.py
})

MODEL_FOLDER_ROOT = './src/temp_connect_four/' # Relative to workspace root
OLD_KERAS_FILENAME = 'best.keras'
OLD_H5_FILENAME = 'best.weights.h5'
NEW_KERAS_FILENAME = 'best.keras' # Overwrite existing best.keras

def ensure_optimizer_state():
    print("Starting optimizer state consolidation process...")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow INFO/WARNING
    tf.get_logger().setLevel('ERROR')

    game = ConnectFourGame()
    # This compiles the model with a fresh optimizer
    nnet = ConnectFourNNet(game, MODEL_ARGS)

    keras_path_root = os.path.join(MODEL_FOLDER_ROOT, OLD_KERAS_FILENAME)
    h5_path_root = os.path.join(MODEL_FOLDER_ROOT, OLD_H5_FILENAME)
    new_keras_path_root = os.path.join(MODEL_FOLDER_ROOT, NEW_KERAS_FILENAME)

    loaded_something = False

    if os.path.exists(keras_path_root):
        print(f"Attempting to load full model from: {keras_path_root}")
        try:
            nnet.load_checkpoint(folder=MODEL_FOLDER_ROOT, filename=OLD_KERAS_FILENAME)
            print(f"Successfully loaded full model from {OLD_KERAS_FILENAME}.")
            loaded_something = True
        except Exception as e:
            print(f"Error loading full model from {keras_path_root}: {e}")
            print("Proceeding to check for H5 weights file.")
    else:
        print(f"Full model file {keras_path_root} not found.")

    if not loaded_something and os.path.exists(h5_path_root):
        print(f"Attempting to load H5 weights from: {h5_path_root}")
        try:
            nnet.load_checkpoint(folder=MODEL_FOLDER_ROOT, filename=OLD_H5_FILENAME)
            print(f"Successfully loaded H5 weights into the model architecture from {OLD_H5_FILENAME}.")
            loaded_something = True
        except Exception as e:
            print(f"Error loading H5 weights from {h5_path_root}: {e}")
    elif not loaded_something:
        print(f"H5 weights file {h5_path_root} also not found.")

    if not loaded_something:
        print("No existing model or weights file found. Cannot proceed.")
        print(f"Please ensure '{OLD_KERAS_FILENAME}' or '{OLD_H5_FILENAME}' exists in '{MODEL_FOLDER_ROOT}'.")
        return

    print(f"Attempting to save the consolidated model to: {new_keras_path_root}")
    try:
        nnet.save_checkpoint(folder=MODEL_FOLDER_ROOT, filename=NEW_KERAS_FILENAME)
        print(f"Successfully saved consolidated model to {new_keras_path_root}.")
        print("This new file should contain the architecture, weights, and the full optimizer state.")
        print("Try running your training again. The Keras optimizer warnings should be resolved after the first load of this new file.")
    except Exception as e:
        print(f"Error saving the consolidated model: {e}")

if __name__ == '__main__':
    ensure_optimizer_state() 
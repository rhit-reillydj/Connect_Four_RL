import tensorflow as tf
import numpy as np
import os

class ConnectFourNNetTFLite:
    def __init__(self, game, model_filename="model.tflite", model_dir="."):
        """
        Initialize the TFLite Neural Network for Connect Four.
        Args:
            game: An instance of the ConnectFourGame class.
            model_filename (str): The name of the TFLite model file.
            model_dir (str): The directory where the TFLite model is located.
        """
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        
        model_path = os.path.join(model_dir, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found at {model_path}. Please run the conversion script first.")

        print(f"Loading TFLite model from: {model_path}")
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            # Assuming the Keras model had outputs named 'policy_head' and 'value_head'
            # TFLite interpreter output details might not retain names as directly, order is more reliable.
            # Typically, policy is the first output and value is the second, based on AlphaZero designs.
            # Verify this order if you encounter issues.
            self.policy_output_index = self.interpreter.get_output_details()[0]['index']
            self.value_output_index = self.interpreter.get_output_details()[1]['index']
            print("TFLite model loaded and interpreter initialized.")
            # print(f"Input details: {self.input_details}")
            # print(f"Policy output details: {self.interpreter.get_output_details()[0]}")
            # print(f"Value output details: {self.interpreter.get_output_details()[1]}")

        except Exception as e:
            print(f"Error loading TFLite model or initializing interpreter: {e}")
            raise

    def predict(self, canonical_board):
        """
        Predict policy and value for a given canonical board state using TFLite interpreter.
        Args:
            canonical_board (np.ndarray): The board state from the current player's perspective.
        Returns:
            tuple: (policy, value)
                     policy (np.ndarray): Probability distribution over actions.
                     value (float): Estimated value of the board state for the current player.
        """
        # Ensure input is float32 and has the correct shape (batch_size, height, width, channels)
        # The TFLite model expects a batch dimension, so reshape to (1, H, W, C)
        board_input = canonical_board.astype(np.float32).reshape(self.input_details[0]['shape'])
        
        self.interpreter.set_tensor(self.input_details[0]['index'], board_input)
        self.interpreter.invoke()
        
        policy = self.interpreter.get_tensor(self.policy_output_index)[0]  # Output shape (1, action_size)
        value = self.interpreter.get_tensor(self.value_output_index)[0][0] # Output shape (1, 1)
        
        return policy, value

# Example usage (for testing purposes, assuming you have a Game class and a model.tflite):
# if __name__ == '__main__':
#     class MockGame:
#         def get_board_size(self):
#             return (6, 7) # Example for Connect Four
#         def get_action_size(self):
#             return 7      # Example for Connect Four
# 
#     # Ensure model.tflite exists in the current directory or run converter_script.py
#     if not os.path.exists("model.tflite"):
#         print("model.tflite not found. Please run converter_script.py first.")
#     else:
#         game_instance = MockGame()
#         try:
#             tflite_nnet = ConnectFourNNetTFLite(game_instance)
#             print("TFLite NNet initialized successfully.")
#             
#             # Create a dummy board for prediction
#             dummy_board = np.zeros((6, 7), dtype=np.float32)
#             policy, value = tflite_nnet.predict(dummy_board)
#             
#             print(f"Predicted Policy: {policy}")
#             print(f"Predicted Value: {value}")
#             
#         except Exception as e:
#             print(f"Error during TFLite NNet example usage: {e}") 
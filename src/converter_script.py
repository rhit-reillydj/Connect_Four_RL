import tensorflow as tf
import os

# Define the path to the Keras model and the output TFLite model path
KERAS_MODEL_DIR = './src/temp_connect_four/' # Path relative to project root
KERAS_MODEL_FILENAME = 'best.keras'
TFLITE_MODEL_OUTPUT_DIR = './src/temp_connect_four/' # Path relative to project root
TFLITE_MODEL_FILENAME = 'model.tflite'

# Adjust paths to be relative to the script's new location (src/) if needed,
# or ensure they are absolute/correctly relative to the project root when script is run.
# The current ./src/... paths assume the script is run from the project root.
# If running from src/, these paths would need to be ../src/temp_connect_four
# For simplicity, assuming it will still be run from project root like `python src/converter_script.py`

keras_model_path = os.path.join(KERAS_MODEL_DIR, KERAS_MODEL_FILENAME)
tflite_model_path = os.path.join(TFLITE_MODEL_OUTPUT_DIR, TFLITE_MODEL_FILENAME)

def convert_model():
    """Loads a Keras model and converts it to TFLite format."""
    # Ensure Keras model path is correct relative to where script is run
    # If script is in src/ and KERAS_MODEL_DIR is ./src/temp_connect_four/ (from root)
    # then the path is fine if run from root.
    # If script is in src/ and run from src/, path should be ./temp_connect_four/
    # Let's make paths more robust assuming script is run from project root.

    if not os.path.exists(keras_model_path):
        print(f"Error: Keras model not found at {keras_model_path} (relative to project root)")
        print("Please ensure you have a trained 'best.keras' model in the specified directory.")
        return

    print(f"Loading Keras model from: {keras_model_path}")
    try:
        model = tf.keras.models.load_model(keras_model_path)
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return

    print("Converting model to TensorFlow Lite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optional: Apply optimizations. DEFAULT includes quantization if beneficial and supported.
    # converter.optimizations = [tf.lite.Optimize.DEFAULT] # COMMENTED OUT TO TEST WITHOUT OPTIMIZATIONS

    tflite_model_content = converter.convert()

    print(f"Saving TFLite model to: {tflite_model_path} (relative to project root)")
    # Ensure the output directory exists
    os.makedirs(TFLITE_MODEL_OUTPUT_DIR, exist_ok=True)
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model_content)
    print("TensorFlow Lite model saved successfully.")
    print(f"You can now use this '{TFLITE_MODEL_FILENAME}' in your Streamlit application with a TFLite interpreter from '{TFLITE_MODEL_OUTPUT_DIR}'.")

if __name__ == '__main__':
    convert_model() 
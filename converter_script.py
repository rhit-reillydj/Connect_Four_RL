import tensorflow as tf
import os

# Define the path to the Keras model and the output TFLite model path
KERAS_MODEL_DIR = './src/temp_connect_four/'
KERAS_MODEL_FILENAME = 'best.keras'
TFLITE_MODEL_FILENAME = 'model.tflite' # Save in the root directory

keras_model_path = os.path.join(KERAS_MODEL_DIR, KERAS_MODEL_FILENAME)
tflite_model_path = TFLITE_MODEL_FILENAME # Save in root

def convert_model():
    """Loads a Keras model and converts it to TFLite format."""
    if not os.path.exists(keras_model_path):
        print(f"Error: Keras model not found at {keras_model_path}")
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
    # For float16 quantization (smaller model, faster on compatible hardware):
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]
    
    # For dynamic range quantization (weights quantized, activations float - good balance):
    # converter.optimizations = [tf.lite.Optimize.DEFAULT] # COMMENTED OUT TO TEST WITHOUT OPTIMIZATIONS

    tflite_model_content = converter.convert()

    print(f"Saving TFLite model to: {tflite_model_path}")
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model_content)
    print("TensorFlow Lite model saved successfully.")
    print("You can now use this 'model.tflite' in your Streamlit application with a TFLite interpreter.")

if __name__ == '__main__':
    convert_model() 
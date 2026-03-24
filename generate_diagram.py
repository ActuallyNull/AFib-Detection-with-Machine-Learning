import tensorflow as tf
from model import ecg_cnn  # Import the model from your existing script

# Create a model instance
model = ecg_cnn(input_length=3000)

# Generate the diagram
tf.keras.utils.plot_model(
    model,
    to_file='model_architecture.png',
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB',  # 'TB' for top-to-bottom, 'LR' for left-to-right
    expand_nested=True,
    dpi=96
)

print("Model architecture diagram saved to model_architecture.png")

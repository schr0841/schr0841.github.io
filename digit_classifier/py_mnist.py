# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# --- 1. Load MNIST Dataset ---
# The dataset is conveniently included in Keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"Training data shape: {x_train.shape}") # (60000, 28, 28)
print(f"Training labels shape: {y_train.shape}") # (60000,)
print(f"Test data shape: {x_test.shape}")     # (10000, 28, 28)
print(f"Test labels shape: {y_test.shape}")   # (10000,)

# --- 2. Preprocess the Data ---

# Normalize pixel values from 0-255 to 0-1
# Convert data type to float32 for division
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape images to add a channel dimension (for CNN input)
# MNIST images are grayscale, so channel dimension is 1
# Input shape expected by Conv2D is (batch_size, height, width, channels)
x_train = np.expand_dims(x_train, -1) # Adds dimension at the end -> (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)   # -> (10000, 28, 28, 1)

print(f"Reshaped training data shape: {x_train.shape}")
print(f"Reshaped test data shape: {x_test.shape}")

# Convert labels to one-hot encoded vectors
# e.g., label 5 becomes [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
num_classes = 10 # Digits 0-9
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(f"Example one-hot encoded training label (first sample): {y_train[0]}")

# --- 3. Define the CNN Model Architecture ---

# Input shape for the model
input_shape = (28, 28, 1) # height, width, channels

model = keras.Sequential(
    [
        # Input layer (implicitly defined by input_shape in the first layer)
        keras.Input(shape=input_shape),

        # Convolutional Layer 1: Learn features from the image
        # 32 filters (feature maps), 3x3 kernel size, ReLU activation
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),

        # Max Pooling Layer 1: Downsample the feature maps, reduce computation
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Convolutional Layer 2: Learn more complex features
        # 64 filters, 3x3 kernel size, ReLU activation
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),

        # Max Pooling Layer 2: Further downsampling
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten Layer: Convert 2D feature maps into a 1D vector
        # This prepares the data for the fully connected layers
        layers.Flatten(),

        # Dropout Layer: Regularization technique to prevent overfitting
        # Randomly sets a fraction (0.5 here) of input units to 0 during training
        layers.Dropout(0.5),

        # Dense Layer (Output Layer): Fully connected layer for classification
        # num_classes (10) units, one for each digit
        # Softmax activation outputs a probability distribution over the classes
        layers.Dense(num_classes, activation="softmax"),
    ]
)

# Print a summary of the model's layers and parameters
model.summary()

# --- 4. Compile the Model ---

# Configure the model for training
# Optimizer: Algorithm to update model weights (Adam is a common choice)
# Loss function: Measures how well the model is doing (categorical_crossentropy for multi-class classification with one-hot labels)
# Metrics: Evaluation metric(s) to monitor during training (accuracy is common for classification)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# --- 5. Train the Model ---

batch_size = 128  # Number of samples per gradient update
epochs = 15       # Number of times to iterate over the entire training dataset

print("\nStarting training...")
# Train the model using the training data
# validation_split reserves a portion of training data for validation during training
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
print("Training finished.")

# --- 6. Evaluate the Model (Optional) ---

# Evaluate the trained model on the test dataset
score = model.evaluate(x_test, y_test, verbose=0)
print("\nTest loss:", score[0])
print("Test accuracy:", score[1])

# --- 7. Save the Trained Model ---

# Save the entire model (architecture + weights + optimizer state) to a single HDF5 file
model_filename = "my_mnist_model.h5"
model.save(model_filename)
print(f"\nModel saved successfully as {model_filename}")

# Now you can use tensorflowjs_converter:
# tensorflowjs_converter --input_format keras my_mnist_model.h5 path/to/output/directory/


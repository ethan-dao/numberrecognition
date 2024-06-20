import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# Data is from MNIST training set; data is split into training and testing already
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize; turn values from 0-255 into values from 0 to 1
x_train, x_test = x_train / 255, x_test / 255

# Convert labels to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = tf.keras.models.Sequential([ # Use sequential model for FF neural network
    tf.keras.layers.Input(shape=(28,28)),
    tf.keras.layers.Flatten(), # 28 x 28 pixel image, flatten it into a 784 element vector
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'), 
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

sample = model(x_train[:1]).numpy() # Get predictions for first image in dataset
tf.nn.softmax(sample).numpy() # Get softmax prediction for sample, or convert logits to probabilities (normalized)

loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True) # Get loss function

model.compile( # Compile our model before training and testing it
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), # Experiment with this learning rate
    loss = loss_function,
    metrics = ['accuracy']
)

# Train the model for five epochs (evaluate the model on the test set)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
model.fit(x_train, y_train, epochs=5, validation_data = (x_test, y_test), callbacks=[early_stopping, tensorboard_callback])

def test_prediction(index):
    # Select a test image and its true label
    test_image = x_train[index]
    true_label = np.argmax(y_train[index])  # Convert one-hot encoded label to integer
    # Predict the label of the test image
    prediction = model.predict(test_image.reshape(1, 28, 28))  # Reshape to match input shape
    predicted_label = np.argmax(prediction)

    # Print out prediction and label
    print("Prediction: ", prediction)
    print("Label: ", predicted_label)

    # Plot the test image and the prediction
    plt.imshow(test_image, cmap=plt.cm.binary)
    plt.title(f"True label: {true_label}, Predicted label: {predicted_label}")
    plt.show()

    # Print the probabilities
    print("Predicted probabilities:", prediction)

# Evaluate the model on the testing set
# model.evaluate(x_test,  y_test, verbose=2)
# test_prediction(5)

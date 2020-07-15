import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def show_results(X_test, Y_test, Y_prediction, start_index=0, end_index=10):
    for i in range(start_index, end_index):
        plt.imshow(X_test[i])
        plt.title("Image Number " + str(i + 1))
        plt.xlabel("Actual : " + class_names[Y_test[i]])
        plt.title("Prediction : " + class_names[np.argmax(Y_prediction[i])])
        plt.show()


# Importing data
data = keras.datasets.fashion_mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Splitting data
(X_train, Y_train), (X_test, Y_test) = data.load_data()

# A bit of pre-processing
X_test = X_test / 255.0
X_train = X_train / 255.0

# Creating the Neural Network
neural_network = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

neural_network.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
neural_network.fit(X_train, Y_train, epochs=5)

# Evaluating the model
test_loss, test_accuracy = neural_network.evaluate(X_test, Y_test)
print("Accuracy : ", test_accuracy)

# Making prediction
Y_prediction = neural_network.predict(X_test)

# Showing results
show_results(X_test, Y_test, Y_prediction, 100, 120)

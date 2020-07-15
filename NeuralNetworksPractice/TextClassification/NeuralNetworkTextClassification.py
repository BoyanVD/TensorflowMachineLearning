import tensorflow as tf
from tensorflow import keras
import numpy as np


def decode(text, dictionary):
    return " ".join(dictionary.get(i, "?") for i in text)


def encode(text, dictionary):
    encoded = [1]

    for word in text:
        if word in dictionary:
            encoded.append(dictionary[word])
        else:
            encoded.append(2)

    return encoded


MAX_REVIEW_LENGTH = 250

# Loading data
data = keras.datasets.imdb

# Splitting data
(X_train, Y_train), (X_test, Y_test) = data.load_data(num_words=100000)

# Create the dictionary for word-index pairs
word_index = data.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# Creating the reversed dictionary
index_word = dict([(value, key) for (key, value) in word_index.items()])

# Trimming test and train data, excluding too long reviews
X_train = keras.preprocessing.sequence.pad_sequences(X_train,
                                                     value=word_index["<PAD>"],
                                                     padding="post",
                                                     maxlen=MAX_REVIEW_LENGTH)

X_test = keras.preprocessing.sequence.pad_sequences(X_test,
                                                    value=word_index["<PAD>"],
                                                    padding="post",
                                                    maxlen=MAX_REVIEW_LENGTH)

# Creating the model
neural_network = keras.Sequential()
neural_network.add(keras.layers.Embedding(100000, 16))
neural_network.add(keras.layers.GlobalAveragePooling1D())
neural_network.add(keras.layers.Dense(16, activation="relu"))
neural_network.add(keras.layers.Dense(1, activation="sigmoid"))

neural_network.summary()

neural_network.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Splitting train data, extracting  validation data
x_validation = X_train[:10000]
x_train = X_train[10000:]

y_validation = Y_train[:10000]
y_train = Y_train[10000:]

# Training the model
neural_network.fit(x_train, y_train,
                   epochs=40,
                   batch_size=512,
                   validation_data=(x_validation, y_validation),
                   verbose=1)

# Tracking results
results = neural_network.evaluate(X_test, Y_test)
print(results)

# Making new prediction
with open("review.txt", encoding="utf-8") as file:
    for line in file.readlines():
        next_line = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
        encoded = encode(next_line, word_index)
        encoded = keras.preprocessing.sequence.pad_sequences([encoded], value=word_index["<PAD>"],
                                                             padding="post",
                                                             maxlen=MAX_REVIEW_LENGTH)
        predict = neural_network.predict(encoded)
        print(line)
        print(encoded)
        print(predict[0])

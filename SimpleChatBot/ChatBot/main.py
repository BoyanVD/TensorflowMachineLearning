"""
Simple AI Chat Bot, using very small amount of training data form the
intents.json file.
"""

import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import json
import pickle
from keras.models import load_model

# The stemmer is used to extract the roots of words, as we want to look for the meaning of the different words
stemmer = LancasterStemmer()

# Loading data from JSON file
with open("intents.json") as file:
    data = json.load(file)

# Preprocessing the loaded data
def preprocess_data():
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    # Making all words lowercase and keeping only the unique words
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    # Ecoding sentences, using kind of one-hot encoding for sentences, marking how many occurances of each word we have.
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = np.array(training)
    output = np.array(output)

    return words, labels, training, output


def save_data(words, labels, training, output):
    with open("data.pickle", "wb") as file:
        pickle.dump((words, labels, training, output), file)


def build_model():
    neural_network = keras.Sequential()
    neural_network.add(tf.keras.layers.InputLayer(input_shape=(len(training[0]),)))
    neural_network.add(keras.layers.Dense(8, activation="relu"))
    neural_network.add(keras.layers.Dense(8, activation="sigmoid"))
    neural_network.add(keras.layers.Dense(len(output[0]), activation="softmax"))

    neural_network.summary()
    neural_network.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return neural_network


def convert_text(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return bag


def parse_neural_network_output(results, labels):
    results_index = np.argmax(results)
    tag = labels[results_index]

    if results[0][results_index] < 0.7:
        return "Sorry, I am not smart enough to respond."

    for data_tag in data["intents"]:
        if data_tag["tag"] == tag:
            responses = data_tag["responses"]
            break

    return random.choice(responses)


def chat(model, words, labels):
    print("You can start chatting with bot !")
    chat_nickname = input("Please enter your nickname : ")

    while True:
        user_input = input(chat_nickname + " : ")
        if user_input == "exit":
            break

        converted = convert_text(user_input, words)

        results = model.predict([converted])
        response = parse_neural_network_output(results, labels)

        print("Bobchoo : " + response)


def main():
    # Trying to load the data from file, otherwise we will preprocess it.
    try :
        with open("data.pickle", "rb") as file:
            words, labels, training, output = pickle.load(file)
    except :
        words, labels, training, output = preprocess_data()
        save_data(words, labels, training, output)


    # Trying to load model, building it otherwise
    try:
        neural_network = load_model("model.h5")
        neural_network.summary()
    except:
        neural_network = build_model()

        neural_network.fit(training, output, epochs=1000, batch_size=8)
        neural_network.save("model.h5")

    chat(neural_network, words, labels)


if __name__ == "__main__":
    main()

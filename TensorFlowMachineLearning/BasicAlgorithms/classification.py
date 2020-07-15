from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pandas as pd


"""
    Importing data :
"""
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

Y_train = train.pop('Species')
Y_test = test.pop('Species')

"""
    Creating the input function :
"""
def input_function(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

"""
    Creating feature columns :
"""
feature_columns = []
for key in train.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))

"""
    Building the model :
"""
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[30, 10], n_classes=3)

"""
    Training the model :
"""
classifier.train(
    input_fn=lambda : input_function(train, Y_train, training=True),
    steps=5000
)

eval_result = classifier.evaluate(input_fn=lambda: input_function(test, Y_test, training=False))
print("Test accuracy : ", eval_result)

"""
    Functions to handle user input for prediction :
"""
def user_input_function(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


def user_input_datapoint(features):
    predict = {}
    print("Please type new datapoint parameters as required : ")
    for feature in features:
        value = input(feature + " : ")
        predict[feature] = [float(value)]

    return predict


def predict(classifier_model, features):
    predictions = classifier_model.predict(input_fn=lambda: user_input_function(features))
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{}" ({:.1f}%)'.format(SPECIES[class_id], 100 * probability))

predict_val = user_input_datapoint(['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
predict(classifier, predict_val)

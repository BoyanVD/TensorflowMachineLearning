"""
Really simple exercise on building linear regression on Google's Titanic dataset, using TensorFlow.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fcib


"""
Basic 2D visual example of line of best fit, also called linear regression.
"""
X = [1, 2, 2.5, 3, 4]
Y = [1, 4, 7, 9, 15]

plt.plot(X, Y, 'ro')
plt.axis([0, 6, 0, 20])
plt.show()

plt.plot(X, Y, 'ro')
plt.axis([0, 6, 0, 20])
plt.plot(np.unique(X), np.poly1d(np.polyfit(X, Y, 1))(np.unique(X)))
plt.show()

"""
Loading the training and test datasets and separating them intoto Y and X sets :
"""
dataframe_train = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dataframe_eval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

print(dataframe_train.describe())

Y_train = dataframe_train.pop('survived')
Y_eval = dataframe_eval.pop('survived')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERICAL_COLUMNS = ['age', 'fare']

feature_columns = []

for feature in CATEGORICAL_COLUMNS:
    vocabulary = dataframe_train[feature].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature, vocabulary))

for feature in NUMERICAL_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature, dtype=tf.float32))

"""
Creating the input function. In our case it is the same as the one from the tensorflow's website :
"""
def make_input_function(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        dataset = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))

        if shuffle:
            dataset.shuffle(1000)

        dataset = dataset.batch(batch_size).repeat(num_epochs)
        return dataset

    return input_function

train_input_function = make_input_function(dataframe_train, Y_train)
eval_input_function = make_input_function(dataframe_eval, Y_eval, num_epochs=1, shuffle=False)

"""
Creating the model :
"""
linear_estimator = tf.estimator.LinearClassifier(feature_columns=feature_columns)

"""
Training and testing the model :
"""
linear_estimator.train(train_input_function)
result = linear_estimator.evaluate(eval_input_function)

print("Accuracy : ", result['accuracy'])

"""
Making predictions :
"""
predictions = list(linear_estimator.predict(eval_input_function))

ID = 0
print(dataframe_eval.loc[ID])
print("Survival probability for id ", ID," : ", predictions[ID]['probabilities'][0])
print("Actual result for id ", ID, " : ", Y_eval.loc[ID])

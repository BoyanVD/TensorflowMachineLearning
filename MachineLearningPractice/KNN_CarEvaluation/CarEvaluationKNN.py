import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

import pandas as pd
import numpy as np

# Reading the data
data = pd.read_csv("car.data")

# Encoding the non-numerical data
encoder = preprocessing.LabelEncoder()
buying_column = encoder.fit_transform(list(data["buying"]))
maint_column = encoder.fit_transform(list(data["maint"]))
door_column = encoder.fit_transform(list(data["door"]))
persons_column = encoder.fit_transform(list(data["persons"]))
lug_boot_column = encoder.fit_transform(list(data["lug_boot"]))
safety_column = encoder.fit_transform(list(data["safety"]))
class_column = encoder.fit_transform(list(data["class"]))

# Constructing X and Y
label_column = "class"
X = list(zip(buying_column, maint_column, door_column, persons_column, lug_boot_column, safety_column))
Y = list(class_column)

# Splitting the data
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# Creating the model
model = KNeighborsClassifier(n_neighbors=9)
model.fit(X_train, Y_train)

# Making prediction
Y_prediction = model.predict(X_test)

# Tracking accuracy
accuracy = model.score(X_test, Y_test)
print("Accuracy : ", accuracy)
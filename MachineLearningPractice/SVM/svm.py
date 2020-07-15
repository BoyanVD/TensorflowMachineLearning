import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


# Importing the data
data = datasets.load_breast_cancer()

# Creating features and targets
X = data.data
Y = data.target

# Splitting the data
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

# Classes we have
classes = ['malignant', 'benign']

# Creating and training the model
svm_classifier = svm.SVC(kernel="linear", C=2)
svm_classifier.fit(X_train, Y_train)

# Predicting test data
Y_prediction_svm = svm_classifier.predict(X_test)

# Tracking accuracy
svm_accuracy = metrics.accuracy_score(Y_test, Y_prediction_svm)
print("Accuracy SVM : ", svm_accuracy)


# KNN vs SVM :

# Creating, training, predicting and tracking accuracy with KNN
knn_classifier = KNeighborsClassifier(n_neighbors=9)
knn_classifier.fit(X_train, Y_train)

Y_prediction_knn = knn_classifier.predict(X_test)

knn_accuracy = metrics.accuracy_score(Y_test, Y_prediction_knn)
print("Accuracy KNN : ", knn_accuracy)
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# fetch dataset 
dataset = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = dataset.data.features 
y = dataset.data.targets
y = y.values

# X = pd.get_dummies(X, dtype=float)

y[y == "Iris-setosa"] = 1
y[y == "Iris-virginica"] = 0
y[y == "Iris-versicolor"] = 0
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

class Perceptron:
    def __init__(self, learningRate = 0.4, n_iter = 100):
        self.learning_rate = learningRate
        self.n_iter = n_iter

    def activation_function(self, value):
        return np.where(value >= 0, 1, 0)

    def update_weights(self, features, y_real, y_pred):
        error = y_real - y_pred
        weight_change = self.learning_rate * error
        self.weights = self.weights + weight_change * features
        self.bias = self.bias + weight_change

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            for i, feature in enumerate(X):
                value = np.dot(feature, self.weights) + self.bias 
                predicted = self.activation_function(value)
                self.update_weights(feature, y[i], predicted)

    def predict(self, X):
        value = np.dot(X, self.weights) + self.bias
        predicted = self.activation_function(value)
        return predicted

model = Perceptron()
model.train(X_train.values, y_train)

y_pred = model.predict(X_test.values)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

report = classification_report(y_test, y_pred)
print("Classification Report: \n", report)

# plt.scatter(X['sepal length'], X['sepal width'])
# plt.xlabel("Sepal Length")
# plt.ylabel("Sepal Width")
# plt.show()

# Accuracy: 1.00
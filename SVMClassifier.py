import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
pd.options.mode.chained_assignment = None

# fetch dataset 
dataset = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = dataset.data.features 
y = dataset.data.targets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

classifier = make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.2))
classifier.fit(X_train, y_train.values.ravel())
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

report = classification_report(y_test, y_pred)
print("Classification Report: \n", report)


# kernel = linear, C = 0.8, gamma = auto: Accuracy is 0.9777
# kernel = linear, C = 0.2, gamma = scale: Accuracy is 0.100

# kernel = poly, degree = 3, C = 0.2, gamma = scale: Accuracy is 0.8666
# kernel = poly, degree = 3, C = 0.6, gamma = scale: Accuracy is 0.9555
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# fetch dataset 
dataset = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = dataset.data.features 
y = dataset.data.targets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42) 

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(X_train, y_train.values.ravel())
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

report = classification_report(y_test, y_pred)
print("Classification Report: \n", report)
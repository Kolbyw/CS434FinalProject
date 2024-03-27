from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# fetch dataset 
dataset = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = dataset.data.features 
y = dataset.data.targets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

report = classification_report(y_test, y_pred)
print("Classification Report: \n", report)
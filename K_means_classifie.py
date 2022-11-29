import numpy as np
exec(open("readData.py").read())

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
def k_NN_classifier(X_train, Y_train, X_test, Y_test):
    neighbors = 1
    classifier = KNeighborsClassifier(n_neighbors = neighbors)
    X_train_norm = np.reshape(X_train, (50000,3072))
    X_test_norm = np.reshape(X_test, (10000,3072))
    classifier.fit(X_train_norm, Y_train) 
    predictions = classifier.predict(X_test_norm)
    acc = accuracy_score(Y_test, predictions)
    return acc *100

def main():
    accuracy = k_NN_classifier(X_train, Y_train, X_test, Y_test)
    print(f"the accuracy of this classifier is {accuracy}")
    return

if __name__=="__main__":
    main()
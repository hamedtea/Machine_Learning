
import numpy as np
exec(open("readData.py").read())

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
def Gaussian_naive_bayes(X_train, Y_train, X_test, Y_test ):
    gnb_model = GaussianNB()
    gnb_pred = gnb_model.fit(X_train.reshape(50000,3072), Y_train).predict(X_test.reshape(10000,3072))
    acc = accuracy_score(Y_test, gnb_pred)
    return acc


def main():
    accuracy = Gaussian_naive_bayes(X_train, Y_train, X_test, Y_test)
    print(f"the accuracy of Gaussian Naive Bayes classifier is {accuracy}")
    return

if __name__== "__main__":
    main()
import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

#datadict = unpickle('/home/kamarain/Data/cifar-10-batches-py/data_batch_1')
#datadict = unpickle('/home/kamarain/Data/cifar-10-batches-py/test_batch')

datadict = unpickle('C:/Users/hamed/anaconda3/envs/dataml100/cifar-10-python/cifar-10-batches-py/data_batch_1')
#datadict = unpickle('C:/Users/hamed/anaconda3/envs/dataml100/cifar-10-python/cifar-10-batches-py/test_batch')


X = datadict["data"]

Y = datadict["labels"]

#print(X.shape)

#labeldict = unpickle('/home/kamarain/Data/cifar-10-batches-py/batches.meta')
labeldict = unpickle('C:/Users/hamed/anaconda3/envs/dataml100/cifar-10-python/cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]
#print(labeldict)
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y = np.array(Y)
print(len(X))
print(len(Y))

for i in range(X.shape[0]):
    # Show some images randomly
    if random() > 0.999:
        plt.figure(1);
        plt.clf()
        plt.imshow(X[i])
        plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
        plt.pause(1)

def class_acc(pred,gt):
    return

def cifar10_classiﬁer_random(x):
    return

def cifar10_classiﬁer_1nn(x,trdata,trlabels):
    return
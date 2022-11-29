import pickle
import numpy as np
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

datadict_1 = unpickle('./cifar-10-batches-py/data_batch_1')
datadict_2 = unpickle('./cifar-10-batches-py/data_batch_2')
datadict_3 = unpickle('./cifar-10-batches-py/data_batch_3')
datadict_4 = unpickle('./cifar-10-batches-py/data_batch_4')
datadict_5 = unpickle('./cifar-10-batches-py/data_batch_5')
X1 = datadict_1["data"]
X2 = datadict_2["data"]  
X3 = datadict_3["data"]  
X4 = datadict_4["data"]  
X5 = datadict_5["data"]  
Y1 = datadict_1["labels"] #10000 labels
Y2 = datadict_2["labels"]
Y3 = datadict_3["labels"]
Y4 = datadict_4["labels"]
Y5 = datadict_5["labels"]
Y_train = np.concatenate((Y1, Y2, Y3, Y4, Y5), axis=0)
Y_train = np.array(Y_train)
X_train = np.concatenate((X1, X2, X3, X4, X5), axis=0)
X_train = X_train.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("float32")/255.0
t_datadic = unpickle('./cifar-10-batches-py/test_batch')
X_test = t_datadic["data"] #nd-data
X_test = X_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float32")/255.0
Y_test = t_datadic["labels"] #10000 labels
Y_test = np.array(Y_test)

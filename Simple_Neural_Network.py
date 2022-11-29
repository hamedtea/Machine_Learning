
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import layers
from keras import utils

exec(open("readData.py").read()) #read cifar_10 data by readfile script

def class_acc(pred,gt):
  subtract = pred - gt
  zero_elements = np.count_nonzero(subtract == 0)
  accuracy = zero_elements / len(subtract)
  return accuracy *100

def scheduler(epoch, lr):
  """a function that changes the learning rate adaptively"""
  if epoch < 18:
    return lr
  else:
    return lr * tf.math.exp(-0.01)

def neural_netwrok_simple(X_train, Y_train, X_test, Y_test):
  """simple neural netwokr wit one layer of 5 neurons with sigmoid activation and decent gradiant optimizer"""
  X_train_n = X_train.reshape((50000,3072))
  X_test_n = np.reshape(X_test, (10000,3072))
  #one hot encoder
  encoded_Y_train = keras.utils.to_categorical(Y_train, dtype ="float32")
  #adaptive learning rate operation
  callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
  #simple model generation
  simple_model = Sequential()
  simple_model.add(layers.Dense(5, input_dim=3072, activation='sigmoid'))
  simple_model.add(layers.Dense(10, activation='sigmoid'))
  #model indicators
  opt = keras.optimizers.SGD(learning_rate=0.01)
  #complie mode
  simple_model.compile(optimizer=opt, loss='mse', metrics=['mse'])
  #fit model
  simple_model.fit(X_train_n, encoded_Y_train, epochs=5, callbacks=[callback], verbose=0, shuffle = True)
  #model predictions
  pri=simple_model.predict(X_test_n)
  simple_NN_predictions = np.argmax(pri, axis = 1)
  #computing accuracy
  acc = class_acc(simple_NN_predictions,Y_test)
  return acc

def main():
  accuracy = neural_netwrok_simple(X_train, Y_train, X_test, Y_test)
  print(f"the accuracy of this classifier is {accuracy}")
  return

if __name__=="__main__":
    main()
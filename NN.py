
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import layers
from keras import utils

exec(open("readData.py").read())

def neural_netwrok(X_train, Y_train, X_test, Y_test):
  #encoded_Y_train = keras.utils.to_categorical(Y_train, dtype ="float32")
  model = Sequential( [
        tf.keras.layers.Input(shape= (32,32,3)),                                          
        tf.keras.layers.Conv2D(32, 5, padding = "valid", activation = "relu"),             
        tf.keras.layers.MaxPooling2D(pool_size = (2,2), padding = "valid"),                                  
        tf.keras.layers.Conv2D(64, 5, padding = "valid", activation = "relu"),                                                            
        tf.keras.layers.MaxPooling2D(pool_size = (2,2), padding = "valid"),                                  
        tf.keras.layers.Conv2D(128, 5, padding = "valid", activation = "relu"),                             
        tf.keras.layers.Flatten(),   
        tf.keras.layers.Dropout(0.1,noise_shape=None,seed=None),                                                     
        tf.keras.layers.Dense(128, activation = "relu"), 
        tf.keras.layers.Dense(64, activation = "relu"),
        tf.keras.layers.Dense(10,),                                                        
    
    ]
  )
  #compile model
  model.compile(
    #loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),     
    optimizer = keras.optimizers.Adam(learning_rate=0.01),
    #metrics =["accuracy"],
    loss='categorical_crossentropy', 
    metrics=['sparse_categorical_accuracy'],                                            
    )
  #fit the moodel with training data and encoded one hot labels 
  training_history = model.fit(X_train, Y_train, batch_size = 62, epochs = 5, 
                              verbose = 1, validation_data = (X_test, Y_test), 
                               shuffle = True)
  #make predictions
  pre=model.predict(X_test)
  NN_predictions = np.argmax(pre, axis = 1)
  #evaluation
  test_loss, test_accuracy = model.evaluate(X_test, Y_test)
  #plot model
  tf.keras.utils.plot_model(
  model,
  to_file="model.png",
  show_shapes=False,
  show_dtype=False,
  show_layer_names=True,
  rankdir="TB",
  expand_nested=False,
  dpi=96,
  layer_range=None,
  show_layer_activations=False,)
  return model.summary, training_history, NN_predictions, test_loss, test_accuracy

def NN_model_summary(Y_test):
  model_summary, training_history, NN_predictions, test_loss, test_accuracy = neural_netwrok(X_train, Y_train, X_test, Y_test)
  acc = class_acc(NN_predictions, Y_test)
  print("the accuracy of better NN model is", acc)
  print('model loss equals to', test_loss)
  print('model accuracy equals to', test_accuracy )
  print(model_summary())
  #plot loss function
  plt.plot(training_history.history['loss'])
  plt.plot(training_history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epochs')
  plt.legend(['train_loss', 'value of loss'], loc = 'upper right')
  plt.show
  #plot accuracy function
  plt.figure()
  plt.plot(training_history.history['accuracy'])
  plt.plot(training_history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epochs')
  plt.legend(['train acuracy', 'value of accuracy'], loc = 'upper right')
  plt.show
  return

def main():
  NN_model_summary(Y_test)
  return

if __name__=="__main__":
    main()
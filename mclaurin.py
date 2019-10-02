import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, concatenate
from tensorflow.keras import Model, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
#_______________________________________________________________________________
#Caricamento dati
validation_x=np.load("validation_x.npy", allow_pickle=True)
validation_y=np.load("validation_y.npy", allow_pickle=True)
train_x=np.load("train_x.npy", allow_pickle=True)
train_y=np.load("train_y.npy", allow_pickle=True)
test_x=np.load("test_x.npy", allow_pickle=True)
test_y=np.load("test_y.npy", allow_pickle=True)
#_______________________________________________________________________________
#STRUTTURA DEL MODELLO
def McLaurin():
  input_vector = Input(shape=(200))
  fc = Dense(500, activation="softsign")(input_vector)
  fc = Dense(1250, activation="softsign")(fc)
  fc = Dense(500, activation="softsign")(fc)
  fc = Dense(200, activation="softsign")(fc)
  fc = Dense(75, activation="softsign")(fc)
  fc = Dense(28, activation="softsign")(fc)
  fc = Dense(10, activation="softsign")(fc)
  output = Dense(4, activation='linear')(fc)
  model=Model([input_vector], output)
  model.compile(loss='MSE', optimizer="RMSprop")
  return model

#_______________________________________________________________________________
#Chiamo la funzione per definire il modello
model = McLaurin()
#_______________________________________________________________________________
#Preparo i dati per l'addestramento della rete
EPOCHS=100
BATCH_SIZE=128
steps_per_epoch = train_x.shape[0]//BATCH_SIZE
validation_steps = validation_y.shape[0]//BATCH_SIZE
#_______________________________________________________________________________
#ADDESTRAMENTO
h = model.fit(train_x,train_y, steps_per_epoch=steps_per_epoch,
                        epochs=EPOCHS, validation_data=(validation_x,validation_y),
                        validation_steps=validation_steps, shuffle=True)
#_______________________________________________________________________________

print(h.history.keys())

#GRAFICO PER L'ANDAMENTO DELLA LOSS
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('PERDITA DEL MODELLO')
plt.ylabel('perdita')
plt.xlabel('epoca')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#_________________________________________________________________
#TEST
print(model.evaluate(test_x,test_y))
#_________________________________________________________________
model.save("McLaurin.h5")


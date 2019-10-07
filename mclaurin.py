import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, Input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
#_______________________________________________________________________________
#Caricamento dati
train_x=np.load("train_x.npy", allow_pickle=True)
train_y=np.load("train_y.npy", allow_pickle=True)
test_x=np.load("test_x.npy", allow_pickle=True)
test_y=np.load("test_y.npy", allow_pickle=True)
#_______________________________________________________________________________
#STRUTTURA DEL MODELLO
def McLaurin():
  input_vector = Input(shape=(200))
  fc = Dense(500, activation="softsign")(input_vector)
  fc = Dense(1500, activation="softsign")(fc)
  fc = Dense(500, activation="softsign")(fc)
  fc = Dense(200, activation="softsign")(fc)
  fc = Dense(80, activation="softsign")(fc)
  fc = Dense(32, activation="softsign")(fc)
  output = Dense(4, activation='linear')(fc)
  model=Model([input_vector], output)
  model.compile(loss='MSE', optimizer="RMSprop")
  return model

#_______________________________________________________________________________
#Chiamo la funzione per definire il modello
model = McLaurin()
#_______________________________________________________________________________
#DIVIDO I DATI IN TRAIN E VALIDATION
train_x, validation_x, train_y, validation_y = train_test_split(train_x,train_y,test_size=0.2,random_state=0)
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

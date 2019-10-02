import numpy as np
import math
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
#___________________________________________
#Creazione dati di input
test = np.zeros(200)
x = -1
for i in range(0,200):
  test[i]= math.log(x+2)#Funzione da modificare durante i test
  x=round(x+0.01,2)
#__________________________________________
#Preparazione dati di input

testX = test.reshape(1,200)
#_____________________________________________
#Caricamento modello e predizione
model = tf.keras.models.load_model("McLaurin.h5")
print("________________________")
print("Predizione del modello")
pred = model.predict(testX)
print(pred)
#_________________________________________________
#Confronto grafico
print("____________________________________________")
print("CONFRONTO GRAFICO")
t = np.arange(-1.0, 1.0, 0.01)
y_true = np.log(t+2)#Funzione da modificare durante i test
y_pred = pred[0][0]+ (pred[0][1]*t) + (pred[0][2]*pow(t,2)) + (pred[0][3]*pow(t,3))
fig, ax = plt.subplots()
ax.plot(t, y_true)
ax.plot(t, y_pred)
ax.grid()
plt.legend(['y_true', 'y_pred'], loc='upper left')
plt.show()

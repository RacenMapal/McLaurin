import tensorflow as tf
import numpy as np
#CARICAMENTO DATI DI TEST
test_x=np.load("test_x.npy", allow_pickle=True)
test_y=np.load("test_y.npy", allow_pickle=True)
#CARICAMENTO MODELLO E TEST
model=tf.keras.models.load_model("McLaurin.h5")
score=model.evaluate(test_x, test_y)
print(score)



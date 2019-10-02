import numpy as np
#DATASET
'''
input alla rete neurale McLaurin: 
a0+(a1*x)+(a2*x^2)+(a3*x^3) per x = -1,-0.9,...,1 --->200 campioni di questa funzione

output desiderato:
(a0,a1,a2,a3)
'''
'''
Per la compressione delle reti neurali inizializzare con distribuzione uniforme
random.rand --->distribuzione uniforme
random.randn--->distribuzioine normale
'''
def create_dataset(number):
  normal_labels=np.array(np.random.randn(number/2,4), dtype=np.float32)
  uniform_labels = np.array(np.random.rand(number/2,4), dtype=np.float32)
  labels = np.concatenate((normal_labels, uniform_labels))
  data_x=np.zeros(shape=(number,200),dtype=np.float32)
  step=0
  for coeff in labels:
      n=0
      x=-1
      campioni = np.zeros(200)
      while True:
        campioni[n] = coeff[0] + coeff[1]*x + coeff[2]*pow(x,2) + coeff[3]*pow(x,3)
        n=n+1
        if(n==200):
          break
        x=round(x+0.01,2)  
      print(step)
      data_x[step]=campioni
      print(step)
      step=step+1
  return labels, data_x
#DATI DI TRAINING
print("TRAIN DATA")
train_y, train_x= create_dataset(40000)
print(train_x)
print("# of training  ", train_x.shape[0]) 
np.save("train_x.npy", train_x)   
np.save("train_y.npy", train_y)
#DATI DI TEST
print("__________________________________________________")
print("TEST DATA")
test_y, test_x= create_dataset(4000)
print(test_x)
print("# of test  ", test_x.shape[0]) 
np.save("test_x.npy", test_x)   
np.save("test_y.npy", test_y)

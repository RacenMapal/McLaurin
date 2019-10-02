import numpy as np
#DATASET
'''
input alla rete neurale McLaurin: 
a0+(a1*x)+(a2*x^2)+(a3*x^3) per x = -1,-0.9,...,1 ---->200 campioni di questa funzione

output desiderato:
(a0,a1,a2,a3)
'''
'''
random.rand --->distribuzione uniforme
random.randn--->distribuzione normale
'''
labels=np.array(np.random.randn(1000,4), dtype=np.float32)
train_x=np.zeros(shape=(1000,200),dtype=np.float32)
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
    print(campioni)
    train_x[step]=campioni
    print(step)
    step=step+1

print(train_x)
print("# of training  ", train_x.shape[0]) #2000
np.save("train_x.npy", train_x)   
np.save("train_y.npy", labels)

import numpy as np
import pandas as pd
import B22CS092_train
import generation
y_testi= generation.y_test
W_new= B22CS092_train.W


print(f"imported W: {W_new}") #imported the trained weights. 
def testing( W_new):
    with open('B22CS092_test.txt', 'r') as file:
        nsam= int(file.readline().strip())
        print(nsam)
        Xt= []
        
        for _ in range(nsam):
            line = file.readline().strip()
            parts= line.split()
            Xs = [(float(x)) for x in parts[:len(parts)]]
            Xt.append(Xs)
    yt= np.array(y_testi)        
    Xt= np.array(Xt)
    print(Xt)
    print(yt)
    min= Xt.min(axis=0)
    max= Xt.max(axis=0)
    Xt= np.round((Xt-min)/(max-min),2) #normalization
    ones= np.ones((Xt.shape[0], 1))
    Xt= np.concatenate((ones, Xt), axis=1)
    errors=0
    for x, y_actual in zip(Xt, yt):
        J= np.dot(x, W_new)
        if J>=0:
            y_pred=1
        else:
            y_pred=0
        if y_pred!= y_actual:
            errors+=1 #checking errors
    print(f"Errors {errors}")
    print (f"Accuracy is: {(len(yt)-errors)/len(yt)*100:.3f}%.")
    
testing(W_new)
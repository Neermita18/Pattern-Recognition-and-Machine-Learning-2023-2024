import pandas as pd
import numpy as np
# np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
W= [] #to store new weights
def training():
    with open('B22CS092_train.txt', 'r') as file:
        nsam= int(file.readline().strip())
        print(nsam)
        X= []
        y=[]
        for _ in range(nsam):
            line = file.readline().strip()
            parts= line.split()
            Xs = [(float(x)) for x in parts[:-1]]
            X.append(Xs)
            ys= int(parts[-1])
            y.append(ys)
    X= np.array(X)
    min= X.min(axis=0)
    max= X.max(axis=0)
    X= np.round((X-min)/(max-min),2) #normalization
    y= np.array(y)
    np.random.seed(1000) #this seed will stay constant.
    ones= np.ones((X.shape[0], 1))
    X= np.concatenate((ones, X), axis=1)
    print(X)
    print(X[0])
    num_samples, num_features = X.shape
    print(num_features)
    a= 0.01 #learning rate. Can be 1 also for normal perceptron learning algo.
    errors=-1 #to enter while loop. 
    epoch=0
    weights = np.round(np.random.uniform(-2,1,5),2)
    print(f"Old weights: {weights}")
    while(errors!=0):
        errors=0
    
        for x, y_actual in zip(X, y):
            J=np.dot(x, weights)
            if J>=0: #J is the function 
                y_pred=1
            else:
                y_pred=0
            
            if y_actual!= y_pred:
                weights= weights + a*(x)*(y_actual- y_pred) #update rule. Explanation in report.
                errors+=1
        epoch+=1
    return weights, epoch
W,i=training()
print(f"New weights after training are:  {W}\n")
print(f"Converged after {i} epochs")
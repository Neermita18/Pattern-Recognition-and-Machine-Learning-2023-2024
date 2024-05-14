import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
samples=1000
#note that this model splits the entire data into two parts only, test and train. Report results with training using 20% of synthetic data: 20% training, 80% testing. The other method is implemented in google colab. Please refer to README.txt
def synthetic(samples):

  np.random.seed(42) #this can be either 1000 or 42. Results reported in report.
  
  W = np.round(np.random.uniform(-2,1,5),2)
  
  w0= W[0]
  w1= W[1]
  w2= W[2]
  w3= W[3]
  w4= W[4]
  
  with open('B22CS092_data.txt', 'w') as file:
        file.write(f"{samples}\n")
        for _ in range(samples):
            x1, x2, x3, x4 = np.random.randint(0, 10, 4)
            f= w0 + w1*x1 + w2*x2 + w3*x3 + w4*x4
            if f>=0:
              label=1
            else:
              label=0
            
            file.write(f"{x1} {x2} {x3} {x4} {label}\n") #data.txt created
            

synthetic(samples)

text = open('B22CS092_data.txt', 'r')
line_list = text.readlines()
for line in line_list:
    print(line)

text.close()
y_test=[]
def read(filename, test_size):
    with open(filename, 'r') as file:
        nsam= int(file.readline().strip())
        print(nsam)
        X= []
        y=[]
        for _ in range(nsam):
            line = file.readline().strip()
            parts= line.split()
            Xs= [int(x) for x in parts[:-1]]
            X.append(Xs)
            ys= int(parts[-1])
            y.append(ys)
    X= np.array(X)
    # print(X)
    y= np.array(y)
    data = list(zip(X,y))
    np.random.seed(42)
    np.random.shuffle(data)
    split= int(len(data)* (1- test_size))
    train_data, test_data = data[:split], data[split:] #splitting into training and testing sets. The entire dataset is split into two. 
    
    with open('B22CS092_train.txt', 'w') as file:
        file.write(f"{len(train_data)}\n")
        
        for X, y in train_data:
            
            line = ' '.join(f"{x}" for x in X) + f" {y}\n"
            file.write(line)
            
    with open('B22CS092_test.txt', 'w') as file:
        
        file.write(f"{len(test_data)}\n")
        for X, y in test_data:
            y_test.append(y)
            line = ' '.join(f"{x}" for x in X) + f"\n"
            file.write(line)

        
read('B22CS092_data.txt', 0.3) #x=test size. Training size = 1-x. Complete dataset split it into training and testing. Kind of like Cross Validation

def check():
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
    y= np.array(y)
    scaler = StandardScaler() 
    X_scaled = scaler.fit_transform(X)
    clf = SVC(kernel='linear') #To check if perceptron applicable or not
    clf.fit(X_scaled, y)
    y_pred = clf.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    print(f"Training accuracy: {accuracy * 100:.2f}%") #If training accuracy is very high then data is linearly separable.
check()
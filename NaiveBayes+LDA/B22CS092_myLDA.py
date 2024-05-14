import csv
import numpy as np
import pandas as pd

def ComputeMeanDiff(X,df):
    # class_0 = X[X[:, 2] == 0][:, :2]
    # class_1 = X[X[:, 2] == 1][:, :2]
    # m1 = np.mean(class_0, axis=0)
    # m2 = np.mean(class_1, axis=0)
    # mean_diff = m1 - m2
    # return mean_diff
    mean_0_feature1 = df.loc[df['class'] == 0.0, 'f1'].mean()
    mean_0_feature2 = df.loc[df['class'] == 0.0, 'f2'].mean()
    mean_1_feature1 = df.loc[df['class'] == 1.0, 'f1'].mean()
    mean_1_feature2 = df.loc[df['class'] == 1.0, 'f2'].mean()
    m_1 = [mean_0_feature1, mean_0_feature2]
    m_2 = [mean_1_feature1, mean_1_feature2]
    # print(m_1)
    mean_diff = np.array(m_1) - np.array(m_2)
    return mean_diff


def ComputeSW(X,df):
    S1= np.zeros((2,2))
    S2= np.zeros((2,2))
    S_W = np.zeros((2,2))
    m_1 = df[df['class'] == 0.0][['f1', 'f2']].mean().values.reshape(2,1)
    m_2 = df[df['class'] == 1.0][['f1', 'f2']].mean().values.reshape(2,1)
    m1= np.array(m_1)
    m2= np.array(m_2)
    for _, row in df[df['class'] == 0.0].iterrows():
        row_vec = row[['f1', 'f2']].values.reshape(2,1) #row_vec is a 2D vector like m_1. (f1,f2) sample given that it is in class 1
        S1 += (row_vec - m1).dot((row_vec - m1).T)
    for _, row in df[df['class'] == 1.0].iterrows():
        row_vec = row[['f1', 'f2']].values.reshape(2,1) #(f1, f2) sample given that it is in class 0.0
        S2 += (row_vec - m2).dot((row_vec - m2).T)
    S_W= S1+ S2
    return S_W

def ComputeSB(X,df):
    S_B = np.zeros((2,2))
    
    m_1 = df[df['class'] == 0.0][['f1', 'f2']].mean().values.reshape(2,1)
    m_2 = df[df['class'] == 1.0][['f1', 'f2']].mean().values.reshape(2,1)
    diff= m_1-m_2
    S_B= np.dot(diff, diff.T)
    return S_B
def GetLDAProjectionVector(X,df):
    S_B= ComputeSB(X,df)
    S_W= ComputeSW(X,df)
    eig_vals, eig_vecs= np.linalg.eig(np.linalg.inv(S_W).dot(S_B)) #returns normalized vectors
    eig_pairs= [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

#Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs= sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    print('Eigenvalues in decreasing order: ')
    for i in eig_pairs:
        print(i[0])
    for i in range(len(eig_vals)):
        eigvec_sc= eig_vecs[:,i].reshape(2,1)   
        print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
        print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))
        index = eig_vals.argmax() #selecting largest eigenvalue will give maximum separation and minimum within class scatter
        w=eig_vecs[:, index] #represents the direction in the feature space that maximizes the class separation when you project your data onto it
    return w
def project(x,y,w):
  p= np.array([x,y])
  projectionmag= np.dot(p,w)
  projdir= np.dot(p,w)/np.dot(w, w) * w
  return projectionmag, projdir

#########################################################
###################Helper Code###########################
#########################################################

X = np.empty((0, 3)) #not used
with open('data.csv', mode ='r')as file:
  csvFile = csv.reader(file)
  for sample in csvFile:
        X = np.vstack((X, [float(item) for item in sample]))

print(X)
print(X.shape)
df= pd.read_csv('data.csv', header=None, names=['f1', 'f2', 'class']) #using this
# X Contains m samples each of formate (x,y) and class label 0.0 or 1.0
print(df)
# m_1 = df[df['class'] == 0.0][['f1', 'f2']].mean().values.reshape(2,1)
# print(m_1)
opt=int(input("Input your option (1-5): "))

match opt:
  case 1:
    meanDiff=ComputeMeanDiff(X,df)
    print("Mean difference using m_1= mean of class 0.0 and m_2 as mean of class 1.0:  ",meanDiff)
    print("Mean difference using m_1= mean of class 1.0 and m_2 as mean of class 0.0:  ",(-meanDiff))
  case 2:
    SW=ComputeSW(X,df)
    print(SW)
  case 3:
    SB=ComputeSB(X,df)
    print(SB)
  case 4:
    w=GetLDAProjectionVector(X,df)
    print(w)
  case 5:
    x=int(input("Input x dimension of a 2-dimensional point :"))
    y=int(input("Input y dimension of a 2-dimensional point:"))
    w=GetLDAProjectionVector(X,df)
    projmag, projdir= project(x,y,w)
    print("Projection is: \n",projmag)
    print("Projection direction is: ", projdir)
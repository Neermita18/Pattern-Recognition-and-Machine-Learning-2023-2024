from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.svm import SVC
iris = datasets.load_iris( as_frame=True )
#TASK 1 (a)
# print(iris) 
#linear classifier. Good for complexity, can throw away most training samples, only need support vectors. SVM doesnt allow noise. To deal with that we use Slack variable. 
data= iris.data
targets =pd.DataFrame({'species':iris.target})

print(data)
df= pd.DataFrame(data)
print(df)
print(df.columns)
da= df.drop(columns=['sepal length (cm)', 'sepal width (cm)'])
print(da)
da= pd.concat([da, targets], axis=1)
print(da)
da= da[(da['species'] == 0) | (da['species'] == 1)]

print(da)

yo = da['species'] 
Xo = da.drop(columns=['species']) 
 # Target variable

scaler = StandardScaler()
X_normalized = scaler.fit_transform(Xo)


X_train, X_test, y_train, y_test = train_test_split(X_normalized, yo, test_size=0.2, random_state=42)

# Print the shapes of train and test sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#plotting the data nicely
plt.figure(figsize=(8, 6))
plt.scatter(X_normalized[yo == 0, 0], X_normalized[yo == 0, 1], label='Setosa', color='purple')  #petal length vs petal width for Setosa
plt.scatter(X_normalized[yo == 1, 0], X_normalized[yo == 1, 1], label='Versicolor', color='lavender')  #petal length vs petal width for Versicolor

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Scatter Plot of Petal Length vs Petal Width')

plt.legend()

plt.show()

# TASK 1 (b)


clf = LinearSVC()
clf.fit(X_train, y_train)

#Function to plot decision boundary
def db(clf, X, y, title):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 #added +=1 because points not seen otherwise
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) #the decision boundary. xx and yy are flattened arrays
    
    
    Z = Z.reshape(xx.shape)
    
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    
    plt.xlabel('Petal Length (normalized)')
    plt.ylabel('Petal Width (normalized)')
    
    plt.title(title)
    plt.show()


db(clf, X_train, y_train, title="scatterplot of train data")


plt.figure()
db(clf, X_test, y_test, title='scatterplot of test data')

# TASK 2 (a)

X, y = make_moons(n_samples=500, noise=0.05, random_state=42)

#add noise
noise= int(0.05 * len(y))
np.random.seed(42)
noise_indices = np.random.choice(len(y), noise, replace=False)
y[noise_indices] = 1 - y[noise_indices] 

#Verify the shape of the dataset
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

#Task 2 (b)
kernels = ['linear', 'poly', 'rbf']
kernel_names = ['Linear', 'Polynomial', 'RBF']

plt.figure(figsize=(15, 5))

for i, kernel in enumerate(kernels):
    # Train SVM model with specified kernel
    clf = SVC(kernel=kernel, random_state=42)
    clf.fit(X, y)
    plt.subplot(1, 3, i + 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=70, edgecolors='k')
    
    #Plot support vectors
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=200, facecolors='none', edgecolors='k')
    
    # Create meshgrid to plot decision boundary
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100),
                         np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 100))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    
    plt.title(f'{kernel_names[i]} Kernel')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
# FOR POLY with degree param
plt.tight_layout()
plt.show()
clpoly = SVC(kernel='poly', random_state=42, degree= 5) #change the degree
clpoly.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=70, edgecolors='k')
    
    #Plot support vectors
plt.scatter(clpoly.support_vectors_[:, 0], clpoly.support_vectors_[:, 1],
                s=200, facecolors='none', edgecolors='k')
    
    # Create meshgrid to plot decision boundary
xxp, yyp = np.meshgrid(np.linspace(X[:, 0].min() -1, X[:, 0].max() + 1, 100),
                         np.linspace(X[:, 1].min() -1, X[:, 1].max() + 1, 100))
Zp = clpoly.predict(np.c_[xxp.ravel(), yyp.ravel()])
Zp = Zp.reshape(xxp.shape)
plt.contourf(xxp, yyp, Zp, cmap=plt.cm.coolwarm, alpha=0.8)
    
plt.title(f'Poly Kernel')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()






# TASK 2 (c) and (d)


from sklearn.model_selection import GridSearchCV
svm_rbf = SVC(kernel='rbf')
param_grid = {'C': [0.001, 0.01, 0.1, 1, 2],
              'gamma': [1, 2, 3, 4, 5, 7, 8, 9, 10, 20, 30, 50]}

grid_search = GridSearchCV(svm_rbf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)



best_C = 10000000
best_gamma = 3
svm_rbf_best = SVC(kernel='rbf', gamma=best_gamma, C=best_C, random_state=42)

# Train the SVM model with the best hyperparameters
svm_rbf_best.fit(X, y)
accuracy = svm_rbf_best.score(X, y)

# Print the accuracy
print("Model Accuracy:", accuracy)
# Plot decision boundary for the RBF kernel SVM with the best hyperparameters
plt.figure(figsize=(8, 6))

# Plot training data
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k', label='Training Data')

# # Plot testing data
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, s=20, marker='s', edgecolors='k', label='Testing Data')

# Create meshgrid to plot decision boundary
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100),
                     np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 100))
Z = svm_rbf_best.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=70, edgecolors='k')
# Plot support vectors
plt.scatter(svm_rbf_best.support_vectors_[:, 0], svm_rbf_best.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

plt.title('Decision Boundary for RBF Kernel SVM (Arbitrary Hyperparameters)')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.legend()

plt.tight_layout()
plt.show()
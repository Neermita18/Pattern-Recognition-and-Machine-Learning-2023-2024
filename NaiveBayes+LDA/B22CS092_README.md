1) LDA
Function provided to acquire X (using np.vstack) was not used. I directly created a dataframe and passed that in all the funcions to compute 
Mean difference, Sb, Sw and eigenvalues and eigenvectors. 
Hence, please note the changes. 
Note that for case 1, I take m_1 as mean of samples in class 0.0 and m_2 as mean of samples in class 1.0.
Note that for case 5, I print the projection magnitude as well as the projection direction. project(x,y,w) returns the magnitude as well as the vector. 
To calculate 1-nn accuracies, random states 42 and 1000 were used. Their results have been tabulated in the report. 

2) Naive Bayes is first done using random state = 1.
Two testcases are acquired and the accuracy is 50%. But they are not good testcases since most pf the feature values are the same. 

Hence using random state=42, I get 2 new testcases.
The accuracy is 100% for this.
Overfitting on training data possible.

Using laplace smoothing on the new testcases (rs=42).
I used lambda=1, 100, 1000 and tabulated the results. All of them show an accuracy of 50%. The posterior probabilities (P(yes|features) and P(no|features)) get closer and closer. 
More generalized.
Dependencies in the features are visible from the correlation matrix. Hence, Naive Bayes wouldn't work perfectly on this dataset. 
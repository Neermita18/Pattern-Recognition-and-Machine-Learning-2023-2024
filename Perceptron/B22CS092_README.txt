I have included a file called generation.py which generates the data.txt as desired.
I can set the number of samples (100 or 1000 taken for the report.) 
Note that B22CS092_data.txt is created by generation.py
B22CS092_train.txt and B22CS092_text.txt are created by generation.py
Method 1:
The training and testing dataset sizes depend on the variable test_size. This is what I thought was meant by "For training, use 20% of synthetic data."
The entire data.txt is split into 20% training data and 80% testing data.
Method 2:
I take 1000 samples and use 20%, 50% or 70% of the entire data.txt as work_data. Over work_data, I apply a 80:20 split and 20:80 split and report results.
Method 3:
Sir made a clarification later hence, please note that on Google Colab, I have taken 1100 samples, keep the first 100 as test data and the from the remaining 1000 samples, use either 20%, 50% or 70% of it. 
This is method 3 in my report.
The results of all the cases are there in the report.

Hence, I have implemented three ways of splitting the data into training and testing sets. 
1. By splitting entire dataset into train and test using percentages given in question.
2. By first splitting data.txt according to the percentages in the question. This becomes work_data. Then using 80:20 and 20:80 split.
3. By using separating out a constant test set of 100 samples and the remaining 1000 samples undergo a further split of 20%, 50% or 70%.
Training is done in B22CS092_train.py.
Testing is done in B22CS092_test.py.
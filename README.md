[![Work in Repl.it](https://classroom.github.com/assets/work-in-replit-14baed9a392b3a25080506f3b7b6d57f295ec2978f6f33ec97e36a161684cbe9.svg)](https://classroom.github.com/online_ide?assignment_repo_id=4875664&assignment_repo_type=AssignmentRepo)
# Background
- Cardiovascular disease (CVD) is the world’s leading cause of death and a major public health problem. CVD prediction is one of the most effective measures to control CVD. Based on this, the project initially determined to select 70,000 copies including age and gender and blood pressure, etc., classify the data according to whether or not they are sick. First, explore the data set, find the features that are highly related to CVD, and then use logistic models to predict, including logistic regression model, SVM and KNN, etc., and find the best model by comparing their predictive performance, optimize and evaluate the best model, and hope to use the best model to provide a reference for the detection of CVD.
# Install
- Before starting this project, we need to install some packages.
```
import pandas as pd   
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns   
import sys
!{sys.executable} -m pip install pandas-profiling
import pandas_profiling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
```
# Step one(Data Pre-processing and Initial Exploratory)
- Before using the predictive model, we need to perform data preprocessing
```
data = pd.read_csv('/Users/miaogangrui/Desktop/cardio_train.csv',sep=';') # Import data set
data.head()  # View data set and variables
data = data.drop(['id'],axis=1)  # Remove variables that are not related to prediction
data.describe()  # Statistical analysis of data

# delete ap_hi outliers
Q1 = data['ap_hi'].quantile(0.25)
Q3 = data['ap_hi'].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data['ap_hi'] < (Q1 - 1.5 * IQR)) |(data['ap_hi'] > (Q3 + 1.5 * IQR)))]
data = data[data['ap_lo'] < 150]   # delete ap_lo outliers

pandas_profiling.ProfileReport(data)  # Exploratory analysis of data

# Correlation analysis
correlations = data.corr()['cardio'].drop('cardio')
correlations
data = data.drop(['gender','height','gluc','smoke','alco','active'],axis=1)  # Remove variables that have low correlation with cardio

# Pearson correlation plot
data_var = data.drop(['cardio'],axis=1)  
plt.figure(figsize=(8, 8))
sns.heatmap(data_var.corr(), vmin=-1, cmap='coolwarm', linewidths=0.1, annot=True)
plt.title('Pearson correlation coefficient between variables', fontdict={'fontsize': 15})
plt.show()
```
**Output**

<img src="https://github.com/ACM40960/project-Aoms95/blob/main/output1.jpeg" width="400px">

<img src="https://github.com/ACM40960/project-Aoms95/blob/main/output2.jpeg" width="400px">

- Next,we take some preliminary explorations of explanatory variable and the target variable
```
# age and cardio
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
sns.boxplot(x='cardio',y='age',data=data)
# weight and cardio
plt.subplot(1,2,2)
sns.boxplot(x='cardio',y='weight',data=data)

# ap_hi and cardio
plt.figure(figsize=(14,5))
plt.subplot(1,3,1)
sns.boxplot(x='cardio',y='ap_hi',data=data)
# ap_lo and cardio
plt.subplot(1,3,2)
sns.boxplot(x='cardio',y='ap_lo',data=data)
# cholesterol and cardio
plt.subplot(1,3,3)
sns.boxplot(x='cardio',y='cholesterol',data=data)
```
**Output**

<img src="https://github.com/ACM40960/project-Aoms95/blob/main/output3.jpeg" width="600px">

<img src="https://github.com/ACM40960/project-Aoms95/blob/main/output4.jpeg" width="600px">

# Step two(Predictive Analysis)
### Logistic Regression Model
```
# Import packages of logistic regression 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Split dataset in explanatory variables and target variable
feature_cols = ['age', 'weight', 'ap_hi', 'ap_lo','cholesterol']
X = data[feature_cols]   # Explanatory variables
X_std = (X-X.mean())/X.std()  # Data standardization
y = data.cardio   # Target variable

# Split X and y into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# Fit the model
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

# Use the classification matrix to view the classification result
matrix_class = metrics.confusion_matrix(y_test, y_pred)
matrix_class

# Compute the accuracy of the model
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
```
**Output**
- array([[5672, 2912],
       [3652, 4697]])
- Accuracy: 0.6123545739089352

### Support Vector Machines(kernel='linear',C=1,gamma=0.01)
```
# Import package of svm model
from sklearn import svm
from sklearn.svm import SVC

# Create a svm Classifier
SVM_linear_1 = svm.SVC(kernel='linear',C=1,gamma=0.01) 
# Train the model using the training sets
SVM_linear_1.fit(X_train, y_train)
# Predict the response for test dataset
y_pred_linear_1 = SVM_linear_1.predict(X_test)

# Compute the accuracy of the model
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_linear_1))
```
**Output**
- Accuracy: 0.7250339573613653

### Support Vector Machines(kernel='linear',C=10,gamma=0.01)
```
# Create a svm Classifier
SVM_linear_2 = svm.SVC(kernel='linear',C=10,gamma=0.01) 
# Train the model using the training sets
SVM_linear_2.fit(X_train, y_train)
# Predict the response for test dataset
y_pred_linear_2 = SVM_linear_2.predict(X_test)

# Compute the accuracy of the model
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_linear_2))
```
**Output**
- Accuracy: 0.7227898186972185

### Support Vector Machines(kernel='linear',C=10,gamma=0.01)
```
# Create a svm Classifier
SVM_linear_2 = svm.SVC(kernel='linear',C=10,gamma=0.01) 
# Train the model using the training sets
SVM_linear_2.fit(X_train, y_train)
# Predict the response for test dataset
y_pred_linear_2 = SVM_linear_2.predict(X_test)

# Compute the accuracy of the model
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_linear_2))
```
**Output**
- Accuracy: 0.7227898186972185

### Support Vector Machines(kernel='rbf', C=1,gamma=0.01)
```
# Create a svm Classifier
SVM_rbf_1 = svm.SVC(kernel='rbf', C=1,gamma=0.01) 
# Train the model using the training sets
SVM_rbf_1.fit(X_train, y_train)
# Predict the response for test dataset
y_pred_rbf_1 = SVM_rbf_1.predict(X_test)

# Compute the accuracy of the model
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_rbf_1))
```
**Output**
- Accuracy: 0.6927301718537766

### Support Vector Machines(kernel='rbf', C=10,gamma=0.01)
```
# Create a svm Classifier
SVM_rbf_2 = svm.SVC(kernel='rbf', C=10,gamma=0.01) 
# Train the model using the training sets
SVM_rbf_2.fit(X_train, y_train)
# Predict the response for test dataset
y_pred_rbf_2 = SVM_rbf_2.predict(X_test)

# Compute the accuracy of the model
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_rbf_2))
```
**Output**
- Accuracy: 0.6552884899309042


### K-Nearest Neighbors(KNN)
```
# Import package of knn
from sklearn.neighbors import KNeighborsClassifier

# Use the k-fold cross-validation to select the optimal k value
from sklearn.model_selection import GridSearchCV
clf = KNeighborsClassifier()
para = range(1,50)  # the range of k
param_dict = {"n_neighbors": para}  
clf = GridSearchCV(clf, param_grid=param_dict, cv=10)  # 10-fold cross-validation
clf.fit(X_train, y_train)

# best parameter
print("Best prarmeter：\n",clf.best_params_)
```
**Output**
- Best prarmeter：
 {'n_neighbors': 28}
 
```
# Fit the model with the optimal k value
knn = KNeighborsClassifier(n_neighbors=28)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

# Use the classification matrix to view the classification result
matrix_class = metrics.confusion_matrix(y_test, y_pred)
print(matrix_class)

# Compute the accuracy of the model
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
```
**Output**
- [[6812 1772]
 [3152 5197]]
Accuracy: 0.709206874151066

# Maintainers
- Gangrui Miao

# project2
Breast Cancer Prediction
# project2
#Breast Cancer Prediction
# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# importing the dataset
df = pd.read_csv('data.csv')
df.head()
# dropping unnecessary columns
df.drop(['Unnamed: 32','id'],axis=1,inplace=True)

# checking for the missing values
df.isnull().sum()
# checking the data types of the columns
df.dtypes

# checking the data description
df.describe()

# coorelation between the columns diagnosis and the other columns
df.corr()['diagnosis'].sort_values()
# bar plot for the number of diagnosis
plt.figure(figsize=(5,5))
sns.barplot(x=df['diagnosis'].value_counts().index,y=df['diagnosis'].value_counts().values)
# create a heatmap to check the correlation
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.drop(['diagnosis'],axis=1),df['diagnosis'],test_size=0.3,random_state=42)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
#predicting the diagnosis
y_pred = dtree.predict(X_test)

# printing samples from predicted and actual values
print('Predicted values: ',y_pred[:10])
print('Actual values: ',y_test[:10])
# model evaluation 
print(dtree.score(X_test,y_test))
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
yhat = logmodel.predict(X_test)
# printing samples from predicted and actual values
print('Predicted values: ',yhat[:10])
print('Actual values: ',y_test[:10])
# model evaluation
print(logmodel.score(X_test,y_test))

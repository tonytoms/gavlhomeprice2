#import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../"))

#read in data 
#training set consists house id 1- 1460
data= pd.read_csv('D:\data_domains2.csv',index_col=0)
#testing set consists house id 1461 - 2919
testdata=pd.read_csv('D:\data_domains2.csv', index_col=0)
#insert saleprice column in testdata just so the they have the same # of columns
testdata["12-price"]=0

#concate the dataframes together so I can clean the data together. They will be seperated later
total=pd.concat([data,testdata])
#Fill NAN with 0's
total=total.apply(lambda x: x.fillna(0))
# conver object to dummy variables
columns=total.columns[data.dtypes == 'object']
total=pd.get_dummies(total, columns =columns)
print(total.head())
#now split the data again
data = total.iloc[0:1460,:]
testdata= total.iloc[1460:, :]
# Now remove SalePrice column from test set
testdata.drop(["12-price"],axis=1)
print(data.head())
print(testdata.head())

#split out validation data set
Y=data[['12-price']]
data.drop(["12-price"], axis=1)
X=data
print(testdata)
print(X)
print(Y)
#Start Testing a couple  models' performance
#Try Lasso, ElasticNet, Ridge, SVR(kernel ='rbf') etc
#import packages
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor 
from sklearn.pipeline import Pipeline
#feature extraction with SelectKBest
from sklearn.feature_selection import SelectKBest 
#Statistical tests can be used to select those features that have the strongest relationship with the output variable. 
#The scikit-learn library provides the SelectKBest class2 that can be used with a suite of different statistical tests to select a specifit number of features.
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA

#Split out validation dataset
X_train,X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size= 0.2,random_state=7)
num_folds=10
scoring = 'mean_squared_error'
seed=7
#Finalize the model

model = Pipeline([('pca',PCA(n_components=100)),('Scaler',StandardScaler()),('Lasso', Lasso(alpha=9))])
model.fit(X_train,Y_train)
print(X_validation.head())
print(testdata.head())
prediction=model.predict(testdata)
print(prediction.shape)

testdata["12-price"]=prediction
output=testdata[["12-price"]]
print(output.head(20))
output.to_csv("prediction_output_ML_test2.csv")
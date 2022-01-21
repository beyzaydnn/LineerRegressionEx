# -*- coding: utf-8 -*-

#Importing the libraries
import pandas as pd
import numpy as np

#Importing the dataset
veri = pd.read_csv('odev_tenis.csv')


#using label-encoding and one-hot-encoder to converting categorical values into numerical values 
from sklearn import preprocessing
outlook =veri.iloc[:,0:1].values
ohe =preprocessing.OneHotEncoder()
outlook=ohe.fit_transform(outlook).toarray()


le =preprocessing.LabelEncoder()
wp=veri.iloc[:,3:5].apply(le.fit_transform)
humidity=veri.iloc[:,2:3].values

print(len(veri))

outlook =pd.DataFrame(data=outlook, index=range(14), columns=['overcast', 'rainy','sunny'])
s=pd.concat([outlook,veri.iloc[:,1:2],wp],axis=1)


#Splitting our dataset to Training and Testing dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(s,humidity,test_size=0.33, random_state=0)

#Fitting Linear Regression to the training set
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

#predicting the Test set result
y_pred =lr.predict(x_test)

#backward elimination
import statsmodels.api as sm
X= s.iloc[:,:].values
model = sm.OLS(humidity,X).fit()

print(model.summary())

#eliminating colomn windy(x5)
x_train = x_train.iloc[:,[0,1,2,3,5]]
x_test = x_test.iloc[:,[0,1,2,3,5]]

lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)

import statsmodels.api as sm
X= s.iloc[:,[0,1,2,3,5]].values
model = sm.OLS(humidity,X).fit()
print(model.summary())


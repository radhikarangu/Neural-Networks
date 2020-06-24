# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:14:00 2020

@author: RADHIKA
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
startups=pd.read_csv("D:\\ExcelR Data\\Assignments\\Neural Networks\\50_Startups.csv")
##preprocessing data
startups.columns
startups['State'].value_counts()
startups.head()
startups.isnull().sum
startups.shape
#from sklearn.preprocessing import LabelEncoder
#label = LabelEncoder()
#startups['State'] = label.fit_transform(startups['State'])
#startups['State'].value_counts()
startups.corr()
startups=startups.drop(['State'],axis=1)
startups.mean()
cont_model = Sequential()

cont_model.add(Dense(50,input_dim=3,activation="relu"))
cont_model.add(Dense(40,activation="relu"))
cont_model.add(Dense(20,activation="relu"))
cont_model.add(Dense(1,kernel_initializer="normal"))
cont_model.compile(loss="mean_squared_error",optimizer = "adam",metrics = ["mse"]) 


column_names = list(startups.columns)

predictors = column_names[0:3]
target = column_names[3]

first_model=cont_model
first_model.fit(np.array(startups[predictors]),np.array(startups[target]),epochs=10)
pred_train = first_model.predict(np.array(startups[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-startups[target])**2))
#86.67%

import matplotlib.pyplot as plt
plt.plot(pred_train,startups[target],"bo")
np.corrcoef(pred_train,startups[target])# we got high correlation 

#############Concrete dataset Assignment###########################


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
concrete=pd.read_csv("D:\\ExcelR Data\\Assignments\\Neural Networks\\concrete.csv")
###preprocessing data
concrete.columns

concrete.shape
#(1030, 9)
concrete.isnull().sum#checking NA values
concrete.head
concrete['strength'].value_counts

cont_model = Sequential()

cont_model.add(Dense(50,input_dim=8,activation="relu"))
cont_model.add(Dense(40,activation="relu"))
cont_model.add(Dense(20,activation="relu"))
cont_model.add(Dense(1,kernel_initializer="normal"))
cont_model.compile(loss="mean_squared_error",optimizer = "adam",metrics = ["mse"])

column_names = list(concrete.columns)
predictors = column_names[0:8]
target = column_names[8] 

first_model = cont_model
first_model.fit(np.array(concrete[predictors]),np.array(concrete[target]),epochs=10)
pred_train = first_model.predict(np.array(concrete[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-concrete[target])**2))

####8.150914707277737

import matplotlib.pyplot as plt
plt.plot(pred_train,concrete[target],"bo")
np.corrcoef(pred_train,concrete[target])# we got high correlation 
#array([[1.        , 0.87568969],
 #      [0.87568969, 1.        ]])



###############ForestFires Assignment#############


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score

forest=pd.read_csv("D:\\ExcelR Data\\Assignments\\Neural Networks\\forestfires.csv")
##preprocessing the data
forest.columns
forest.head
forest.shape
#(517, 31)
#cheking NA values
forest.isnull().sum
#summary
forest=forest.drop(['dayfri','daymon','daysat','daysun','daythu','daytue','daywed','monthapr','monthaug','monthdec', 'monthfeb','monthjan', 'monthjul', 'monthjun', 'monthmar', 'monthmay', 'monthnov','monthoct', 'monthsep'],axis=1)
forest['month'],month=pd.factorize(forest['month'])
forest['day'],day=pd.factorize(forest['day'])
forest.head()
forest.describe()
forest.columns
forest.corr()

from sklearn.preprocessing import scale 
forestnorm=pd.DataFrame(scale(forest.iloc[:,0:-1]))
forestnorm.describe()
datao=pd.DataFrame(forest.iloc[:,-1])
x=pd.concat([forestnorm,datao],axis=1)
x['size_category'],size_category=pd.factorize(x['size_category'])


cont_model = Sequential()
cont_model.add(Dense(50,input_dim=11,activation="relu"))
cont_model.add(Dense(40,activation="relu"))
cont_model.add(Dense(20,activation="relu"))
cont_model.add(Dense(1,kernel_initializer="normal"))
cont_model.compile(loss="mean_squared_error",optimizer = "adam",metrics = ["mse"])

column_names = list(x.columns)
predictors = column_names[0:11]
target = column_names[11]

first_model = cont_model
first_model.fit((x[predictors]),(x[target]),epochs=10)
pred_train = first_model.predict((x[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
bool(pred_train)
pred_train=pd.DataFrame(pred_train)
pred_train['pred']=pd.DataFrame(pred_train)
pred_train['pred']=np.where(pred_train['pred']< 0.5,'0','1')
pred_train['pred'],pred=pd.factorize(pred_train['pred'])
from sklearn.metrics import accuracy_score as acc
accuracy=acc(x['size_category'],pred_train['pred']) 

#0.7833655705996132 accuracy


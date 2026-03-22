import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression
from 
from sklearn.model_selection import cross_val_score,train_test_split

# Data lode
df = pd.read_excel(r"C:\Users\LENOVO\Downloads\bankdata.xlsx")
df = df[[
    "Area Insured",
    "Sum Insured",
    "Indemnity Level",
    "Gross Premium",
    ]]
print("Data Loaded Successfully ")
print(df.shape)
print(df.head())  # Hade call 

# dev
x= df.iloc[ :,:-1]
print(x)

y= df.iloc[:,-1]
print(y.head())

# converted cros velidation score 

x= x.astype(float)
y= y.astype(float)

linr_reg= LinearRegression()

mse= cross_val_score(linr_reg , x,y, scoring= 'neg_mean_squared_error',cv = 5)
print(mse)

# fine mean squred error 

main_mse= np.mean(mse)
print("MSE",main_mse)


# devide of data 
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size= 0.2,random_state= 42)
linr_reg.fit(x_train , y_train)


# predecation 
predcation = linr_reg.predict(x_test)
print(predcation)




# create dat farme 




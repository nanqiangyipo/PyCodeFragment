# Required Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from pandas import DataFrame,Series
data= DataFrame(data=[[150.0,6450.0], [200.0,7450.0], [250.0,8450.0], [300.0, 9450.0], [350.0, 11450.0], [400.0, 15450.0], [600.0, 18450.0]],columns=['square_feet','price'])
x=list(data['square_feet'].astype('float'))
y=list(data['price'].astype('float'))
x=[[i] for i in x]

lr=linear_model.LinearRegression()
lr.fit(x,y)

predictResult=lr.predict(600)

para={}
para['intercept']=lr.intercept_
para['coefficient']=lr.coef_
para['predicted_value']=predictResult


# 手工计算w 和 b的值
xMin=np.mean(list(data['square_feet'].astype('float')))
x_xMin=data['square_feet'].astype('float')-xMin
x_xMinXy=x_xMin*data['price'].astype('float')
sum1=x_xMinXy.sum()
sum2=(data['square_feet'].astype('float')*data['square_feet'].astype('float')).sum()
sum3=data['square_feet'].astype('float').sum()
sum3=(sum3**2)/data.shape[0]
w=sum1/(sum2-sum3)

wx=data['square_feet'].astype('float')*w
b=(data['price'].astype('float')-wx).sum()/data.shape[0]
print(w,b,sep='-')

# 可视化
x=list(data['square_feet'].astype('float'))
y=list(data['price'].astype('float'))
plt.scatter(x,y,color='blue')
plt.plot(x,lr.predict([[i] for i in x]),color='red',linewidth=3)
plt.show()
print(para)
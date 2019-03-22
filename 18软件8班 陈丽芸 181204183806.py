
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn


# In[5]:


file = 'E:\大一下\机器学习\qiche.csv'
from pandas import read_csv
data = read_csv(file,delimiter='\t',engine='python')
print(data)


# In[6]:


X=np.c_[data["x"]]
y=np.c_[data["y"]]


# In[7]:


data.plot(kind="scatter",x="x",y="y")
plt.show()


# In[8]:


from sklearn import linear_model
lr_model = linear_model.LinearRegression()
lr_model.fit(X,y)
print("斜率:%s,截距:%s"%(lr_model.coef_[0][0],lr_model.intercept_[0]))
print("估计模型为：y=%sx+%sy"%(lr_model.coef_[0][0],lr_model.intercept_[0]))


# In[9]:


data.plot(kind="scatter",x="x",y="y")
plt.plot(X,lr_model.predict(X.reshape(-1,1)),color='red',linewidth=4)
plt.show()


# In[10]:


X_new = [[100]]
print(lr_model.predict(X_new))


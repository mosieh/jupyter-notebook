#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# In[2]:


import numpy as np
import scipy
from math import sqrt


# In[3]:


rawData = pd.read_csv('default of credit card clients.csv',
header=1)
rawData.head()


# In[4]:


rawData.info()


# In[6]:


depVar=rawData['default payment next month']


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(rawData, depVar, test_size=0.3)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# In[12]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[13]:


modelKNN=KNeighborsClassifier()
modelTREE=DecisionTreeClassifier()
modelRF=RandomForestClassifier()


# In[14]:


print(depVar)


# In[17]:


modelKNN.fit(X_train,y_train)


# In[18]:


modelTREE.fit(X_train,y_train)


# In[19]:


modelRF.fit(X_train,y_train)


# In[20]:


from sklearn.model_selection import cross_val_score


# In[21]:


print(cross_val_score(modelKNN, X_train, y_train))


# In[22]:


modelKNN.score(X_train,y_train)


# In[23]:


print(cross_val_score(modelTREE,X_train,y_train))
modelTREE.score(X_train,y_train)


# In[24]:


print(cross_val_score(modelRF,X_train,y_train))
modelRF.score(X_train,y_train)


# In[26]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[27]:


from math import sqrt


# In[28]:


predictions = modelKNN.predict(X_test)


# In[29]:


rmse=sqrt(mean_squared_error(y_test,predictions))


# In[30]:


predRsquared = r2_score(y_test,predictions)


# In[31]:


print('R Squared:%.3f'%predRsquared)
print('RMSE:%.3f'% rmse)


# In[44]:


modelKNN.score(X_train,y_train)
modelRF.score(X_train,y_train)
modelTREE.score(X_train,y_train)


# In[45]:


predictionsTREE=modelTREE.predict(X_test)


# In[46]:


predRsquared=r2_score(y_test,predictions)
rmse=sqrt(mean_squared_error(y_test,predictions))
print('R Squared: %.3f'%predRsquared)
print('RMSE: %.3f'% rmse)


# In[47]:


from sklearn.feature_selection import RFE


# In[48]:


rfe = RFE(modelKNN, 3)


# In[52]:


features=rawData.iloc[:,2:25]


# In[53]:


print('Summary of feature sample')


# In[54]:


features.head()


# In[55]:


print(depVar)


# In[57]:


X_train, X_test, y_train, y_test = train_test_split(rawData, depVar, test_size=0.3)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(features, depVar, test_size=0.3)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# In[59]:


modelKNN.fit(X_train,y_train)


# In[60]:


print(cross_val_score(modelKNN,X_train,y_train))
modelKNN.score(X_train,y_train)


# In[61]:


modelTREE.fit(X_train,y_train)
print(cross_val_score(modelTREE,X_train,y_train))
modelTREE.score(X_train,y_train)


# In[62]:


modelRF.fit(X_train,y_train)
print(cross_val_score(modelRF,X_train,y_train))
modelRF.score(X_train,y_train)


# In[63]:


predictions=modelKNN.predict(X_test)


# In[64]:


predRsquared=r2_score(y_test,predictions)
rmse=sqrt(mean_squared_error(y_test,predictions))
print('R Squared: %.3f'% predRsquared)
print('RMSE:%.3f'%rmse)


# In[67]:


plt.scatter(y_test, predictions, marker='o')


# In[68]:


plt.plot(y_test, predictions, '-p', color='gray',
         markersize=15, linewidth=4,
         markerfacecolor='white',
         markeredgecolor='gray',
         markeredgewidth=2)
plt.ylim(-1.2, 1.2);


# In[69]:


plt.plot(y_test, predictions, '-p', color='red',
         markersize=15, linewidth=4,
         markerfacecolor='blue',
         markeredgecolor='green',
         markeredgewidth=2)
plt.ylim(-1.2, 1.2);


# In[75]:


pd.scatter_matrix(features, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# In[76]:


predictions


# In[80]:


plt.scatter(y_test, predictions)
plt.xlabel('GroundTruth')
plt.ylabel('Predictions')


# In[82]:


from sklearn.metrics import accuracy_score


# In[85]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


# In[86]:


accuracy_score(y_test,predictions)


# In[87]:


precision_score(y_test,predictions)


# In[88]:


recall_score(y_test,predictions)


# In[89]:


from sklearn.svm import SVC


# In[90]:


model = SVC(kernel="rbf", C=1.0, gamma=1e-4)
model.fit(X_train,y_train)
y_predicted=model.predict(X_test)


# In[91]:


y_predicted


# In[92]:


print(y_predicted)


# In[94]:


model.score


# In[95]:


accuracy_score(y_test,y_predicted)


# In[103]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(y_test,y_predicted)
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.show();


# In[ ]:





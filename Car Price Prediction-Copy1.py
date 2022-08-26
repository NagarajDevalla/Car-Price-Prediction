#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


# In[5]:


##DATA COLLECTION AND PROCESSING


# In[6]:


#loading the data from csv file to pandas dataframe


# In[12]:


car_dataset = pd.read_csv('car data.csv')


# In[13]:


#inspecting the first 5 rows ofthe dataframe
car_dataset.head()


# In[14]:


# checking the num of rows and columns
car_dataset.shape


# In[15]:


#getting some information about the dataset
car_dataset.info()


# In[16]:


#checking number of missing values
car_dataset.isnull().sum()


# In[17]:


#checking the distribution of categorical data
print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())


# In[34]:


#encoding"Fuel_Type" column
car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
#encoding"Seller_Type" column
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
#encoding"Transmission" column
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)


# In[38]:


car_dataset.head()


# In[39]:


#splitting the data into target
X = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
Y = car_dataset['Selling_Price']


# In[40]:


print(X)


# In[41]:


print(Y)


# In[42]:


#splitting training and test data
X_train, X_test , Y_train, Y_test = train_test_split(X, Y, test_size=0.1,random_state=2)


# In[43]:


lin_reg_model = LinearRegression()


# In[44]:


lin_reg_model.fit(X_train,Y_train)


# prediction on training data

# In[45]:


training_data_prediction = lin_reg_model.predict(X_train)


# In[46]:


# R squared error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)


# visualize the actual prices and predicted prices

# In[47]:


plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.ylabel("Actual Prices vs Predicted Prices")
plt.show


# prediction on training data

# In[48]:


test_data_prediction = lin_reg_model.predict(X_test)


# In[49]:


# R squared Error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error : ", error_score)


# In[51]:


plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.ylabel("Actual Prices vs Predicted Prices")
plt.show


# Lasso regression

# In[52]:


lass_reg_model = Lasso()


# In[53]:


lass_reg_model.fit(X_train,Y_train)


# In[54]:


training_data_prediction = lass_reg_model.predict(X_train)


# In[55]:


# R squared error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)


# In[56]:


plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.ylabel("Actual Prices vs Predicted Prices")
plt.show


# In[57]:


test_data_prediction = lass_reg_model.predict(X_test)


# In[59]:


# R squared Error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error : ", error_score)


# In[60]:


plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.ylabel("Actual Prices vs Predicted Prices")
plt.show


# In[ ]:





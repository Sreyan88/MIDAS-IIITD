#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
import datetime as dt
warnings.filterwarnings('ignore')


# In[2]:

print("Reading dataset..")
data = pd.read_csv('TrainAndValid.csv')
test = pd.read_csv('Test.csv')


# In[3]:

print("Preprocessing...")
target = data['SalePrice']
target = np.log(target)


# In[4]:


data.drop('SalePrice', axis=1, inplace=True)


# In[5]:


# data.head()


# In[6]:


# data.shape


# In[7]:


# test.shape


# In[10]:


percent_missing = (data.isnull().sum()/len(data)).sort_values(ascending=False) * 100


# In[11]:


percent_missing_test = (test.isnull().sum()/len(test)).sort_values(ascending=False) * 100


# In[12]:


percent_missing= pd.DataFrame(percent_missing[percent_missing>0])


# In[13]:


percent_missing_test= pd.DataFrame(percent_missing_test[percent_missing_test>0])


# In[14]:


missing_cols = pd.DataFrame(percent_missing.index)


# In[15]:


missing_cols_test = pd.DataFrame(percent_missing_test.index)


# In[8]:


# percent_missing.index


# In[9]:


# percent_missing_test.index


# In[17]:


# set(percent_missing.index).difference(percent_missing_test.index)


# In[18]:


cat_feat = data.select_dtypes(include=['object'])


# In[19]:


cat_feat_test = test.select_dtypes(include=['object'])


# In[20]:


num_feat = data.select_dtypes(exclude=['object'])


# In[21]:


num_feat_test = test.select_dtypes(exclude=['object'])


# In[23]:


# cat_feat.shape, cat_feat_test.shape


# In[25]:


# num_feat.shape, num_feat_test.shape


# In[26]:


cat_percent_missing = (cat_feat.isnull().sum()/len(cat_feat)).sort_values(ascending=False) * 100


# In[27]:


cat_percent_missing= pd.DataFrame(cat_percent_missing[cat_percent_missing>0])


# In[28]:


cat_percent_missing_test = (cat_feat_test.isnull().sum()/len(cat_feat_test)).sort_values(ascending=False) * 100
cat_percent_missing_test = pd.DataFrame(cat_percent_missing_test[cat_percent_missing_test>0])


# In[29]:


uni_val = []
for col in cat_feat.columns:
    uni_val.append(cat_feat[col].nunique())


# In[30]:


uni_val = pd.DataFrame(uni_val, index=cat_feat.columns, columns={'value'})


# In[31]:


uni_val_test = []
for col in cat_feat_test.columns:
    uni_val_test.append(cat_feat_test[col].nunique())
uni_val_test = pd.DataFrame(uni_val_test, index=cat_feat_test.columns, columns={'value'})


# In[32]:


#since filModelDesc = fiBaseModel + fiSecondaryDesc + fiModelSeries + fiModelDescriptor
cat_feat.drop(['fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor'], inplace=True, axis=1)


# In[33]:


#since filModelDesc = fiBaseModel + fiSecondaryDesc + fiModelSeries + fiModelDescriptor
cat_feat_test.drop(['fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor'], inplace=True, axis=1)


# In[35]:


# cat_feat.head()


# In[36]:


# cat_feat_test.head()


# In[37]:


#more_unival.drop(['fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor'], inplace=True, axis=0)


# In[38]:


# cat_feat.columns


# In[39]:


# cat_feat_test.columns


# In[41]:


# cat_feat.ProductSize.value_counts()


# In[42]:


# cat_feat_test.ProductSize.value_counts()


# In[43]:


# cat_feat.UsageBand.value_counts()


# In[44]:


# cat_feat_test.UsageBand.value_counts()


# In[45]:


# cat_feat.UsageBand.isnull().sum()/len(data)*100


# In[46]:


# cat_feat_test.UsageBand.isnull().sum()/len(cat_feat_test)*100


# In[47]:


# cat_feat.ProductSize.isnull().sum()/len(data)*100


# In[48]:


# cat_feat_test.ProductSize.isnull().sum()/len(cat_feat_test)*100


# In[49]:


cat_feat.UsageBand.fillna('-999', inplace=True)
cat_feat.ProductSize.fillna('-999', inplace=True)


# In[50]:


cat_feat_test.UsageBand.fillna('-999', inplace=True)
cat_feat_test.ProductSize.fillna('-999', inplace=True)


# In[51]:


usageBand = []
for i in range(len(cat_feat)):
    if cat_feat.UsageBand[i]=='-999' and cat_feat.ProductSize[i] != '-999':
        if cat_feat.ProductSize[i] == 'Mini':
            usageBand.append('Low')
        elif cat_feat.ProductSize[i] == 'Medium' or cat_feat.ProductSize[i] == 'Small' or cat_feat.ProductSize[i] == 'Compact':
            usageBand.append('Medium')
        else:
            usageBand.append('High')
    else:
        usageBand.append(cat_feat.UsageBand[i])


# In[52]:


usageBand_test = []
for i in range(len(cat_feat_test)):
    if cat_feat_test.UsageBand[i]=='-999' and cat_feat_test.ProductSize[i] != '-999':
        if cat_feat_test.ProductSize[i] == 'Mini':
            usageBand_test.append('Low')
        elif cat_feat_test.ProductSize[i] == 'Medium' or cat_feat_test.ProductSize[i] == 'Small' or cat_feat_test.ProductSize[i] == 'Compact':
            usageBand_test.append('Medium')
        else:
            usageBand_test.append('High')
    else:
        usageBand_test.append(cat_feat_test.UsageBand[i])


# In[55]:


# pd.DataFrame(usageBand, columns={'usageBand'})['usageBand'].value_counts()


# In[56]:


# pd.DataFrame(usageBand_test, columns={'usageBand'})['usageBand'].value_counts()


# In[58]:


# len(usageBand), len(usageBand_test)


# In[59]:


cat_feat['UsageBand'] = usageBand


# In[60]:


cat_feat_test['UsageBand'] = usageBand_test


# In[61]:


productSize = []
for i in range(len(cat_feat)):
    if cat_feat.ProductSize[i] == '-999' and cat_feat.UsageBand[i]!='-999':
        if cat_feat.UsageBand[i] == 'Low':
            productSize.append('Mini')
        elif cat_feat.UsageBand[i] == 'Medium':
            productSize.append('Medium')
        else:
            productSize.append('Large')
    else:
        productSize.append(cat_feat.ProductSize[i])


# In[62]:


productSize_test = []
for i in range(len(cat_feat_test)):
    if cat_feat_test.ProductSize[i] == '-999' and cat_feat_test.UsageBand[i]!='-999':
        if cat_feat_test.UsageBand[i] == 'Low':
            productSize_test.append('Mini')
        elif cat_feat_test.UsageBand[i] == 'Medium':
            productSize_test.append('Medium')
        else:
            productSize_test.append('Large')
    else:
        productSize_test.append(cat_feat_test.ProductSize[i])


# In[65]:


# pd.DataFrame(productSize)[0].value_counts()


# In[66]:


# pd.DataFrame(productSize_test)[0].value_counts()


# In[67]:


# len(productSize), len(productSize_test)


# In[68]:


cat_feat.ProductSize = productSize


# In[69]:


cat_feat_test.ProductSize = productSize_test


# In[70]:


saleyear = []
for i in range(len(cat_feat)):
    saleyear.append(int(cat_feat['saledate'][i].split()[0].split('/')[2]))


# In[71]:


num_feat['saleYear'] = saleyear


# In[72]:


num_feat['Age'] = num_feat['saleYear'] - num_feat['YearMade']


# In[73]:


num_feat.drop(['saleYear', 'YearMade'], axis=1, inplace=True)


# In[74]:


cat_feat.drop('saledate', axis=1, inplace=True)


# In[75]:


saleyear_test = []
for i in range(len(cat_feat_test)):
    saleyear_test.append(int(cat_feat_test['saledate'][i].split()[0].split('/')[2]))
num_feat_test['saleYear'] = saleyear_test
num_feat_test['Age'] = num_feat_test['saleYear'] - num_feat_test['YearMade']
num_feat_test['Age'] = num_feat_test['saleYear'] - num_feat_test['YearMade']
num_feat_test.drop(['saleYear', 'YearMade'], axis=1, inplace=True)
cat_feat_test.drop('saledate', axis=1, inplace=True)


# In[76]:


num_feat = pd.concat([num_feat, pd.get_dummies(cat_feat['UsageBand'])], axis=1)


# In[77]:


num_feat_test = pd.concat([num_feat_test, pd.get_dummies(cat_feat_test['UsageBand'])], axis=1)


# In[80]:


# cat_feat.shape


# In[81]:


# cat_feat_test.shape


# In[82]:


from fastai.imports import *
from fastai.structured import *


# In[83]:


train = pd.concat([num_feat, cat_feat.astype('category'), target], axis=1)


# In[84]:


test_df = pd.concat([num_feat_test, cat_feat_test.astype('category')], axis=1)


# In[85]:


#train.shape, test_df.shape


# In[86]:


#train.head()


# In[87]:


df, y, nas = proc_df(train, 'SalePrice')


# In[88]:

from sklearn.ensemble import RandomForestRegressor


# In[89]:


rf = RandomForestRegressor(n_jobs=-1)


# In[90]:


from sklearn.model_selection import train_test_split


# In[91]:


X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=1)


# In[92]:

print("Training...")
rf.fit(X_train, y_train)


# In[93]:


from sklearn.metrics import mean_squared_error


# In[94]:

print("Testing...")
pred = rf.predict(X_test)


# In[95]:


mse = mean_squared_error(y_test, pred)**1/2
print("Root mean squared error on the validation set :" ,mse )

# In[96]:


df.shape


# In[97]:


test_df.shape


# In[98]:


df_test,y,nas = proc_df(test_df)


# In[99]:


df.shape, df_test.shape


# In[100]:


# set(df.columns).difference(set(df_test.columns))


# In[101]:


data.auctioneerID.isnull().sum(), test.auctioneerID.isnull().sum()


# In[102]:


df_test['auctioneerID_na'] = False
# set(df.columns).difference(set(df_test.columns))


# In[103]:

print("Predicting on the test set...")
pred_test = rf.predict(df_test)


# In[104]:


pred_test = np.exp(pred_test)


# In[105]:


sub = pd.DataFrame({'SalesID': test_df.SalesID, 'SalePrice': pred_test})


# In[106]:

print("Saving the output as bulldozer.csv")
sub.to_csv('bulldozer.csv', index=False)


# In[ ]:

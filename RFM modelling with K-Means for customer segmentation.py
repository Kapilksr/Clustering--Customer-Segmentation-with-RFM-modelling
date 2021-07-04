#!/usr/bin/env python
# coding: utf-8

# In[111]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn import set_config
set_config(print_changed_only=False)


# In[2]:


data=pd.read_excel('Online Retail.xlsx')


# In[3]:


df=data
df.head()


# In[31]:


df.info()


# In[4]:


df.shape


# In[5]:


df['Country'].value_counts()


# In[7]:


df.isnull().sum()


# In[8]:


countrywise_customers=df[['Country','CustomerID']].drop_duplicates()


# In[10]:


countrywise_customers.groupby(['Country'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID',ascending=False)


# In[12]:


df1=df[df['Country']=='United Kingdom']


# In[18]:


df1=df1.reset_index(drop=True)


# In[20]:


df1.head()


# In[21]:


df1.isnull().sum()


# In[22]:


df1.shape


# In[23]:


df1=df1.dropna(axis=0)


# In[24]:


df1.shape


# In[25]:


df1.Quantity.min()


# In[26]:


df1.UnitPrice.min()


# In[27]:


df1=df1[df1['Quantity']>0]
df1.shape


# In[30]:


df1.InvoiceDate.dtype


# In[32]:


df1['Amount']=df1['Quantity']*df1['UnitPrice']


# In[33]:


df1.head()


# In[34]:


import datetime as dt


# In[36]:


df1.InvoiceDate.min()


# In[37]:


df1.InvoiceDate.max()


# In[38]:


Latest_Date=dt.datetime(2011,12,10)


# In[40]:


df_RFM=df1.groupby('CustomerID').agg({'InvoiceDate': lambda x:(Latest_Date-x.max()).days,
                                     'InvoiceNo' : lambda x:len(x),
                                     'Amount': lambda x:x.sum()})


# In[41]:


df_RFM.head()


# In[42]:


df_RFM['InvoiceDate'].dtype


# In[44]:


df_RFM=df_RFM.rename(columns={'InvoiceDate':'Recency',
                             'InvoiceNo':'Frequency',
                             'Amount':'Monetary'})


# In[45]:


df_RFM.head()


# In[46]:


df_RFM=df_RFM.reset_index()
df_RFM.head()


# In[48]:


df_RFM['Recency'].describe()


# In[50]:


sns.distplot(df_RFM['Recency'])


# In[51]:


df_RFM['Frequency'].describe()


# In[52]:


sns.distplot(df_RFM['Frequency'])


# In[53]:


df_RFM['Monetary'].describe()


# In[54]:


sns.distplot(df_RFM['Monetary'])


# In[55]:


## Arguments (x=value,p=recency,frequency or monetary value,d=quantiles dict)

## lower the recency better the result

def RScore(x,p,d):
    if x<=d[p][0.25]:
        return 4
    elif x<=d[p][0.50]:
        return 3
    elif x<=d[p][0.75]:
        return 2
    else:
        return 1

## Higher the frequency better the result

def FScore(x,p,d):
    if x<=d[p][0.25]:
        return 1
    elif x<=d[p][0.50]:
        return 2
    elif x<=d[p][0.75]:
        return 3
    else:
        return 4
    
    
## higher the monetary better the result

def MScore(x,p,d):
    if x<=d[p][0.25]:
        return 1
    elif x<=d[p][0.50]:
        return 2
    elif x<=d[p][0.75]:
        return 3
    else:
        return 4


# In[56]:


## quantile is like 25% , 50% and 75% level of values. example if we have 100 values first 25 in 1st quartile 25% 
## then second contain next 25% which is 50% and there after next 25% which is 75% and 4th quartile is more than 75% 

quantiles = df_RFM.quantile(q=[0.25,0.5,0.75])
quantiles


# In[57]:


quantiles.to_dict()


# In[58]:


df_RFM['R_quartile']=df_RFM['Recency'].apply(RScore,args=('Recency',quantiles))
df_RFM['F_quartile']=df_RFM['Frequency'].apply(FScore,args=('Frequency',quantiles))
df_RFM['M_quartile']=df_RFM['Monetary'].apply(MScore,args=('Monetary',quantiles))


# In[59]:


df_RFM.head()


# In[60]:


df_RFM['RFM_score']=df_RFM.R_quartile.map(str)+df_RFM.F_quartile.map(str)+df_RFM.M_quartile.map(str)
df_RFM.head()


# In[75]:


print("Best Customers: ",len(df_RFM[df_RFM['RFM_score']=='444']))
print('Loyal Customers: ',len(df_RFM[df_RFM['F_quartile']==4]))
print("Big Spenders: ",len(df_RFM[df_RFM['M_quartile']==4]))
print('Almost Lost: ', len(df_RFM[df_RFM['RFM_score']=='144']))
print('Lost Customers: ',len(df_RFM[df_RFM['RFM_score']=='114']) +len(df_RFM[df_RFM['RFM_score']=='113'])
     +len(df_RFM[df_RFM['RFM_score']=='112']))
print('Lost Cheap Customers: ',len(df_RFM[df_RFM['RFM_score']=='111']))


# In[76]:


df_RFM.groupby('RFM_score')['CustomerID'].count()


# In[77]:


df_RFM.groupby('RFM_score')['Monetary'].mean()


# In[98]:


plt.figure(figsize=(20,10))
df_RFM.groupby('RFM_score')['Monetary'].mean().plot(kind='bar', colormap='Blues_r')
plt.xlabel('RFM Score',size=20)
plt.ylabel('Mean Monetary Value',size=20)


# In[78]:


df_RFM.groupby('RFM_score')['Recency'].mean()


# In[101]:


plt.figure(figsize=(20,10))
df_RFM.groupby('RFM_score')['Recency'].mean().plot(kind='bar', colormap='Blues_r')
plt.xlabel('RFM Score',size=20)
plt.ylabel('Mean Recency Value',size=20)


# In[79]:


df_RFM.groupby('RFM_score')['Frequency'].mean()


# In[100]:


plt.figure(figsize=(20,10))
df_RFM.groupby('RFM_score')['Frequency'].mean().plot(kind='bar', colormap='Blues_r')
plt.xlabel('RFM Score',size=20)
plt.ylabel('Mean Frequency Value',size=20)


# In[106]:


df2=df_RFM[['Recency','Frequency','Monetary']]


# In[107]:


df2.head()


# In[108]:


from sklearn.cluster import KMeans


# In[109]:


from sklearn.preprocessing import StandardScaler


# In[112]:


scaler=StandardScaler()


# In[113]:


X_scaled=scaler.fit_transform(df2)
X_scaled


# In[157]:


wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=60)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)


# In[158]:


plt.figure(figsize=(15,5))
plt.plot(range(1,11),wcss)
plt.title('Elbow graph')
plt.xlabel('Number of K(clusters)')
plt.ylabel('WCSS')
plt.show()


# In[159]:


kmeans=KMeans(n_clusters=3,init='k-means++',random_state=60)


# In[160]:


Y=kmeans.fit_predict(X_scaled)


# In[161]:


clusters=kmeans.cluster_centers_


# In[162]:


kmeans.labels_


# In[163]:


df2['Label']=kmeans.labels_


# In[164]:


df2.head()


# In[165]:


df2.Label.value_counts()


# In[173]:


sns.scatterplot(df2['Frequency'],df2['Monetary'],hue='Label',palette='tab10',data=df2)


# In[167]:


sns.boxplot(df2.Label,df2.Recency)


# In[168]:


sns.boxplot(df2.Label,df2.Frequency)


# In[170]:


sns.boxplot(df2.Label,df2.Monetary)
("")


# In[171]:


## group them in platinum gold and silver  category


# ## 1- Platinum Category-  we see that the customers with label 2 are recent visitors and have higher freequency and have high monetory value so they fall in Platinum label

# ## 2- Gold category-  we see that the customers with label 0 have relative better recency than customers of label 1 and they have better frequency than customers of label 1 and monetary value is also better than them so they fall in our gold category

# ## 3- Silver Category- rest customers with label 1 are from silver category as their recency is not that good and same with other 2 categories

# In[ ]:





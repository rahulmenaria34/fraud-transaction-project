#!/usr/bin/env python
# coding: utf-8

# <h2 style="color:Red",align="center" >ML Project-Fraud Transcation Detection</h2>
# 

# # Import all required libraries

# In[2]:


import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
#from plotly.offline import iplot
#import plotly as py
#import plotly.tools as tls
#import cufflinks as cf
import matplotlib.pyplot as plt
# from IPython import get_ipython
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import warnings


# In[3]:


# get_ipython().system('pip install cassandra-driver')


# In[4]:


from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

cloud_config= {
        'secure_connect_bundle': 'C:/Users/krish/Downloads/secure-connect-lovelesh.zip'
}
auth_provider = PlainTextAuthProvider('bXGOCeXqINbxsPtUHCyZpYcS', 'p55Rdfj8emH2Y7WdXWGDhESMLr51d2u6H,T+,e+2N9l9XJhkfiLBQxwCeZY-W6ounly_RIKAigY3FkmFDy+Di0dse-X41C,2SHSf3l_I6t0ZSg9ICbKghXmrlPZb1UU0')
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()
session.set_keyspace("jindals")
session.default_fetch_size=None

count=0
results = session.execute("SELECT * FROM creditcard_project ",timeout=None)
for i in results:
    count+=1
    data=pd.DataFrame(results)


# In[5]:


data


# In[6]:


X = data.drop('Class', axis = 1)
Y=data["Class"]


# In[7]:


data.hist(figsize= (20,10) , color = 'r' , alpha  = .9)


# In[8]:


data.boxplot(figsize=(20,10))


# In[9]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)


# In[10]:


x_train


# In[11]:


x_test


# In[12]:


y_train


# In[13]:


y_test


# In[14]:


d_train = pd.concat([x_train,y_train], axis =1)
d_train


# In[15]:


class0 = d_train[d_train['Class']==0]

class1 = d_train[d_train['Class']==1]


# In[16]:


class0.head()


# In[17]:


class0.head()


# In[18]:


frames = ['Time', 'Amount']
x= d_train[frames]
y=d_train.drop(frames, axis=1)


# In[19]:


x.head()


# In[20]:


y.head()


# In[21]:


scaler = StandardScaler()


# In[22]:


temp_col=scaler.fit_transform(x)
pd.DataFrame(temp_col)


# In[23]:


scaled_col = pd.DataFrame(temp_col, columns=frames)
scaled_col


# In[24]:


d_temp = d_train.drop(frames, axis=1)
d_temp


# In[25]:


d_temp.reset_index()


# In[26]:


d_=d_temp.reset_index().drop("index",axis=1)
d_.head()


# In[27]:


d_scaled = pd.concat([scaled_col, d_], axis =1)
d_scaled


# In[28]:


X___= d_scaled.drop('Class', axis = 1)
Y___=pd.DataFrame(d_scaled["Class"])
Y___


# In[29]:


"""# Dimensionality Reduction"""

from sklearn.decomposition import PCA

pca = PCA(n_components=15)

X_temp_reduced = pca.fit_transform(d_scaled)


# In[30]:


X_temp_reduced = pca.fit_transform(X___)
X_reduce=pd.DataFrame(X_temp_reduced)
X_reduce


# In[31]:


pca.explained_variance_ratio_


# In[32]:


pca.explained_variance_


# In[33]:


new_data=pd.concat([X_reduce,Y___],axis=1)
new_data


# In[34]:


new_data.to_csv('final_data.csv')


# In[35]:


X_train, X_test, y_train, y_test= train_test_split(x,y['Class'], test_size = 0.25, random_state = 42)


# In[36]:


print(X_train.shape)
print(X_test.shape)


# In[37]:


print(y_train.shape)
print(y_test.shape)


# In[38]:


count_classes = pd.value_counts(data['Class'], sort = True)

count_classes.plot(kind = 'bar', rot=0,figsize=(20,10))

plt.title("Transaction Class Distribution Before the fit")

plt.xlabel("Class")

plt.ylabel("Frequency")


# In[39]:


# get_ipython().system('pip install --user imblearn')


# In[40]:


print(y_train.value_counts())
print(y_test.value_counts())


# In[41]:


from collections import Counter 
Counter(y_train)


# # SMOTE (Synthetic Minority Oversampling Technique) – Oversampling

# In[42]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(0.7,random_state = 42)


# In[43]:


X_train_Smote, Y_train_Smote=sm.fit_resample(X_train,y_train)


# In[44]:


X_train_Smote


# In[45]:


d_smote = pd.concat([X_train_Smote,Y_train_Smote], axis =1)
d_smote


# In[46]:


print("The Number of classes Before the fit {}".format(Counter(y_train)))
print("The Number of classes After the fit {}".format(Counter(Y_train_Smote)))


# In[47]:


class0_ = d_smote[d_smote['Class']==0]
class0_


# In[48]:


class1_ = d_smote[d_smote['Class']==1]
class1_.tail(n=20)


# In[49]:


count_classes = pd.value_counts(d_smote['Class'], sort = True)

count_classes.plot(kind = 'bar', rot=0,figsize=(20,10))

plt.title("Transaction Class Distribution After the fit ")

plt.xlabel("Class")

plt.ylabel("Frequency")


# # RANDOM FOREST

# In[50]:


from sklearn.ensemble import RandomForestClassifier
rf_SMOTE = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=20,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
rf_SMOTE


# In[51]:


rf_SMOTE.fit(X_train_Smote, Y_train_Smote)


# In[52]:


y_pred_regressor_rf1 = rf_SMOTE.predict(X_test)
y_pred_regressor_rf1


# In[53]:


RF_Accuracy_SMOTE = rf_SMOTE.score(X_test,y_test)
RF_Accuracy_SMOTE


# In[54]:


cm_regressor_rf1 = confusion_matrix(y_test, y_pred_regressor_rf1)
print(cm_regressor_rf1)
print("Accuracy score of the model:",accuracy_score(y_test, y_pred_regressor_rf1))
print(classification_report(y_test, y_pred_regressor_rf1))


# In[55]:


import pickle


# In[58]:


file = open('Fraud_Transaction_Detection_.pkl', 'wb')
pickle.dump(rf_SMOTE, file)


# In[59]:


f = pd.read_pickle(r'Fraud_Transaction_Detection_.pkl')
f


# In[ ]:





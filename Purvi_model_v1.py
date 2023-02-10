#!/usr/bin/env python
# coding: utf-8

# # Import all libraries

# In[102]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# # Load the dataset

# In[103]:


df = read_csv("breast-cancer-wisconsin (1).data", header=None)
df = read_csv("wdbc.data", header=None)
df.head


# In[104]:


dataset['target'].value_counts()


# In[105]:


X = dataset['data']
y = dataset['target']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.25, random_state=0)


# # Data preprocessing

# In[106]:


from sklearn.preprocessing import StandardScaler

ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)

ss_test = StandardScaler()
X_test = ss_test.fit_transform(X_test)


# # Build a model

# In[107]:


# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# In[108]:


# Make predictions on the test data
y_pred = model.predict(X_test)


# In[109]:


# Calculate the accuracy of the model
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)


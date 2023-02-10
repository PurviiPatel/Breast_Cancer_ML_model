#!/usr/bin/env python
# coding: utf-8

# # Import all libraries

# In[97]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# # Load the dataset

# In[98]:


df = read_csv("breast-cancer-wisconsin (1).data", header=None)
df = read_csv("wdbc.data", header=None)
df.head


# In[99]:


dataset['target'].value_counts()


# # Data preprocessing

# In[100]:


X = dataset['data']
y = dataset['target']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.25, random_state=0)


# # Build a model

# In[101]:


# Build the Decision Tree model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the model's performance
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv('feature.csv')


# In[4]:


data.head(2)


# In[ ]:





# In[5]:


data.keys()


# In[6]:


data_feat=data[['Mean', 'Standard_Deviation', 'Variance']]


# In[7]:


data_feat.head(2)


# In[8]:


class_names=['Port 1', 'Port 2']


# In[9]:


type(class_names)


# In[10]:


data_label=data[['label']]


# In[11]:


data_label.head(2)


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X=data_feat;
y=data_label
X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(y), test_size=0.30, random_state=101)


# In[14]:


from sklearn.svm import SVC


# In[15]:


model=SVC()


# In[16]:


model.fit(X_train, y_train)


# In[17]:


predictions = model.predict(X_test)


# In[18]:


from sklearn.metrics import classification_report,confusion_matrix


# In[19]:


print(confusion_matrix(y_test,predictions))


# In[20]:


print(classification_report(y_test,predictions))


# In[21]:


from sklearn.grid_search import GridSearchCV


# In[22]:


param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 


# In[23]:


grid= GridSearchCV(SVC(), param_grid, refit=True, verbose=3)


# In[24]:


grid.fit(X_train, y_train)


# In[25]:


grid.best_params_


# In[26]:


grid.best_estimator_


# In[27]:


grid_predictions = grid.predict(X_test)


# In[28]:


grid_predictions


# In[29]:


print(confusion_matrix(y_test,grid_predictions))


# In[30]:


print(classification_report(y_test, grid_predictions))


# In[31]:


plt.scatter(y_test, grid_predictions)


# In[58]:


import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
   
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, grid_predictions)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
beingsaved = plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

beingsaved.savefig('destination_path.png', format='png', dpi=1000)
# Plot normalized confusion matrix
beingsaved1 = plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
beingsaved1.savefig('destination_path1.png', format='png', dpi=1000)
plt.show()


# In[ ]:





# In[ ]:





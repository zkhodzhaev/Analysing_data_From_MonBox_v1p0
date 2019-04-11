#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


data = pd.read_csv('data')

data_feat=data[['mean', 'std', 'var']]

class_names=['Sensor 1', 'Sensor 2']

data_label=data[['label']]


from sklearn.model_selection import train_test_split


X=data_feat;
y=data_label
X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(y), test_size=0.1, random_state=55,shuffle=True)

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1, 10, 30], 'gamma': [80,50,40,14,12,10,8,6,4,2,1.8,1.5,1.4,1.3,1.2,1.19,1.18,1.17], 'kernel': ['rbf']} 

grid= GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

grid.fit(X_train, y_train)

grid_predictions = grid.predict(X_test)


print(confusion_matrix(y_test,grid_predictions))

print(classification_report(y_test, grid_predictions))

plt.scatter(y_test, grid_predictions)

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
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 06:26:55 2024

@author: CHANDRA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# load data - read from the .csv files
def load_dataset(path):
    df = pd.read_csv(path)
    return df.dropna(axis=1, how='all')
    
train_df = load_dataset('../Dataset/dataset_cybersecurity_michelle.csv')
X = train_df.iloc[:, :-1]
y = train_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
f1_score = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)
print("test accuracy :", clf.score(X_test, y_test))
print(classification_report(y_test, y_pred))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.show()


accuracies=cross_val_score(estimator=clf,X=X_train,y=y_train,cv=5)

print("average accuracy :",np.mean(accuracies))
print("average std :",np.std(accuracies))
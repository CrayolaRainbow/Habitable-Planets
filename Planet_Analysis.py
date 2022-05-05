# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:15:32 2022

@author: Rachel
"""
 
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import scipy.stats as ss


df = pd.read_csv('C:/Users/Rachel/Downloads/planet_19/planet_19.csv')
df = df.dropna()
X = df.iloc[:,:-1]
y = df.iloc[:,-1]


X.describe()
ss.describe(X)
ss.ttest_ind(X,y)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)

not_habitable = 0

habitable = 0

for i in y_train:
    if i == 0:
        not_habitable+=1
    else:
        habitable+=1
        
values = np.array([not_habitable, habitable])
label = ['Not-Habitable', 'Habitable']

plt.pie(values, labels = label)
#plt.bar(label, values)
plt.show()

print("Habitable : ", habitable, '\n Not Habitable : ', not_habitable)


###### UNDERSAMPLING ##########



print("Before undersampling: ", Counter(y_train))

undersample = RandomUnderSampler(sampling_strategy='majority')

X_under, y_under = undersample.fit_resample(X, y)



not_habitable = 0

habitable = 0

for i in y_under:
    if i == 0:
        not_habitable+=1
    else:
        habitable+=1
        
values = np.array([not_habitable, habitable])
label = ['Not-Habitable', 'Habitable']

#plt.pie(values, labels = label)
plt.bar(label, values)
plt.show()

print("Habitable : ", habitable, '\n Not Habitable : ', not_habitable)

X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(X_under,y_under,test_size=0.30)
classifier= LogisticRegression()
classifier.fit(X_train_under,y_train_under)

print(classifier.coef_, classifier.intercept_)

y_pred = classifier.predict(X_test_under)
print(accuracy_score(y_test_under, y_pred))

cm = confusion_matrix(y_test_under, y_pred, labels=classifier.classes_)

sns.heatmap(cm,annot=True)

print(classification_report(y_test_under, y_pred))



###### UNDERSAMPLING ##########



###### OVERSAMPLING ##########



from imblearn.over_sampling import SMOTE


SMOTE = SMOTE()

X_SMOTE, y_SMOTE = SMOTE.fit_resample(X,y)

not_habitable = 0

habitable = 0

for i in y_SMOTE:
    if i == 0:
        not_habitable+=1
    else:
        habitable+=1
        
values = np.array([not_habitable, habitable])
label = ['Not-Habitable', 'Habitable']

plt.bar(label, values)
plt.show()

print("Habitable : ", habitable, '\n Not Habitable : ', not_habitable)

X_train_SMOTE, X_test_SMOTE, y_train_SMOTE, y_test_SMOTE = train_test_split(X_SMOTE,y_SMOTE,test_size=0.30)
classifier= LogisticRegression()
classifier.fit(X_train_SMOTE,y_train_SMOTE)

print(classifier.coef_, classifier.intercept_)

y_pred = classifier.predict(X_test_SMOTE)
print(accuracy_score(y_test_SMOTE, y_pred))

cm = confusion_matrix(y_test_SMOTE, y_pred, labels=classifier.classes_)

sns.heatmap(cm,annot=True)

print(classification_report(y_test_SMOTE, y_pred))

classifier.coef_
classifier.intercept_
###### OVERSAMPLING ##########






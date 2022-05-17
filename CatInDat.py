# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 23:48:53 2020

@author: rasto
"""


""" Importing the dataset """

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

""" Loading the dataset """

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

""" Plotting the Graph """
x = sns.countplot(train.target, label = "Count")
ax.scatter(x = train['nom_5'], y = train['target'])
plt.xlabel('nom_0', fontsize = 13)
plt.ylabel('target', fontsize = 13)
plt.show()

"""finding corelation(heat Map)"""

cormat = train.corr()
plt.subplots(figsize = (12,9))
sns.heatmap(cormat,vmax = 1.0,vmin=0,square = True,annot = True)

cormat = test.corr()
plt.subplots(figsize = (12,9))
sns.heatmap(cormat,vmax = 1.0,vmin=0,square = True,annot = True)




""" Merging the two dataset """

df = pd.concat([train,test])
df['bin_3'] = df['bin_3'].apply(lambda x: 1 if x == 'T' else (0 if x == 'F' else None))
df['bin_4'] = df['bin_4'].apply(lambda x: 1 if x == 'Y' else (0 if x == 'N' else None))




""" Ordinal encoding the ordinal data """

from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder() 
df['ord_1'] = oe.fit_transform(df['ord_1'].values.reshape(-1,1)) 
df['ord_2'] = oe.fit_transform(df['ord_2'].values.reshape(-1,1))


""" one hot encoding using dummies """

df = pd.get_dummies(df, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'],
                          prefix=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'], drop_first=True)
df.head(10)

#categorical encoding ordinal data using categorical datatype
'''from pandas.api.types import CategoricalDtype 
ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g',
                                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)
ord_4 = CategoricalDtype(categories=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                                     'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                                     'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], ordered=True)'''

""" Frequency encoding the nominal data """

fq = df.groupby('ord_3').size()/len(df)    
# mapping values to dataframe 
df.loc[:, "{}_freq_encode".format('ord_3')] = df['ord_3'].map(fq)   
# drop original column. 
df = df.drop(['ord_3'], axis = 1)  
df.head(10)

fq1 = df.groupby('ord_4').size()/len(df)    
# mapping values to dataframe 
df.loc[:, "{}_freq_encode".format('ord_4')] = df['ord_4'].map(fq1)   
# drop original column. 
df = df.drop(['ord_4'], axis = 1)  

                                                                     

""" Encoding using ascii values for ord_5 """

import string
df['ord_5_oe1'] = df['ord_5'].apply(lambda x:(string.ascii_letters.find(x[0])+1))

df['ord_5_oe2'] = df['ord_5'].apply(lambda x:(string.ascii_letters.find(x[1])+1))


for col in ['ord_5_oe1', 'ord_5_oe2']:
    df[col]= df[col].astype('float64')
    

df.drop(['ord_5'],axis = 1, inplace = True)

df.drop(['id'],axis = 1, inplace = True)


""" Cyclic Encoding """

columns=['day','month']
for col in columns:
    df[col+'_sin']=np.sin((2*np.pi*df[col])/max(df[col]))
    df[col+'_cos']=np.cos((2*np.pi*df[col])/max(df[col]))

df.drop(['day','month'],axis = 1, inplace = True)
df.head(10)


""" Hash Encoding the nominal Features """

nom_features = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
for col in nom_features:
    df[f'hash_{col}'] = df[col].apply( lambda x: hash(str(x)) % 5000 )
    
df.drop(['nom_5','nom_6','nom_7','nom_8','nom_9'],axis = 1, inplace = True)

df.head(10)


""" Seperating the train and test data """

dtrain = df.head(300000)
dtest = df.tail(200000)

dtest = dtest.drop(['target'],axis = 1)

Y = dtrain['target']
dtrain = dtrain.drop(['target'],axis =1)

""" Splitting the train and test data """

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =train_test_split(dtrain, Y, test_size = 0.2, random_state = 0) 

""" Model used: Logistic Rregression """

'''from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train) '''


""" Model used: Random Forest Regressor """

from sklearn.ensemble import RandomForestClassifier
classfier = RandomForestClassifier(n_estimators = 100, random_state = 0)
classfier.fit(X_train, Y_train)


""" Predicting the Y values """

Y_pred = classfier.predict(dtest)
Y_pred_probab = classfier.predict_proba(dtest)[:,1]


""" Confusion Matrix"""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)


""" Finding Mean Square Log Error """

from sklearn.metrics import mean_squared_log_error
print("Score = ",mean_squared_log_error(Y_test,Y_pred))

""" Fnding  the accuracy score """

from sklearn.metrics import accuracy_score
print('Accuracy : ',accuracy_score(Y_test,Y_pred))


""" Submitting the results """

ID=test["id"]
submissionRandom = pd.DataFrame(
    {'id': ID, 'target':Y_pred_probab,})
submissionRandom.to_csv('submissioncat.csv', index=False)
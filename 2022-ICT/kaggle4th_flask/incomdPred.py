import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

print(os.getcwd())
path = "D:\\Github\\MachineLearning_Basic_Class\\ch07_project_flask_ml\\kaggle4th_flask\\"
train = pd.read_csv(path + "data\\train.csv")
test = pd.read_csv(path + "data\\test.csv")

print(train.shape)
print(train.columns)
print(train.info())
print(train.head())

train.loc[ train['income']=='>50K' , 'target'] = 1
train.loc[ train['income']=='<=50K' , 'target'] = 0
train['target'] = train.target.astype("int")

# 'capital_gain', 'capital_loss'
sel = ['age', 'fnlwgt', 'education_num', 'hours_per_week']

X = train[sel]
y = train['target']

test_X = test[sel]

print(X.describe())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                   stratify=train.target,
                                                   random_state=42)

print(X.describe())
# model = RandomForestClassifier(random_state=30).fit(X_train, y_train)
# score = cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc')
# print("교차 검증 점수(AUC) - rf : ", np.mean(score))

model = GradientBoostingClassifier(random_state=30).fit(X_train, y_train)
score = cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc')
print("교차 검증 점수(AUC) - gd : ", np.mean(score))

# 최종 모델
pickle.dump(model, open(path+'\\model\\income_gb.pkl', 'wb'))


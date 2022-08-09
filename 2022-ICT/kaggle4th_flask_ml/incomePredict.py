# 01. CTRL + SHIFT + P 로 인터프리터 선택
# 02. ml_app 을 선택한다.

import pandas as pd
import numpy as np
import pickle
# import seaborn as sns 
import sklearn as sk
import flask

print(pd.__version__)
print(np.__version__)
print(sk.__version__)
print(flask.__version__)

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

import os

print(os.getcwd())

train = pd.read_csv("data\\train.csv")
test = pd.read_csv("data\\test.csv")
sub = pd.read_csv("data\\sample_submission.csv")

print(train.shape, test.shape, sub.shape)
print(train.columns)
print(train.info())
print(train.head())

train.loc[(train['income']=='>50K'), 'target'] = 1
train.loc[(train['income']=='<=50K'), 'target'] = 0

train['target'] = train['target'].astype('int')

lb = LabelEncoder()
sex_lb = train['sex']
sex_lb = pd.DataFrame(sex_lb)
sex_lb['sex_lb'] = lb.fit_transform(train['sex'])
work_lb = train['workclass']
work_lb = pd.DataFrame(work_lb)
work_lb['work_lb'] = lb.fit_transform(train['workclass'])
train = pd.concat([train, sex_lb['sex_lb'], work_lb['work_lb']], axis=1)
sex_lb = test['sex']
sex_lb = pd.DataFrame(sex_lb)
sex_lb['sex_lb'] = lb.fit_transform(test['sex'])
work_lb = test['workclass']
work_lb = pd.DataFrame(work_lb)
work_lb['work_lb'] = lb.fit_transform(test['workclass'])
test = pd.concat([test, sex_lb['sex_lb'], work_lb['work_lb']], axis=1)

print(train.head())
print(train.value_counts('work_lb'))

sel = ['age', 'fnlwgt', 'education_num', 'hours_per_week', 'sex_lb', 'work_lb']
X = train[sel]
y = train['target']
test_X = test[sel]

X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,random_state=77)

model = GradientBoostingClassifier(max_depth=2).fit(X_train, y_train)
score = cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc')
print(score, score.mean())

pickle.dump(model, open('model\\income_base.pkl', 'wb'))
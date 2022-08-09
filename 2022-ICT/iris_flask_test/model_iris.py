import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

import os

print(os.getcwd())

df = pd.read_csv("data\\iris.csv")
print(df.shape)
print(df.columns)

df.loc[ df['Species']=='Iris-setosa', 'target'] = 0
df.loc[ df['Species']=='Iris-versicolor', 'target'] = 1
df.loc[ df['Species']=='Iris-virginica', 'target'] = 2
df['target'] = df['target'].astype("int")

print(df.head())

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# 데이터 선택
sel = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df[sel]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 기본 모델 만들어보기(평가) 및 pkl(모델) 파일로 저장하기
model_rf = RandomForestClassifier(random_state=30).fit(X_train, y_train)
score = cross_val_score(model_rf, X_test, y_test, cv=5, scoring='accuracy')
print("교차 검증 점수(AUC) - 랜덤포레스트 : ", np.mean(score))

model_gd = GradientBoostingClassifier(random_state=30).fit(X_train, y_train)
score = cross_val_score(model_gd, X_test, y_test, cv=5, scoring='accuracy')
print("교차 검증 점수(AUC) - 그래디언트 부스팅 : ", np.mean(score))

pickle.dump(model_rf, open("ml_model\\iris_base.pkl", 'wb'))
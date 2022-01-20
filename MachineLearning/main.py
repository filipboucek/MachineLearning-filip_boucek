import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time

model = LogisticRegression()

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')

print("---------------DESCRIPTION---------------")
print(df.describe())
print("---------------TRAIN&TEST----------------")

X = df[['Pclass', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values

y = df['Survived'].values

X_train,X_test,y_train,y_test = train_test_split(X,y)

print("train set: ",X_train.shape,y_train.shape)
print("test set: ", X_test.shape," ",y_test.shape)

model.fit(X_train,y_train)

print("---------------RESULTS-------------------")

modelScore = model.score(X_test,y_test)
print(round(modelScore,2)*100,"%")

print("-----------------------------------------")

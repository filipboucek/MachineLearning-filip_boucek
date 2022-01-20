from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/home/filip/Desktop/MachineLearning/titanic.csv')

df['male'] = df['Sex'] == 'male'

y = df['Survived']
X = df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']]
benchmark = []
knn = KNeighborsClassifier(n_neighbors = 1)
lgr = LogisticRegression()

#Benchmark na test_size
i=0
testSplitNumber=0
XAxisNumbers=[]
XAxisCounter=0
testSplitNumberExponent=0.05
whileLoopLenght = 1 / testSplitNumberExponent -5
while i<=whileLoopLenght:
	print("benchmark loop: ",i)
	testSplitNumber+=testSplitNumberExponent
	XAxisCounter+=testSplitNumberExponent*100
	X_train,X_test, y_train,y_test = train_test_split(X,y,test_size = testSplitNumber, random_state=20)
	lgr.fit(X_train,y_train)
	benchmark.append(lgr.score(X_test,y_test)*100)
	XAxisNumbers.append(XAxisCounter)
	i+=1

#Predikce s optimální test_size
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.35, random_state=20)
knn.fit(X_train,y_train)
lgr.fit(X_train,y_train)

print("\n \n \n \n")
print("K-Nearest neighbours results:")
print("accuracy: ",round(knn.score(X_test,y_test),3)*100,"%")
print("\n accuracy train: ", knn.score(X_train,y_train))
print("\n accuracy raw:", knn.score(X_test,y_test))
print("\n \n")

print("Logistic regression:")
print("accuracy: ",round(lgr.score(X_test,y_test),3)*100,"%")
print("\n accuracy train: ",lgr.score(X_test,y_test))
print("\n accuracy raw: ", lgr.score(X_test,y_test))

plt.plot(XAxisNumbers,benchmark)
plt.ylabel('Model accuracy %')
plt.xlabel('Size of train split %')
plt.show()



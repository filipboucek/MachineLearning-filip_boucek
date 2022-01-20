from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/home/filip/Desktop/MachineLearning/dataset.csv')
#df.drop('date',axis=1)
y = df['deaths']
X = df[['cases','date']]
benchmark = []
knn = KNeighborsClassifier(n_neighbors = 2)

#Benchmark na test_size
i=0
testSplitNumber=0
XAxisNumbers=[]
XAxisCounter=0
testSplitNumberExponent=0.01
whileLoopLenght = 1 / testSplitNumberExponent -5
while i<=whileLoopLenght:
	print("benchmark loop: ",i)
	testSplitNumber+=testSplitNumberExponent
	XAxisCounter+=testSplitNumberExponent*100
	X_train,X_test, y_train,y_test = train_test_split(X,y,test_size = testSplitNumber, random_state=20)
	knn.fit(X_train,y_train)
	benchmark.append(round(knn.score(X_test,y_test),3)*100)
	XAxisNumbers.append(XAxisCounter)
	i+=1

#Predikce s optimální test_size
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.16, random_state=20)
knn.fit(X_train,y_train)

print("\n \n \n \n")
print("accuracy: ",round(knn.score(X_test,y_test),3)*100,"%")
print("\n accuracy raw:", knn.score(X_test,y_test))


plt.plot(benchmark)
plt.ylabel('Model accuracy %')
plt.xlabel('Size of train split %')
plt.show()
#print(y)

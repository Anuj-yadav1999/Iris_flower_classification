# Iris_flower_classification
# Here i have just used knn 

# Now importing all libraries

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from sklearn import datasets

# The dataset is also availabe in sklearn library
data = datasets.load_iris()

X = data.data
Y = data.target

# Here we can just import a important function train_test_split to split out data
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train ,Y_test = train_test_split(X,Y,test_size = 0.2)

# Now making the object of knn algorithm
knn = KNeighborsClassifier()

# Jut fitting the data to the algorithm 
a = knn.fit(X_train,Y_train)

# For testing the data
y_pred = a.predict(X_test)

print("Pred====",y_pred)


# Here the important thing "confusion matrix" to evaluate the output easily
cm = confusion_matrix(Y_test,y_pred)

print("The confusion metrix==\n",cm)

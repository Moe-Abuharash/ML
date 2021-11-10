# The Iris dataset is referred to as a “toy dataset” because it has only 150 samples and four features.
# The dataset describes 50 samples for each of three Iris flower species—Iris setosa, Iris versicolor and Iris
# virginica. Each sample’s features are the sepal length, sepal width, petal
# length and petal width, all measured in centimeters. The sepals are the larger outer parts of each flower
# that protect the smaller inside petals before the flower buds bloom.

# EXERCISE
# load the iris dataset and use classification
# to see if the expected and predicted species
# match up
# from math import exp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# display the shape of the data, target and target_names
iris = load_iris()
print(iris.data.shape)
print(iris.target.shape)
print(iris.target_names)

# display the first 10 predicted and expected results using
# the species names not the number (using target_names)
data_train, data_test, target_train, target_test = train_test_split(
    iris.data, iris.target, random_state=11
)
# display the values that the model got wrong
print(data_train.shape)
print(target_train.shape)
print(data_test.shape)
print(target_test.shape)
# visualize the data using the confusion matrix
knn.fit(x=data_train, y=target_train)

predicted = knn.predicted(x=data_test)

expected = target_test

print(predicted[:20])
print(expected[:20])
print(iris.target_names)

predicted = [iris.target_names[x] for x in predicted]
expected = [iris.target_names[x] for x in expected]

print(predicted[:20])
print(expected[:20])

confusion = confusion_matrix(y_true=expected,)
confusion_df = pd.DataFrame(confusion, index=range(3), columns=range(3))
figure = plt.figure()
axes = sns.heatmap(confusion_df, annot=True, cmap=plt.cm.nipy_spectral_r))
plt.xlabel("Expected")
plt.ylabel("Predicted")
plt.show
print("Done")



'''Look back at LEACTURE VIDEO TO FIX THIS CODE!!!'''

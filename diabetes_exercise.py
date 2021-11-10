''' Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line '''

from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.datasets import load_diabetes
import seaborn as sns

# how many sameples and How many features?

diabetes = datasets.load_diabetes()
print(diabetes.data.shape)

# What does feature s6 represent?
print(diabetes.DESCR)

# print out the coefficient

x_train, x_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, random_state=11
)

mymodel = LinearRegression()

mymodel.fit(X=x_train, y=y_train)

coef = mymodel.coef_
print(coef)

# print out the intercept
intercept = mymodel.intercept_
print(intercept)

# step 3: use predict to test your model
predcited = mymodel.predict(x_test)
expected = y_test

# print(predcited[:20])
# print(expected[:20])

# create a scatterplot with regression line


plt.plot(expected, predcited, ".")
# plt.show

x = np.linspace(0, 330, 100)
print(x)
y = x

plt.plot(x, y)
plt.show()


'''
def predict(x): return coef * x + intercept


print(predict(2025))


axes = sns.scatterplot(
    data=diabetes,
    x="Date",
    y="Temperature",
    hue="Temperature",
    palette="winter",
    legend=False
)

axes.set_ylim(10, 70)


x = np.array([min(diabetes.Date.values), max(diabetes.Date.values)])
print(x)
y = predict(x)
print(y)

line = plt.plot(x, y)

plt.show()
'''

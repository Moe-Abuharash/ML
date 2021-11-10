from re import X
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sys import platform
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split

nyc = pd.read_csv("ave_hi_nyc_jan_1895-2018.csv")

print(nyc.head(3))

print(nyc.Date.values)

# the -1 tells it to create as many rows as you need to make the columns
print(nyc.Date.values.reshape(-1, 1))

x_train, x_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1, 1), nyc.Temperature.values, random_state=11
)


lr = LinearRegression()

lr.fit(X=x_train, y=y_train)

coef = lr.coef_
intercept = lr.intercept_

predcited = lr.predict(x_test)
expected = y_test

print(predcited[:20])
print(expected[:20])


def predict(x): return coef * x + intercept


print(predict(2025))


axes = sns.scatterplot(
    data=nyc,
    x="Date",
    y="Temperature",
    hue="Temperature",
    palette="winter",
    legend=False
)

axes.set_ylim(10, 70)


x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
print(x)
y = predict(x)
print(y)

line = plt.plot(x, y)

plt.show()

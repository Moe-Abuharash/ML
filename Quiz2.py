import csv
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv("gameratings.csv")
test = pd.read_csv("test_esrb.csv")
names = pd.read_csv("target_names.csv")

x_train = train.loc[0:, 'console':'violence']
y_train = train.loc[0:, 'Target']

x_test = test.loc[0:, 'console':'violence']
y_test = test.loc[0:, 'Target']

knn = KNeighborsClassifier()

knn.fit(X=x_train, y=y_train)

predicted = knn.predict(X=x_test)
expected = y_test

knn1 = {}
y = 0


for x in names.T:
    target_class = names.target_class.values[y]
    target_name = names.target_name.values[y]
    y += 1
    knn1[target_class] = target_name

predicted = [knn1[x] for x in predicted]
expected = [knn1[x] for x in expected]

# print(predicted[:20])
# print(expected[:20])

wrong = [(p, e) for (p, e) in zip(predicted, expected) if p != e]
print(wrong)


f = open("List of Predictions.csv", 'w')
f.write('title, prediction\n')

y = 0
for x in test.T:
    f.write(test.values[y][0] + ',' + predicted[y] + '\n')
    y += 1
f.close()

# pip install sklearn, pandas, seaborn, matplotlib
#from itertools import Predicate
import matplotlib.pyplot as plt2
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from re import X
from sklearn.datasets import load_digits

digits = load_digits()

# print(digits.DESCR)

# print(digits.data[5])
# print(digits.target[5])

# print(digits.data.shape)
# print(digits.target.shape)


figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6, 4))

# plt.show()

for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)

plt.tight_layout()
# plt.show()


data_train, data_test, target_train, target_test = train_test_split(
    digits.data, digits.target, random_state=11
)

print(data_train.shape)
print(data_test.shape)
print(target_train.shape)
print(target_test.shape)


knn = KNeighborsClassifier()

knn.fit(X=data_train, y=target_train)

predicted = knn.predict(X=data_test)
expected = target_test

print(predicted[:20])
print(expected[:20])

print(format(knn.score(data_test, target_test), ".2%"))

wrong = [(p, e) for (p, e) in zip(predicted, expected) if p != e]

print(wrong)


confusion = confusion_matrix(y_true=expected, y_pred=predicted)

print(confusion)


confusion_df = pd.DataFrame(confusion, index=range(10), columns=range(10))

figure = plt2.figure(figsize=(7, 6))
axes = sns.heatmap(confusion_df, annot=True, cmap=plt2.cm.nipy_spectral_r)

plt2.show()

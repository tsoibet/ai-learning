import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

df = pd.read_csv('./data/sample08_2_1.csv')
t1 = df[df.type == 1]
t2 = df[df.type == 2]
plt.plot(t1['x'], t1['y'], 'ro')
plt.plot(t2['x'], t2['y'], 'b*')

clf = svm.LinearSVC()
clf.fit(df[['x', 'y']], df['type'])

# clf.predict([[10, 5], [10, 10], [15, 5]]) returns the predicted types of coor[10, 5], [10, 10] and [15, 5] respectively.

print('''Formula for decision boundary is
ax + by + c = 0
y = (-ax - c) / b''')
print("where [a, b] =", clf.coef_[0], "and c =", clf.intercept_)

# Visualise the linear decision boundary
a = clf.coef_[0][0]
b = clf.coef_[0][1]
c = clf.intercept_[0]
x = np.arange(5, 35)
plt.plot(x, (-a * x - c) / b)
plt.grid()
plt.show()
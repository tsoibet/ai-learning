import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn import model_selection

# Q1
df = pd.read_csv('./data/sample08_5_1.csv')
reg = linear_model.LinearRegression()
reg.fit(df[['x']], df['y'])
print("Q1 Answer:", reg.predict([[30], [40]]))

# Q2
df = pd.read_csv('./data/sample08_5_2.csv')
clf = svm.SVC()
clf.fit(df[['x', 'y']], df['type'])
print("Q2 Answer:", clf.predict([[0, 0], [25, 25]]))
x = np.arange(0, 50, 0.1)
y = x
X, Y = np.meshgrid(x, y)
pair = np.c_[X.ravel(), Y.ravel()]
Z = clf.predict(pair)
t1 = df[df.type == 1]
t2 = df[df.type == 2]
plt.subplot(2, 2, 1)
plt.title("Q2")
plt.plot(t1['x'], t1['y'], 'ro')
plt.plot(t2['x'], t2['y'], 'b*')
plt.contour(X, Y, Z.reshape(X.shape))
plt.grid()

# Q3
df = pd.read_csv('./data/sample08_5_3.csv')
data = df[['x', 'y']]
label = df['type']
clf = svm.SVC()
clf.fit(data, label)
scores = model_selection.cross_val_score(clf, data, label, cv=5)
print("Q3 Answer:", scores.mean())
x = np.arange(0, 10, 0.1)
y = x
X, Y = np.meshgrid(x, y)
pair = np.c_[X.ravel(), Y.ravel()]
Z = clf.predict(pair)
t1 = df[df.type == 1]
t2 = df[df.type == 2]
plt.subplot(2, 2, 2)
plt.title("Q3")
plt.plot(t1['x'], t1['y'], 'ro')
plt.plot(t2['x'], t2['y'], 'b*')
plt.contour(X, Y, Z.reshape(X.shape))
plt.grid()

# Q4
df = pd.read_csv('./data/sample08_5_3.csv')
data = df[['x', 'y']]
label = df['type']
params = [
  {'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [0.5, 0.1, 0.05, 0.01]}
]
clf = model_selection.GridSearchCV(svm.SVC(), params, cv=5)
clf.fit(data, label)
print("Q4 Answer:", clf.best_params_)

plt.show()
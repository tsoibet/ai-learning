import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn import model_selection

df = pd.read_csv('./data/sample08_3_2.csv')

t1 = df[df.type == 1]
t2 = df[df.type == 2]
plt.plot(t1['x'], t1['y'], 'ro')
plt.plot(t2['x'], t2['y'], 'b*')

data = df[['x', 'y']]
label = df['type']
clf = svm.SVC(kernel='rbf', gamma='auto')
clf.fit(data, label)

# Cross Validation
scores = model_selection.cross_val_score(clf, data, label, cv=5)
print("Average accuracy of clf model is", scores.mean())

# Visualisation of decision boundary
x = np.arange(0, 50, 0.1)
y = x
X, Y = np.meshgrid(x, y)
pair = np.c_[X.ravel(), Y.ravel()]
Z = clf.predict(pair)
plt.contour(X, Y, Z.reshape(X.shape))
plt.grid()
plt.show()
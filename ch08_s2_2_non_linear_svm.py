import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

df = pd.read_csv('./data/sample08_2_3.csv')
t1 = df[df.type == 1]
t2 = df[df.type == 2]

# Non-linear decision boundary formed with 
# Kernel Trick 1: Polynomial Kernel
clf1 = svm.SVC(kernel='poly')
clf1.fit(df[['x', 'y']], df['type'])
plt.subplot(2, 1, 1)
plt.title('Polynomial Kernel')
plt.plot(t1['x'], t1['y'], 'ro')
plt.plot(t2['x'], t2['y'], 'b*')
x = np.arange(0, 10, 0.1)
y = x
X, Y = np.meshgrid(x, y)
pair = np.c_[X.ravel(), Y.ravel()]
Z = clf1.predict(pair)
plt.contour(X, Y, Z.reshape(X.shape))
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.grid()

# Non-linear decision boundary formed with 
# Kernel Trick 2: Radial Basis Function (RBF) Kernel
clf2 = svm.SVC(kernel='rbf', gamma='auto')
clf2.fit(df[['x', 'y']], df['type'])
plt.subplot(2, 1, 2)
plt.title('RBF Kernel')
plt.plot(t1['x'], t1['y'], 'ro')
plt.plot(t2['x'], t2['y'], 'b*')
Z = clf2.predict(pair)
plt.contour(X, Y, Z.reshape(X.shape))
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.grid()

plt.show()
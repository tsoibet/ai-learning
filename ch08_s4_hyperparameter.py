import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn import model_selection

df = pd.read_csv('./data/sample08_3_2.csv')
t1 = df[df.type == 1]
t2 = df[df.type == 2]
data = df[['x', 'y']]
label = df['type']

# Hyperparameters
params = [
  {'kernel': ['rbf'], 'C':[1, 10, 100, 1000], 'gamma': [0.5, 0.1, 0.05, 0.01]}
]

# Grid Search
clf = model_selection.GridSearchCV(svm.SVC(), params, cv=5)
clf.fit(data, label)
print("The best parameters are ", clf.best_params_, "with score of", clf.best_score_)
# Detailed results of different combinations of hyperparameters: 
# results = pd.DataFrame(clf.cv_results_)
# print(results[['params', 'mean_test_score']])

# Visualisation of the decision boundary formed with default parameters
clf_default = svm.SVC(kernel="rbf", gamma='auto')
clf_default.fit(data, label)
plt.subplot(2, 1, 1)
plt.title('Default parameters')
plt.plot(t1['x'], t1['y'], 'ro')
plt.plot(t2['x'], t2['y'], 'b*')
x = np.arange(0, 50, 0.1)
y = x
X, Y = np.meshgrid(x, y)
pair = np.c_[X.ravel(), Y.ravel()]
Z = clf_default.predict(pair)
plt.contour(X, Y, Z.reshape(X.shape))
plt.grid()

# Visualisation of the decision boundary formed with best hyperparameters
clf_best = svm.SVC(**clf.best_params_)
clf_best.fit(data, label)
plt.subplot(2, 1, 2)
plt.title('Best hyperparameters')
plt.plot(t1['x'], t1['y'], 'ro')
plt.plot(t2['x'], t2['y'], 'b*')
x = np.arange(0, 50, 0.1)
y = x
X, Y = np.meshgrid(x, y)
pair = np.c_[X.ravel(), Y.ravel()]
Z = clf_best.predict(pair)
plt.contour(X, Y, Z.reshape(X.shape))
plt.grid()

plt.show()


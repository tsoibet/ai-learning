import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv('./data/sample08_1_1.csv')
plt.plot(df['x'], df['y'], 'ro')

reg = linear_model.LinearRegression()
reg.fit(df[['x']], df['y'])

print("y = ax + b, where a =", reg.coef_[0], "and b =", reg.intercept_)
# reg.predict([[100], [110], [120]]) returns the predicted values of y when x is 100, 110 and 120 respectively.

# Visualise linear regression result
x = np.arange(10, 100, 1)
y = reg.coef_[0] * x + reg.intercept_
plt.plot(x, y)
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


x = np.array([[1], [2], [3], [4], [5]])                     # PLR
y = np.array([2, 4, 9, 16, 25])

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)


model = LinearRegression()
model.fit(x_poly, y)

x_new = poly.transform([[10]])
pred = model.predict(x_new)


print(pred)


x_sorted = np.linspace(min(x), max(x), 100).reshape(-1, 1)
x_poly_sorted = poly.transform(x_sorted)
x_poly_pred = model.predict(x_poly_sorted)

plt.scatter(x, y, color='blue', label='Original Data')
plt.plot(x_sorted, x_poly_pred, color='green')
plt.xlabel("X - Feature")
plt.ylabel("Y - Target")
plt.title ("Polynomial Regression Model")
plt.show()

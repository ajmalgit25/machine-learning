import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression


x = np.array([[1], [2], [3], [4], [5]])                    # SLR
y = np.array([2, 4, 5, 6, 7])

model = LinearRegression()
model.fit(x, y)

pred = model.predict([[10]])

print(pred)

plt.scatter(x, y, color='blue', label='Original Data')
plt.plot(x, model.predict(x), color='green')
plt.xlabel("X - Feature")
plt.ylabel("Y - Target")
plt.title ("Polynomial Regression Model")
plt.show()

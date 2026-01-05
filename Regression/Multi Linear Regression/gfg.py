import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import xlwt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_excel("../../../Data_Files/Body_Mass_Index.xlsx")

X = dataset.iloc[:, [1,2,3]]
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


X_train = pd.DataFrame(X_train, columns=['Height','Weight','Age'])
X_test = pd.DataFrame(X_test, columns=X.columns)    # columns=['Height','Weight','Age']

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred_train = regressor.predict (X_train)
y_pred_test = regressor.predict (X_test)


print("Predicted BMI on Training Set:", y_pred_train)
print("Predicted BMI on Test Set:", y_pred_test)


plt.scatter(X_train['Height'], Y_train, color='red', label="Actual Training Data")
plt.scatter(X_train['Height'], y_pred_train, color='blue', label="Predicted Training Data")

plt.xlabel("Height (cm)")
plt.ylabel("BMI")
plt.title("Height vs. BMI (Training Data)")
plt.show()

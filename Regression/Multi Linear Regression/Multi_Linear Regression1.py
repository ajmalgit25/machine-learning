import numpy as np
import pandas as pd
import xlrd
import xlwt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_excel("../Data_Files/House_Price.xlsx")

X = dataset.iloc[:, [0,1,2]].values      #['Area', 'Bedrooms', 'Age']
y = dataset.iloc[:, 3].values                      #'Price'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state = 0);

regressor = LinearRegression();

regressor.fit(X_train, y_train);
print(dataset)
print(X)
print(y)
# Make predictions
y_pred = regressor.predict(X_test)
print(y_pred)
print("Price is:",regressor.predict([[1500,3,10]]))




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#importing the dataset
dataset = pd.read_csv("../Data_Files/Experience_Salary.csv")
#dataset = pd.DataFrame([[12,11,13,11,14,12],[1.1,1.05,1.2,1.07,1.24,1.12]])
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


#Splitting dataset into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#Training Simple Linear Regression model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results5
y_pred = regressor.predict(X_test)

#Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
#plt.show()

print("X_train: ", X_train)
print("y_predict: ", y_pred)

### Taking user input for prediction
##X_input = float(input("Enter a value for X: "))
##X_input_array = np.array([[X_input]])
##y_output = regressor.predict(X_input_array)
##print(f"Predicted Salary: ${y_output[0][0]:,.2f}")

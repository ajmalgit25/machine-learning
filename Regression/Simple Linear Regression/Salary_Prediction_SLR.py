import pandas as pd
import numpy as np
import xlrd
import xlwt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("../../../Data_Files/Experience_Salary.csv")


# Ensure X is a DataFrame (to retain column names)
X = dataset.iloc[:, :-1]            # Keeping it as DataFrame
Y = dataset.iloc[:, -1].values      # Convert target column to NumPy array
#print (Y)

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size=0.3, random_state=0)

# Initialize and Train model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# Predictions
y_pred_train = regressor.predict(X_train)   # Predictions on training data
y_pred_test = regressor.predict(X_test)     # Predictions on test data
#print(y_pred)


#Visualising the Training set results
plt.scatter(X_train, Y_train, color = 'red', label="Actual Training Data")
plt.plot(X_train, y_pred_train, color = 'blue', label="Regression Line")
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
#plt.show()

plt.scatter(X_test, Y_test, color = 'red', label="Actual Test Data")
plt.plot(X_test, y_pred_test, color = 'blue', label="Regression Line")
plt.title('Salary vs Experience (Testing set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
#plt.show()


user_input = float(input("Enter Year of Experience: "))
X_new = pd.DataFrame ([[user_input]], columns=X.columns)   #  columns=["Experience"]

pred_salary = regressor.predict(X_new)
print(f"Predicted Salary for {user_input} years is: {pred_salary[0]}")  # OR use .item()




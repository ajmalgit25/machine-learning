import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import xlwt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_excel("../../../Data_Files/Body_Mass_Index.xlsx")
#print(dataset.columns)  # Show actual column names


X = dataset.iloc[:, [1,2,3]]            # dataset[['Height', 'Weight', 'Age']]
Y = dataset.iloc[:, -1].values          # Target (BMI)

# Splitting dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


# Convert train and test sets to DataFrames
X_train = pd.DataFrame(X_train, columns=['Height','Weight','Age'])
X_test = pd.DataFrame(X_test, columns=X.columns)    # columns=['Height','Weight','Age']


# Initialize and train the model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# Predictions on training data and test data
y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

print("Predicted BMI on Training Set:", y_pred_train)
print("Predicted BMI on Test Set:", y_pred_test)




# Scatter plot: Height vs. Actual BMI

plt.scatter(X_train['Height'], Y_train, color='red', label="Actual Training Data")
plt.scatter(X_train['Height'], y_pred_train, color='blue', label="Predicted Training Data")
#plt.plot(X_train['Height'], y_pred_train, color='blue', label="Predicted BMI (Training)", linestyle='dashed')
plt.xlabel("Height (cm)")
plt.ylabel("BMI")
plt.title("Height vs. BMI (Training Data)")
plt.legend()
plt.show()


height = float(input("Enter Height (cm): "))
weight = float(input("Enter Weight (kg): "))
age = int(input("Enter Age (years): "))

# Creating a DataFrame for user input
user_input = pd.DataFrame([[height, weight, age]], columns=['Height', 'Weight', 'Age'])

# Predict BMI for the user input
predicted_bmi = regressor.predict(user_input)
print(f"Predicted BMI: {predicted_bmi}")         # predicted_bmi[0] ensures print a single value


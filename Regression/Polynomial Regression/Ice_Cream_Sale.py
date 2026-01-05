import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


# Load dataset
dataset = pd.read_csv("../../../Data_Files/Ice_cream_Selling_data.csv")
#print (dataset)

# Extract Features (X) & Target (Y)
X = dataset.iloc[:, [0]]
Y = dataset.iloc[:, [1]].values

#print (dataset.columns)

# Splitting dataset training & testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


# Applying Polynomial Transformation (degree=2 for quadratic regression)
poly = PolynomialFeatures(degree=2)


# Fit & Transform only on training data
X_poly_train = poly.fit_transform(X_train)

# Transform test data (using already learned transformation)
X_poly_test = poly.transform(X_test)


# Initialize and Train the model
regressor = LinearRegression()
regressor.fit(X_poly_train, Y_train)


# Predictions
y_pred_train = regressor.predict(X_poly_train)
y_pred_test = regressor.predict(X_poly_test)


# Visualizing Polynomial Regression Fit
X_range = np.linspace(min(X.values), max(X.values), 100).reshape(-1, 1)    # range for plotting

X_range_df = pd.DataFrame(X_range, columns=X_train.columns)
X_range_poly = poly.transform(X_range_df)
y_range_pred = regressor.predict(X_range_poly)



plt.scatter(X, Y, color='blue', label="Original Data")
plt.plot(X_range, y_range_pred, color='red', label="Polynomial Regression Fit")
plt.xlabel("Temperature (°C)")
plt.ylabel("Ice Cream Sales (units)")
plt.legend()
plt.title("Polynomial Regression Model")
plt.show()


# Predictions for user input

user_input = float(input("Enter temperature (°C): "))
X_new = pd.DataFrame([[user_input]], columns=X.columns)

# Apply Polynomial Transformation
X_new_poly = poly.transform(X_new)  # Transform input using the same poly object

pred_sale = regressor.predict(X_new_poly)

print (f"Predicted sale for {user_input}°C = {pred_sale[0][0]} unit!")


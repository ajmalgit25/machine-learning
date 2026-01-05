import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Sample dataset
data = {'X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Y': [2, 4, 9, 16, 25, 36, 49, 64, 81, 100]}


# Convert to DataFrame
dataset = pd.DataFrame(data)


# Extracting X (feature) and y (target)
X = dataset[['X']]          # Feature should be 2D
Y = dataset['Y']            # Target is 1D


# Splitting dataset training & testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# Applying Polynomial Transformation (degree=2 for quadratic regression)
poly = PolynomialFeatures(degree=2)  # You can change the degree
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)


# Training the model
model = LinearRegression()
model.fit(X_poly_train, Y_train)

# Predictions
y_pred_test = model.predict(X_poly_test)


# Visualizing Polynomial Regression Fit
X_sorted = pd.DataFrame(np.linspace(min(X.values), max(X.values), 100).reshape(-1, 1))

X_poly_sorted = poly.transform(X_sorted)
y_sorted_pred = model.predict(X_poly_sorted)

# Plot the results
plt.scatter(X, Y, color='blue', label="Original Data")
plt.plot(X_sorted, y_sorted_pred, color='red', label="Polynomial Regression Fit")
plt.xlabel("X - Feature")
plt.ylabel("Y - Target")
plt.legend()
plt.title("Polynomial Regression Model")
plt.show()


user_input = int(input("Enter value of X: "))
X_new = pd.DataFrame ([[user_input]], columns=X.columns)

# Apply polynomial transformation
X_new_poly = poly.transform(X_new)  # ✅ Now has 3 features (X, X², 1)

# Make prediction
predicted_Y = model.predict(X_new_poly)

print (f"Y prediction for {user_input} = {predicted_Y[0]}")


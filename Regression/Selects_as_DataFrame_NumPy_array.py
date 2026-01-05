import numpy as np
import pandas as pd

dataset = pd.read_csv("../../Data_Files/Experience_Salary.csv")

X = dataset.iloc[:, :]          # Selects as a pandas DataFrame
Y = dataset.iloc[:, :].values   # Selects as a NumPy array (.to_numpy() can be used)

print (X)           # Output as DataFrame
print (Y)           # Output (NumPy array format)


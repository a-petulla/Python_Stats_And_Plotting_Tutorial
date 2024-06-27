#!/usr/bin/env python
# coding: utf-8

# # Linear Modeling in Python

import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sys


# Check if the correct number of arguments is provided
if len(sys.argv) != 2:
    print("Usage: python script.py <parameter>")
    sys.exit(1)

# Retrieve the parameter from command-line arguments
filename = sys.argv[1]


df = pd.read_csv(filename)

# Extract x and y values from the DataFrame
x = df['x']
y = df['y']

# Create a scatter plot
plt.figure(figsize=(8, 6))  # Optional: adjust figure size

plt.scatter(x, y, color='blue', label='Data Points')  # Scatter plot
plt.title('Scatter Plot of X vs Y')  # Title
plt.xlabel('X')  # X-axis label
plt.ylabel('Y')  # Y-axis label
#plt.legend()  # Show legend (optional)

plt.grid(False)  # Optional: add grid
plt.tight_layout()  # Optional: improve layout

plt.savefig('py_orig.png')

model = LinearRegression()


# Have to reshape the first column of the dataframe
x = np.array(df['x']).reshape(-1, 1)
y = np.array(df['y'])


model.fit(x,y)

intercept = model.intercept_
slope = model.coef_[0]
y_pred = model.predict(x)
r_squared = r2_score(y, y_pred)


# Plotting the data points and the linear regression line
plt.figure(figsize=(8, 6))
plt.plot(x, y_pred, color='red', linewidth=2, label='Linear Regression')  # Line plot of the linear regression line

plt.title('Linear Regression Model')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.grid(True)
plt.tight_layout()


# Plotting the data points and the linear regression line
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Data Points')  # Scatter plot of the data points
plt.plot(x, y_pred, color='red', linewidth=2, label='Linear Regression')  # Line plot of the linear regression line

plt.title('Linear Regression Model')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.grid(True)
plt.tight_layout()

plt.savefig('py_lm.png')


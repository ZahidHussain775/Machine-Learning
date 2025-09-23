import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data (Study Hours vs Test Scores)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # feature (independent variable)
y = np.array([2, 4, 5, 4, 5])  # target (dependent variable)

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Print coefficients
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

# Visualization
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, y_pred, color="red", label="Regression line")
plt.xlabel("Study Hours")
plt.ylabel("Test Score")
plt.legend()
plt.show()

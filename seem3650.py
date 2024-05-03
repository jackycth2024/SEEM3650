import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
population_data = pd.read_csv('population_data.csv')
transportation_data = pd.read_csv('transportation_data.csv')

# Merge population and transportation data
merged_data = pd.merge(population_data, transportation_data, on='Year')

# Check for missing values and drop rows with missing values
print("\nMissing Values:")
print(merged_data.isnull().sum())
merged_data.dropna(inplace=True)

# Split data into features and target variables
X = merged_data[['Population', 'Growth Rate']]
y_ridership = merged_data['Ridership']
y_train_frequency = merged_data['Train frequency']

# Split data into training and testing sets
X_train, X_test, y_ridership_train, y_ridership_test = train_test_split(X, y_ridership, test_size=0.2, random_state=42)
X_train, X_test, y_train_frequency_train, y_train_frequency_test = train_test_split(X, y_train_frequency, test_size=0.2, random_state=42)

# Define regression models
ridge_model_ridership = Ridge(alpha=1.0, random_state=42)
lasso_model_ridership = Lasso(alpha=1.0, random_state=42)
ridge_model_train_frequency = Ridge(alpha=1.0, random_state=42)
lasso_model_train_frequency = Lasso(alpha=1.0, random_state=42)

# Fit regression models
ridge_model_ridership.fit(X_train, y_ridership_train)
lasso_model_ridership.fit(X_train, y_ridership_train)
ridge_model_train_frequency.fit(X_train, y_train_frequency_train)
lasso_model_train_frequency.fit(X_train, y_train_frequency_train)

# Make predictions using regression models
y_ridership_pred_ridge = ridge_model_ridership.predict(X_test)
y_ridership_pred_lasso = lasso_model_ridership.predict(X_test)
y_train_frequency_pred_ridge = ridge_model_train_frequency.predict(X_test)
y_train_frequency_pred_lasso = lasso_model_train_frequency.predict(X_test)

# Evaluate model performance
print("\nModel Performance for MTR Ridership (Ridge Regression):")
print("Mean Squared Error:", mean_squared_error(y_ridership_test, y_ridership_pred_ridge))
print("R-squared:", r2_score(y_ridership_test, y_ridership_pred_ridge))


print("\nModel Performance for Train Frequency (Ridge Regression):")
print("Mean Squared Error:", mean_squared_error(y_train_frequency_test, y_train_frequency_pred_ridge))
print("R-squared:", r2_score(y_train_frequency_test, y_train_frequency_pred_ridge))

# Visualize results
plt.figure(figsize=(12, 12))

# Scatter plot for population vs MTR ridership
plt.subplot(2, 2, 1)
plt.scatter(merged_data['Population'], merged_data['Ridership'], color='blue', label='Actual Ridership')
plt.plot(merged_data['Population'], ridge_model_ridership.predict(X), color='red', linestyle='--', label='Ridge Regression')
plt.title('Population vs MTR Ridership')
plt.xlabel('Population')
plt.ylabel('MTR Ridership')
plt.legend()

# Scatter plot for MTR ridership vs Train Frequency
plt.subplot(2, 2, 2)
plt.scatter(merged_data['Ridership'], merged_data['Train frequency'], color='blue', label='Actual Train Frequency')
plt.plot(merged_data['Ridership'], ridge_model_train_frequency.predict(X), color='red', linestyle='--', label='Ridge Regression')
plt.title('MTR Ridership vs Train Frequency')
plt.xlabel('MTR Ridership')
plt.ylabel('Train Frequency')
plt.legend()

plt.tight_layout()
plt.show()

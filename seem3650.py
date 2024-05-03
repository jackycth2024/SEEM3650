import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RANSACRegressor, LinearRegression
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

# Define RANSAC regression models
ransac_model_ridership = RANSACRegressor(LinearRegression(), random_state=42)
ransac_model_train_frequency = RANSACRegressor(LinearRegression(), random_state=42)

# Fit RANSAC regression models
ransac_model_ridership.fit(X_train, y_ridership_train)
ransac_model_train_frequency.fit(X_train, y_train_frequency_train)

# Make predictions using RANSAC regression models
y_ridership_pred_ransac = ransac_model_ridership.predict(X_test)
y_train_frequency_pred_ransac = ransac_model_train_frequency.predict(X_test)

# Evaluate model performance
print("\nModel Performance for MTR Ridership:")
print("Mean Squared Error:", mean_squared_error(y_ridership_test, y_ridership_pred_ransac))
print("R-squared:", r2_score(y_ridership_test, y_ridership_pred_ransac))

print("\nModel Performance for Train Frequency:")
print("Mean Squared Error:", mean_squared_error(y_train_frequency_test, y_train_frequency_pred_ransac))
print("R-squared:", r2_score(y_train_frequency_test, y_train_frequency_pred_ransac))

# Visualize results with RANSAC regression
plt.figure(figsize=(12, 6))

# Scatter plot for population vs MTR ridership with RANSAC regression line
plt.subplot(1, 2, 1)
plt.scatter(merged_data['Population'], merged_data['Ridership'], color='blue', label='Actual Ridership')
plt.plot(merged_data['Population'], ransac_model_ridership.predict(merged_data[['Population', 'Growth Rate']]), color='red', label='Predicted Ridership')
plt.title('Population vs MTR Ridership')
plt.xlabel('Population')
plt.ylabel('MTR Ridership')
plt.legend()

# Scatter plot for MTR ridership vs Train Frequency with RANSAC regression line
plt.subplot(1, 2, 2)
plt.scatter(merged_data['Ridership'], merged_data['Train frequency'], color='blue', label='Actual Train Frequency')
plt.plot(merged_data['Ridership'], ransac_model_train_frequency.predict(merged_data[['Population', 'Growth Rate']]), color='red', label='Predicted Train Frequency')
plt.title('MTR Ridership vs Train Frequency')
plt.xlabel('MTR Ridership')
plt.ylabel('Train Frequency')
plt.legend()

plt.tight_layout()
plt.show()
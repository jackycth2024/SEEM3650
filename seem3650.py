import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

population_data = pd.read_csv('population_data.csv')
transportation_data = pd.read_csv('transportation_data.csv')

# Exploratory Data Analysis (EDA)
print("Population Data:")
print(population_data.head())

# Explore transportation data
print("\nTransportation Data:")
print(transportation_data.head())

# Merge population and transportation data
merged_data = pd.merge(population_data, transportation_data, on='date')

# Check for missing values and drop rows with missing values
print("\nMissing Values:")
print(merged_data.isnull().sum())
merged_data.dropna(inplace=True)

# Split data into features and target variable
X = merged_data[['population', 'growth_rate']]
y_ridership = merged_data['ridership']
y_waiting_time = merged_data['waiting_time']

# Split data into training and testing sets
X_train, X_test, y_ridership_train, y_ridership_test = train_test_split(X, y_ridership, test_size=0.2, random_state=42)
X_train, X_test, y_waiting_time_train, y_waiting_time_test = train_test_split(X, y_waiting_time, test_size=0.2, random_state=42)

# Build regression models
# Model for predicting MTR ridership
model_ridership = LinearRegression()
model_ridership.fit(X_train, y_ridership_train)

# Model for predicting average waiting time
model_waiting_time = LinearRegression()
model_waiting_time.fit(X_train, y_waiting_time_train)

# Make predictions
y_ridership_pred = model_ridership.predict(X_test)
y_waiting_time_pred = model_waiting_time.predict(X_test)

# Evaluate model performance
print("\nModel Performance for MTR Ridership:")
print("Mean Squared Error:", mean_squared_error(y_ridership_test, y_ridership_pred))
print("R-squared:", r2_score(y_ridership_test, y_ridership_pred))

print("\nModel Performance for Average Waiting Time:")
print("Mean Squared Error:", mean_squared_error(y_waiting_time_test, y_waiting_time_pred))
print("R-squared:", r2_score(y_waiting_time_test, y_waiting_time_pred))

# Visualize results
plt.figure(figsize=(12, 6))

# Scatter plot for MTR ridership predictions vs. actual values
plt.subplot(1, 2, 1)
plt.scatter(y_ridership_test, y_ridership_pred, color='blue')
plt.plot([min(y_ridership_test), max(y_ridership_test)], [min(y_ridership_test), max(y_ridership_test)], color='red', linestyle='--')
plt.title('MTR Ridership: Actual vs. Predicted')
plt.xlabel('Actual Ridership')
plt.ylabel('Predicted Ridership')

# Scatter plot for average waiting time predictions vs. actual values
plt.subplot(1, 2, 2)
plt.scatter(y_waiting_time_test, y_waiting_time_pred, color='green')
plt.plot([min(y_waiting_time_test), max(y_waiting_time_test)], [min(y_waiting_time_test), max(y_waiting_time_test)], color='red', linestyle='--')
plt.title('Average Waiting Time: Actual vs. Predicted')
plt.xlabel('Actual Waiting Time')
plt.ylabel('Predicted Waiting Time')

plt.tight_layout()
plt.show()
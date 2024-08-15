import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('home_prices.csv')

# Define features and target variable
X = data[['income', 'schools', 'hospitals', 'crime_rate']]
y = data['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Example new data
new_data = pd.DataFrame({
    'income': [80000, 60000],
    'schools': [7, 5],
    'hospitals': [3, 2],
    'crime_rate': [0.2, 0.5]
})

# Standardize the new data
new_data_scaled = scaler.transform(new_data)

# Predict home prices
price_predictions = model.predict(new_data_scaled)
print("Predicted Home Prices:", price_predictions)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

data = pd.read_csv("E:\\PROJECTS\\ML BASED PROJECT\\DATA SET\\House_Rent_Dataset.csv")
print(data.head())

print(data.isnull().sum())

print(data.describe())

print(f"Mean Rent: {data.Rent.mean()}")
print(f"Median Rent: {data.Rent.median()}")
print(f"Highest Rent: {data.Rent.max()}")
print(f"Lowest Rent: {data.Rent.min()}")

import matplotlib.pyplot as plt
import seaborn as sns

# Bar chart: Rent by City and BHK
plt.figure(figsize=(10,6))
sns.barplot(x='City', y='Rent', hue='BHK', data=data)
plt.title("Rent in Different Cities According to BHK")
plt.show()

# Bar chart: Rent by City and Area Type
plt.figure(figsize=(10,6))
sns.barplot(x='City', y='Rent', hue='Area Type', data=data)
plt.title("Rent in Different Cities According to Area Type")
plt.show()

# Bar chart: Rent by City and Furnishing Status
plt.figure(figsize=(10,6))
sns.barplot(x='City', y='Rent', hue='Furnishing Status', data=data)
plt.title("Rent in Different Cities According to Furnishing Status")
plt.show()

# Bar chart: Rent by City and Size
plt.figure(figsize=(10,6))
sns.barplot(x='City', y='Rent', hue='Size', data=data)
plt.title("Rent in Different Cities According to Size")
plt.show()


import matplotlib.pyplot as plt

# Pie chart: Number of Houses Available for Rent by City
city_counts = data["City"].value_counts()
city_labels = city_counts.index
city_values = city_counts.values
city_colors = ['gold', 'lightgreen'] * (len(city_labels)//2 + 1)  # repeat colors if more cities

plt.figure(figsize=(8, 8))
plt.pie(city_values, labels=city_labels, colors=city_colors[:len(city_labels)],
        autopct='%1.1f%%', startangle=140, wedgeprops={'linewidth': 3, 'edgecolor': 'black'})
plt.title('Number of Houses Available for Rent')
plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
plt.show()

# Pie chart: Tenant Preference
tenant_counts = data["Tenant Preferred"].value_counts()
tenant_labels = tenant_counts.index
tenant_values = tenant_counts.values
tenant_colors = ['gold', 'lightgreen'] * (len(tenant_labels)//2 + 1)

plt.figure(figsize=(8, 8))
plt.pie(tenant_values, labels=tenant_labels, colors=tenant_colors[:len(tenant_labels)],
        autopct='%1.1f%%', startangle=140, wedgeprops={'linewidth': 3, 'edgecolor': 'black'})
plt.title('Preference of Tenant in India')
plt.axis('equal')
plt.show()

data["Area Type"] = data["Area Type"].map({"Super Area": 1, 
                                           "Carpet Area": 2, 
                                           "Built Area": 3})
data["City"] = data["City"].map({"Mumbai": 4000, "Chennai": 6000, 
                                 "Bangalore": 5600, "Hyderabad": 5000, 
                                 "Delhi": 1100, "Kolkata": 7000})
data["Furnishing Status"] = data["Furnishing Status"].map({"Unfurnished": 0, 
                                                           "Semi-Furnished": 1, 
                                                           "Furnished": 2})
data["Tenant Preferred"] = data["Tenant Preferred"].map({"Bachelors/Family": 2, 
                                                         "Bachelors": 1, 
                                                         "Family": 3})
print(data.head())


from sklearn.model_selection import train_test_split
x = np.array(data[["BHK", "Size", "Area Type", "City", 
                   "Furnishing Status", "Tenant Preferred", 
                   "Bathroom"]])
y = np.array(data[["Rent"]])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.10, 
                                                random_state=42)

from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, 
               input_shape= (xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size=1, epochs=10)

print("Voting Classifier")
# Define regressors
rf = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
gbr = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)
xgbr = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)

# Create Voting Regressor
voting_reg = VotingRegressor(estimators=[
    ('rf', rf),
    ('gbr', gbr),
    ('xgbr', xgbr)
])

# Train ensemble
voting_reg.fit(xtrain, ytrain)

# Evaluate
y_pred = voting_reg.predict(xtest)
print("Voting Regressor RMSE:", np.sqrt(mean_squared_error(ytest, y_pred)))
print("Voting Regressor R^2:", r2_score(ytest, y_pred))

# Prediction function
def predict_rent(BHK, Size, Area_Type, City, Furnishing_Status, Tenant_Preferred, Bathroom):
    input_arr = np.array([[BHK, Size, Area_Type, City, Furnishing_Status, Tenant_Preferred, Bathroom]])
    prediction = voting_reg.predict(input_arr)
    return prediction[0]

xgbr = xgb.XGBRegressor(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    objective='reg:squarederror'  # Use squared error for regression
)

# Train the model
xgbr.fit(xtrain, ytrain.ravel())  # Use ravel to convert ytrain to 1d array if necessary

# Predict on test set
y_pred = xgbr.predict(xtest)

# Prediction based on user input
def predict_rent_xgb(BHK, Size, Area_Type, City, Furnishing_Status, Tenant_Preferred, Bathroom):
    input_arr = np.array([[BHK, Size, Area_Type, City, Furnishing_Status, Tenant_Preferred, Bathroom]])
    prediction = xgbr.predict(input_arr)
    return prediction[0]


print("Enter House Details to Predict Rent")
a = int(input("Number of BHK: "))
b = int(input("Size of the House: "))
c = int(input("Area Type (Super Area = 1, Carpet Area = 2, Built Area = 3): "))
d = int(input("Pin Code of the City: "))
e = int(input("Furnishing Status of the House (Unfurnished = 0, Semi-Furnished = 1, Furnished = 2): "))
f = int(input("Tenant Type (Bachelors = 1, Bachelors/Family = 2, Only Family = 3): "))
g = int(input("Number of bathrooms: "))

print("LSTM Model")
features = np.array([[a, b, c, d, e, f, g]])
print("Predicted House Price Based in LSTM = ", model.predict(features))


predicted_rent = predict_rent(a, b, c, d, e, f, g)
print("Predicted House Rent Based on Voting Classifier = ", predicted_rent)

# Example usage (using your inputs a,b,c,d,e,f,g)
print("XG-Boost")
predicted_rent_xgb = predict_rent_xgb(a, b, c, d, e, f, g)
print("Predicted House Rent from XGBoost = ", predicted_rent_xgb)
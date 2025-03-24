import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Generate sample data
np.random.seed(42)
data = pd.DataFrame({
    "Age": np.random.randint(18, 60, 100),
    "BMI": np.random.uniform(18, 35, 100),
    "Duration": np.random.randint(10, 50, 100),
    "Heart Rate": np.random.randint(60, 130, 100),
    "Body Temp": np.random.uniform(36, 40, 100),
    "Gender_Male": np.random.choice([0, 1], 100),
})
data["Calories"] = data["Duration"] * 5 + data["Heart Rate"] * 0.2 + data["BMI"] * 3

# Train-Test Split
X = data.drop("Calories", axis=1)
y = data["Calories"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save Model
with open("fitness_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained and saved!")

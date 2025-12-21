import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load temperature data
data = pd.read_csv("temperature_data.csv")

X = data[["Temperature"]]
y = data["Status"]

# Train AI model
model = DecisionTreeClassifier()
model.fit(X, y)

print("AI model trained successfully!")

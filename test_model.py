import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load data
data = pd.read_csv("temperature_data.csv")

X = data[["Temperature"]]
y = data["Status"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Test input
test_temperature = [[37]]
prediction = model.predict(test_temperature)

print("Temperature:", test_temperature[0][0])
print("Status:", prediction[0])

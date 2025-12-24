import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load data
data = pd.read_csv("temperature_data.csv")

X = data[["Temperature"]]
y = data["Status"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Ask user for temperature
temp = float(input("Enter temperature: "))
test_temperature = [[temp]]

prediction = model.predict(test_temperature)

print("Temperature:", temp)
print("Status:", prediction[0])

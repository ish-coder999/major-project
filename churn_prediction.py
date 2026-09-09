import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("customer_churn.csv")

# Convert categorical columns into numbers
encoder = LabelEncoder()

data["Gender"] = encoder.fit_transform(data["Gender"])
data["Contract"] = encoder.fit_transform(data["Contract"])

# Separate features and target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create machine learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, predictions))

import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load placement dataset
data = pd.read_csv("placement_data.csv")

# Input features
X = data[["cgpa", "placement_exam_marks"]]

# Target
y = data["placed"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create ML model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# Save trained model
joblib.dump(model, "placement_model.pkl")

# Create metrics
metrics = {
    "accuracy": accuracy,
    "training_records": len(X_train),
    "testing_records": len(X_test)
}

# Save metrics
with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

# Save test predictions
predictions_data = X_test.copy()
predictions_data["actual_placed"] = y_test.values
predictions_data["predicted_placed"] = predictions

predictions_data.to_csv(
    "placement_predictions.csv",
    index=False
)

print("Placement model trained successfully!")
print("Accuracy:", accuracy)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Model saved as placement_model.pkl")
print("Metrics saved as metrics.json")
print("Predictions saved as placement_predictions.csv")

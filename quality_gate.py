import json

# Read model metrics
with open("metrics.json", "r") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]

print("Model Accuracy:", accuracy)

# Quality requirement
if accuracy >= 0.45:
    print("QUALITY GATE PASSED")
else:
    print("QUALITY GATE FAILED")
    exit(1)

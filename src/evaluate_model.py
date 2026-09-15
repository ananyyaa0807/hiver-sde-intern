import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

MODEL_PATH = "data/demo_intent_model.pkl"
GOLDEN_PATH = "data/golden_set.csv"

# Load model
model = joblib.load(MODEL_PATH)

# Load human-reviewed evaluation set
df = pd.read_csv(GOLDEN_PATH)

X = df["text"].fillna("")
y_true = df["gold_intent"]

# Predict
y_pred = model.predict(X)

# Accuracy
accuracy = accuracy_score(y_true, y_pred)

print("=" * 70)
print("GOLDEN SET EVALUATION")
print("=" * 70)

print("\nNumber of examples:", len(df))
print("Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
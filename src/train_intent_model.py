import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib


INPUT_PATH = "data/spotify_labeled.csv"
MODEL_PATH = "data/intent_model.pkl"


# Load data
df = pd.read_csv(INPUT_PATH)

X = df["text"].fillna("")
y = df["intent"]


# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Build model
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])


# Train
print("Training model...")
model.fit(X_train, y_train)


# Predict
predictions = model.predict(X_test)


# Evaluate
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy, 4))

print("\nClassification report:")
print(classification_report(y_test, predictions))


# Save model
joblib.dump(model, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)
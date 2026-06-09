import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "train.csv"))

x = df["text"]
y = df["label"]

# Create pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression()),
])

# Training
pipeline.fit(x, y)

model_path = os.path.join(BASE_DIR, "models", "model.pkl")
joblib.dump(pipeline, model_path)

# Evaluation train set
accuracy = pipeline.score(x, y)
print(f"Accuracy sur le train set : {accuracy:.4f}")

# Load test data
df_test = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "test.csv"))

x_test = df_test["text"]
y_test = df_test["label"]

# Evaluation test set
accuracy_test = pipeline.score(x_test, y_test)
print(f"Accuracy sur le test set : {accuracy_test:.4f}")
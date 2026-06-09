import os
import joblib
import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load train data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "train.csv"))

x_train = df["text"]
y_train = df["label"]

# Load test data
df_test = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "test.csv"))

x_test = df_test["text"]
y_test = df_test["label"]

# Mlflow run
with mlflow.start_run():
    # Create pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression()),
    ])

    # Training
    pipeline.fit(x_train, y_train)

    # Evaluation train set
    accuracy_train = pipeline.score(x_train, y_train)
    print(f"Accuracy sur le train set : {accuracy_train:.4f}")

    # Evaluation test set
    accuracy_test = pipeline.score(x_test, y_test)
    print(f"Accuracy sur le test set : {accuracy_test:.4f}")

    mlflow.log_param("max_iter", 200)
    mlflow.log_param("model_type", "logistic_regression")

    mlflow.log_metric("accuracy_train", accuracy_train)
    mlflow.log_metric("accuracy_test", accuracy_test)

    # Save model
    model_path = os.path.join(BASE_DIR, "models", "model.pkl")
    joblib.dump(pipeline, model_path)
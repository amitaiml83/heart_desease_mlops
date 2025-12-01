import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
import os
from google.cloud import storage


# -------------------------------
# GCS UPLOAD FUNCTION
# -------------------------------
def upload_to_gcs(local_path, bucket_name, blob_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded to GCS → gs://{bucket_name}/{blob_path}")


# -------------------------------
# LOAD & CLEAN DATA
# -------------------------------
def load_data(path):
    df = pd.read_csv(path)

    if "sno" in df.columns:
        df.drop("sno", axis=1, inplace=True)

    df['gender'] = df['gender'].apply(lambda x: 1 if x == "male" else 0)
    df = df[df['chol'] <= 400]

    imputer = SimpleImputer(strategy="median")
    cols = ['trestbps', 'thalach']
    df[cols] = imputer.fit_transform(df[cols])

    df['target'] = df['target'].apply(lambda x: 1 if x == "yes" else 0)

    return df


# -------------------------------
# TRAIN MODEL & SAVE OUTPUTS
# -------------------------------
def train_model():
    df = load_data("data.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(solver="liblinear", max_iter=300)
    model.fit(X_train, y_train)

    print("Train Score:", model.score(X_train, y_train))
    print("Test Score:", model.score(X_test, y_test))

    y_pred = model.predict(X_test)

    # Create output folder
    os.makedirs("models", exist_ok=True)

    # -------------------------------
    # SAVE MODEL LOCALLY
    # -------------------------------
    model_path = "model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

    # -------------------------------
    # NEW: SAVE MODEL WEIGHTS AS .h5
    # -------------------------------
    weights_path = "best_model.h5"
    joblib.dump(model, weights_path)
    print(f"Model weights saved at {weights_path}")

    # -------------------------------
    # SAVE CLASSIFICATION REPORT CSV
    # -------------------------------
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = "classification_report.csv"
    report_df.to_csv(report_path, index=True)
    print(f"Classification report saved at {report_path}")

    # -------------------------------
    # SAVE METRICS CSV
    # -------------------------------
    metrics = {
        "accuracy": [accuracy_score(y_test, y_pred)],
        "train_score": [model.score(X_train, y_train)],
        "test_score": [model.score(X_test, y_test)]
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_path = "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved at {metrics_path}")

    # -------------------------------
    # UPLOAD ALL FILES TO GCS
    # -------------------------------
    bucket = "mlops_datasets_am"
    base_path = "heart_disease_detection_ds"

    upload_to_gcs(model_path, bucket, f"{base_path}/model.pkl")
    upload_to_gcs(weights_path, bucket, f"{base_path}/best_model.h5")
    upload_to_gcs(report_path, bucket, f"{base_path}/classification_report.csv")
    upload_to_gcs(metrics_path, bucket, f"{base_path}/metrics.csv")

    print("\nAll artifacts successfully uploaded to GCS.")


if __name__ == "__main__":
    train_model()

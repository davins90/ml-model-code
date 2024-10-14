import argparse
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from google.cloud import storage
from google.cloud import monitoring_v3
from google.cloud import aiplatform
import time
import os

def export_accuracy_metric(accuracy):
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{os.environ['GOOGLE_CLOUD_PROJECT']}"

    series = monitoring_v3.TimeSeries()
    series.metric.type = 'custom.googleapis.com/ml_model/accuracy'
    series.resource.type = 'global'
    point = series.points.add()
    point.value.double_value = accuracy
    now = time.time()
    point.interval.end_time.seconds = int(now)
    point.interval.end_time.nanos = int(
        (now - point.interval.end_time.seconds) * 10**9)

    client.create_time_series({"name": project_name, "time_series": [series]})
    print("Accuracy metric exported to Cloud Monitoring.")

def upload_model_to_registry(model_path):
    aiplatform.init(project=os.environ['GOOGLE_CLOUD_PROJECT'], location='us-central1')

    model = aiplatform.Model.upload(
        display_name="iris_classification_model",
        artifact_uri=model_path,
        serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest',
    )
    print(f"Model registered with ID: {model.resource_name}")

def main(bucket_name):
    # Carica il dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Dividi il dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Crea e addestra il modello
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Valuta il modello
    accuracy = clf.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")

    # Esporta la metrica
    export_accuracy_metric(accuracy)

    # Salva il modello
    joblib.dump(clf, 'model.joblib')

    # Carica il modello su Cloud Storage
    destination_blob_name = 'models/iris/model.joblib'

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename('model.joblib')

    print(f"Model uploaded to gs://{bucket_name}/{destination_blob_name}")

    # Registra il modello nel Model Registry
    upload_model_to_registry(f'gs://{bucket_name}/models/iris/')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name', type=str, required=True, help='GCS bucket name to store the model')
    args = parser.parse_args()

    # Assicurati di impostare la variabile d'ambiente GOOGLE_CLOUD_PROJECT
    os.environ['GOOGLE_CLOUD_PROJECT'] = 'test-mlops-pipeline-438622'  # Sostituisci con il tuo project ID

    main(args.bucket_name)

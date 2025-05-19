#!/bin/bash

# Start MLflow tracking server with file store in shared-data
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri /shared-data/mlruns \
    --default-artifact-root /shared-data/mlruns
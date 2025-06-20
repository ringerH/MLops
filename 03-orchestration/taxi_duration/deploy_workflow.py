#!/usr/bin/env python
# coding: utf-8

from prefect import serve
from main_workflow import taxi_duration_prediction_flow

if __name__ == "__main__":
    # Create a deployment for the workflow
    deployment = taxi_duration_prediction_flow.to_deployment(
        name="taxi-duration-prediction-deployment",
        version="1.0.0",
        description="Taxi duration prediction model training workflow",
        tags=["ml", "taxi", "xgboost", "mlflow"],
        parameters={
            "year": 2023,
            "month": 1,
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "nyc-taxi-experiment-prefect"
        }
    )
    
    # Serve the deployment
    serve(deployment)
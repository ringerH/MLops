#!/usr/bin/env python
# coding: utf-8

from prefect import flow, get_run_logger
from pathlib import Path

# Import our custom tasks
from data_processing import read_dataframe, get_validation_data_params
from feature_engineering import create_features, extract_target, save_preprocessor
from model_training import setup_mlflow, get_model_params, train_xgboost_model, log_preprocessor_to_mlflow


@flow(name="taxi-duration-prediction")
def taxi_duration_prediction_flow(
    year: int, 
    month: int,
    tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "nyc-taxi-experiment-prefect"
):
    """
    Main workflow for taxi duration prediction using Prefect orchestration.
    
    Args:
        year: Year of the training data
        month: Month of the training data
        tracking_uri: MLflow tracking URI
        experiment_name: MLflow experiment name
    
    Returns:
        run_id: MLflow run ID
    """
    logger = get_run_logger()
    logger.info(f"Starting taxi duration prediction workflow for {year}-{month:02d}")
    
    # Create models directory
    models_folder = Path('models')
    models_folder.mkdir(exist_ok=True)
    
    # Setup MLflow
    setup_mlflow(tracking_uri, experiment_name)
    
    # Get validation data parameters
    val_year, val_month = get_validation_data_params(year, month)
    logger.info(f"Validation data: {val_year}-{val_month:02d}")
    
    # Load training and validation data
    df_train = read_dataframe(year, month)
    df_val = read_dataframe(val_year, val_month)
    
    # Create features
    X_train, dv = create_features(df_train)
    X_val, _ = create_features(df_val, dv)
    
    # Extract target variables
    y_train = extract_target(df_train)
    y_val = extract_target(df_val)
    
    # Save preprocessor
    preprocessor_path = save_preprocessor(dv)
    
    # Get model parameters
    params = get_model_params()
    
    # Train model and get run ID
    run_id = train_xgboost_model(X_train, y_train, X_val, y_val, params)
    
    # Log preprocessor to MLflow (this needs to be done after training starts)
    log_preprocessor_to_mlflow(preprocessor_path)
    
    logger.info(f"Workflow completed successfully. MLflow run ID: {run_id}")
    return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration using Prefect.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    parser.add_argument('--tracking-uri', type=str, default="http://localhost:5000", help='MLflow tracking URI')
    parser.add_argument('--experiment-name', type=str, default="nyc-taxi-experiment-prefect", help='MLflow experiment name')
    
    args = parser.parse_args()

    # Run the workflow
    run_id = taxi_duration_prediction_flow(
        year=args.year,
        month=args.month,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name
    )

    # Save run ID to file
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    
    print(f"Workflow completed. MLflow run_id: {run_id}")
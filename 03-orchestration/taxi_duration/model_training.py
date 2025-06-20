#!/usr/bin/env python
# coding: utf-8

import xgboost as xgb
import mlflow
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from prefect import task, get_run_logger
from typing import Dict, Any


@task
def setup_mlflow(tracking_uri: str = "http://localhost:5000", experiment_name: str = "nyc-taxi-experiment-prefect"):
    """Setup MLflow tracking."""
    logger = get_run_logger()
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    logger.info(f"MLflow tracking URI set to: {tracking_uri}")
    logger.info(f"MLflow experiment set to: {experiment_name}")


@task
def get_model_params() -> Dict[str, Any]:
    """Get the best hyperparameters for the XGBoost model."""
    return {
        'learning_rate': 0.09585355369315604,
        'max_depth': 30,
        'min_child_weight': 1.060597050922164,
        'objective': 'reg:linear',
        'reg_alpha': 0.018060244040060163,
        'reg_lambda': 0.011658731377413597,
        'seed': 42
    }


@task
def train_xgboost_model(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray,
    params: Dict[str, Any],
    num_boost_round: int = 30,
    early_stopping_rounds: int = 50
) -> str:
    """Train XGBoost model and log to MLflow."""
    logger = get_run_logger()
    
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params(params)
        logger.info(f"Logged parameters: {params}")
        
        # Create DMatrix objects for XGBoost
        train_dmatrix = xgb.DMatrix(X_train, label=y_train)
        valid_dmatrix = xgb.DMatrix(X_val, label=y_val)
        
        logger.info("Training XGBoost model...")
        
        # Train the model
        booster = xgb.train(
            params=params,
            dtrain=train_dmatrix,
            num_boost_round=num_boost_round,
            evals=[(valid_dmatrix, 'validation')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )
        
        # Make predictions and calculate RMSE
        y_pred = booster.predict(valid_dmatrix)
        rmse = root_mean_squared_error(y_val, y_pred)
        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        logger.info(f"Validation RMSE: {rmse}")
        
        # Log model
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
        logger.info("Model logged to MLflow")
        
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")
        
        return run_id


@task
def log_preprocessor_to_mlflow(preprocessor_path: str):
    """Log the preprocessor artifact to MLflow."""
    logger = get_run_logger()
    
    # Get the current active run
    run = mlflow.active_run()
    if run is None:
        logger.error("No active MLflow run found")
        return
    
    mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")
    logger.info(f"Preprocessor logged to MLflow from: {preprocessor_path}")
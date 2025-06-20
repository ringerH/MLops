#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction import DictVectorizer
from prefect import task, get_run_logger
from typing import Tuple, Optional
import numpy as np


@task
def create_features(df: pd.DataFrame, dv: Optional[DictVectorizer] = None) -> Tuple[np.ndarray, DictVectorizer]:
    """Create feature matrix from dataframe using DictVectorizer."""
    logger = get_run_logger()
    
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    
    logger.info(f"Creating features with categorical: {categorical}, numerical: {numerical}")
    
    # Convert to dictionary format for DictVectorizer
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    if dv is None:
        logger.info("Fitting new DictVectorizer")
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        logger.info("Using existing DictVectorizer")
        X = dv.transform(dicts)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    return X, dv


@task
def extract_target(df: pd.DataFrame, target_column: str = 'duration') -> np.ndarray:
    """Extract target variable from dataframe."""
    logger = get_run_logger()
    
    y = df[target_column].values
    logger.info(f"Target shape: {y.shape}")
    
    return y


@task
def save_preprocessor(dv: DictVectorizer, models_folder: str = "models") -> str:
    """Save the fitted DictVectorizer to disk."""
    logger = get_run_logger()
    
    # Create models folder if it doesn't exist
    Path(models_folder).mkdir(exist_ok=True)
    
    preprocessor_path = f"{models_folder}/preprocessor.b"
    
    with open(preprocessor_path, "wb") as f_out:
        pickle.dump(dv, f_out)
    
    logger.info(f"Preprocessor saved to: {preprocessor_path}")
    return preprocessor_path
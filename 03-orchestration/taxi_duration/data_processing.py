#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from prefect import task, get_run_logger
from typing import Tuple


@task
def read_dataframe(year: int, month: int) -> pd.DataFrame:
    """Read and preprocess taxi data for a given year and month."""
    logger = get_run_logger()
    
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    logger.info(f"Reading data from: {url}")
    
    df = pd.read_parquet(url)
    logger.info(f"Raw data shape: {df.shape}")

    # Calculate trip duration
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    # Filter data based on duration
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    logger.info(f"Filtered data shape: {df.shape}")

    # Convert categorical columns to string
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    # Create pickup-dropoff location combination
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    logger.info("Data preprocessing completed")
    return df


@task
def get_validation_data_params(year: int, month: int) -> Tuple[int, int]:
    """Calculate the year and month for validation data (next month)."""
    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    return next_year, next_month
# Taxi Duration Prediction with Prefect Orchestration

This project implements a machine learning workflow for predicting NYC taxi trip durations using Prefect for orchestration and MLflow for experiment tracking.

## Project Structure

```
├── data_processing.py          # Data loading and preprocessing tasks
├── feature_engineering.py      # Feature creation and target extraction tasks
├── model_training.py           # Model training and MLflow logging tasks
├── main_workflow.py            # Main Prefect workflow
├── deploy_workflow.py          # Deployment script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Features

- **Modular Design**: Code is split into logical components using Prefect tasks
- **Data Processing**: Automated data loading and preprocessing
- **Feature Engineering**: Automated feature creation using DictVectorizer
- **Model Training**: XGBoost model training with hyperparameter logging
- **MLflow Integration**: Automatic experiment tracking and model logging
- **Workflow Orchestration**: Complete pipeline orchestration using Prefect

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start MLflow tracking server:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

3. Start Prefect server (in a separate terminal):
```bash
prefect server start
```

## Usage

### Running the Workflow Directly

```bash
python main_workflow.py --year 2023 --month 1
```

### Creating and Running a Deployment

1. Create a deployment:
```bash
python deploy_workflow.py
```

2. In another terminal, start a worker:
```bash
prefect worker start --pool default-agent-pool
```

3. Run the deployment through the Prefect UI or CLI:
```bash
prefect deployment run taxi-duration-prediction/taxi-duration-prediction-deployment
```

## Workflow Steps

1. **Setup**: Initialize MLflow tracking and create necessary directories
2. **Data Loading**: Load training and validation datasets from remote parquet files
3. **Data Preprocessing**: Clean data, calculate trip duration, and create categorical features
4. **Feature Engineering**: Create feature matrices using DictVectorizer
5. **Model Training**: Train XGBoost model with optimal hyperparameters
6. **Model Logging**: Log model, metrics, and preprocessor to MLflow
7. **Artifact Storage**: Save run ID for downstream processes

## Key Improvements Over Original Script

- **Better Error Handling**: Each task can be retried independently
- **Logging**: Comprehensive logging at each step
- **Modularity**: Easy to modify individual components
- **Orchestration**: Visual workflow monitoring through Prefect UI
- **Scalability**: Can be easily deployed to cloud infrastructure
- **Dependency Management**: Clear task dependencies and data flow

## Monitoring

- **Prefect UI**: Visit `http://localhost:4200` to monitor workflow runs
- **MLflow UI**: Visit `http://localhost:5000` to view experiment results

## Configuration

You can customize the workflow by modifying parameters in:
- `main_workflow.py`: Default parameters
- `deploy_workflow.py`: Deployment parameters
- `model_training.py`: Model hyperparameters

## Next Steps

Consider adding:
- Data validation tasks
- Model evaluation and comparison
- Automated model deployment
- Scheduled retraining
- Data drift monitoring
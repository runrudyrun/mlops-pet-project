# Requirements Document

## Introduction

The "End-to-End MLOps Pipeline" project is a comprehensive machine learning pipeline for wine quality classification. The project demonstrates modern MLOps practices using a free technology stack: GitHub Actions for CI/CD, DVC for data versioning, MLflow + DagsHub for experiment tracking, FastAPI for model serving, and Docker for containerization.

The main goal is to create a reproducible, automated, and production-ready ML pipeline that can serve as a reference project for demonstrating MLOps skills.

## Requirements

### Requirement 1: Environment Setup and Data Versioning (DVC)

**User Story:** As a ML engineer, I want to set up data versioning with DVC, so that data is separated from code and can be reproduced by any team member.

#### Acceptance Criteria

1. WHEN the project is initialized THEN the system SHALL contain a correct directory structure (data/, src/, models/, configs/)
2. WHEN DVC is initialized THEN the system SHALL create .dvc/ directory and .dvcignore file
3. WHEN wine_quality.csv dataset is added to DVC THEN the system SHALL create data/wine_quality.csv.dvc file with metadata
4. WHEN remote storage is configured THEN the system SHALL have DVC remote configuration in .dvc/config
5. WHEN user executes `dvc pull` THEN the system SHALL download data from remote storage

### Requirement 2: Experiment Tracking with MLflow

**User Story:** As a ML engineer, I want to track experiments with MLflow, so that all hyperparameters and metrics are saved and can be compared.

#### Acceptance Criteria

1. WHEN train.py script is executed THEN the system SHALL log model hyperparameters via mlflow.log_param()
2. WHEN training is complete THEN the system SHALL log metrics (accuracy, f1-score, precision, recall) via mlflow.log_metric()
3. WHEN model is trained THEN the system SHALL save model artifact via mlflow.log_artifact() or mlflow.sklearn.log_model()
4. WHEN experiment is complete THEN results SHALL be available in MLflow/DagsHub web interface
5. IF MLflow tracking URI is configured THEN the system SHALL send data to remote DagsHub server

### Requirement 3: DVC Pipeline for Reproducibility

**User Story:** As a ML engineer, I want to create a reproducible pipeline with DVC, so that any user can reproduce the entire training process with `dvc repro` command.

#### Acceptance Criteria

1. WHEN dvc.yaml is created THEN the system SHALL contain stages: prepare, train, evaluate
2. WHEN prepare stage is executed THEN the system SHALL split data into train/test sets
3. WHEN train stage is executed THEN the system SHALL train the model and save it to models/
4. WHEN evaluate stage is executed THEN the system SHALL compute metrics and save them to metrics.json
5. WHEN user executes `dvc repro` THEN the system SHALL execute all stages in correct order
6. WHEN stage dependencies have not changed THEN the system SHALL skip execution of that stage (caching)

### Requirement 4: CI/CD with GitHub Actions and CML

**User Story:** As a ML engineer, I want to automate training and reporting through CI/CD, so that on each Pull Request the model is retrained and results are published automatically.

#### Acceptance Criteria

1. WHEN Pull Request is created THEN GitHub Actions SHALL trigger workflow from .github/workflows/cml.yaml
2. WHEN workflow is triggered THEN the system SHALL execute `dvc repro` to train the model
3. WHEN training is complete THEN CML SHALL generate a report with model metrics
4. WHEN report is ready THEN CML SHALL publish a comment to the Pull Request with results
5. IF metrics improved compared to main branch THEN the report SHALL contain information about improvement
6. WHEN workflow completes successfully THEN the system SHALL have "passed" status in GitHub

### Requirement 5: Model Serving API (FastAPI)

**User Story:** As a developer, I want to have a REST API for getting predictions, so that the model can be integrated into other applications.

#### Acceptance Criteria

1. WHEN FastAPI application is running THEN the system SHALL provide POST /predict endpoint
2. WHEN POST /predict receives JSON with wine features THEN the system SHALL return quality prediction
3. WHEN request contains invalid data THEN the system SHALL return HTTP 422 with error description
4. WHEN application is running THEN the system SHALL provide Swagger UI documentation at /docs
5. WHEN /health endpoint is called THEN the system SHALL return service health status

### Requirement 6: Containerization with Docker

**User Story:** As a DevOps engineer, I want to package the application in a Docker container, so that deployment is consistent across any environment.

#### Acceptance Criteria

1. WHEN Dockerfile is created THEN the system SHALL contain all dependencies to run the API
2. WHEN `docker build` is executed THEN the system SHALL create an image without errors
3. WHEN `docker run` is executed THEN the container SHALL start FastAPI application on specified port
4. WHEN container is running THEN API SHALL be accessible and respond to requests
5. IF model is not found in container THEN the system SHALL handle the error gracefully at startup

### Requirement 7: Configuration and Parameterization

**User Story:** As a ML engineer, I want to manage parameters through configuration files, so that experiments are easy to reproduce and modify.

#### Acceptance Criteria

1. WHEN params.yaml is created THEN the system SHALL contain all model hyperparameters
2. WHEN params.yaml is modified THEN DVC SHALL detect the change and rebuild dependent stages
3. WHEN scripts are executed THEN the system SHALL read parameters from params.yaml
4. IF parameter is not specified in config THEN the system SHALL use default value

### Requirement 8: Model Monitoring and Data Drift Detection

**User Story:** As a ML engineer, I want to track data drift and model degradation, so that I can retrain the model in time when data distribution changes.

#### Acceptance Criteria

1. WHEN new data arrives THEN the system SHALL compute feature distribution statistics
2. WHEN data drift is detected THEN the system SHALL generate a report indicating affected features
3. WHEN monitoring script is executed THEN the system SHALL compare current distribution with baseline (training data)
4. WHEN drift exceeds threshold value THEN the system SHALL log a warning
5. WHEN /monitoring/drift endpoint is called THEN API SHALL return current drift status and metrics
6. IF Evidently AI is used THEN the system SHALL generate HTML reports on data quality

### Requirement 9: Extended CI/CD (Tests and Linting)

**User Story:** As a developer, I want to have automatic code quality checks and tests, so that code meets standards and works correctly.

#### Acceptance Criteria

1. WHEN Pull Request is created THEN GitHub Actions SHALL run code linting (flake8/ruff)
2. WHEN Pull Request is created THEN GitHub Actions SHALL run formatting checks (black/isort)
3. WHEN Pull Request is created THEN GitHub Actions SHALL run unit tests (pytest)
4. WHEN tests fail THEN workflow SHALL fail and block merge
5. WHEN linting detects issues THEN workflow SHALL output list of violations
6. WHEN all checks pass THEN PR SHALL receive "All checks passed" status
7. IF pre-commit hooks are configured THEN the system SHALL check code locally before commit

### Requirement 10: Multiple Model Support for Comparison

**User Story:** As a ML engineer, I want to train and compare multiple models, so that I can choose the best model for production.

#### Acceptance Criteria

1. WHEN params.yaml contains a list of models THEN the system SHALL train each model
2. WHEN multiple models are trained THEN the system SHALL save metrics for each in MLflow
3. WHEN experiments are complete THEN the system SHALL provide a comparative metrics table
4. WHEN best model is selected THEN the system SHALL tag it as "production" in MLflow Model Registry
5. WHEN API is started THEN the system SHALL load the model with "production" tag
6. IF model is not explicitly specified THEN the system SHALL use default model (RandomForest)
7. WHEN new model is added to config THEN the system SHALL automatically include it in training pipeline

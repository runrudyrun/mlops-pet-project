# Implementation Plan

## Phase 1: Project Setup and Infrastructure

- [x] 1. Initialize project structure and dependencies
  - [x] 1.1 Create directory structure (src/, data/, models/, configs/, tests/, reports/, .github/workflows/)
    - Create all directories as defined in design document
    - Add `__init__.py` files to Python packages
    - _Requirements: 1.1_

  - [x] 1.2 Create pyproject.toml with project configuration
    - Define project metadata and dependencies
    - Configure ruff, black, isort, pytest settings
    - _Requirements: 9.1, 9.2_

  - [x] 1.3 Create requirements.txt and requirements-dev.txt
    - Production deps: pandas, scikit-learn, mlflow, fastapi, uvicorn, evidently, pyyaml, python-dotenv
    - Dev deps: pytest, pytest-cov, ruff, black, isort, httpx, pre-commit
    - _Requirements: 1.1, 9.7_

  - [x] 1.4 Create .gitignore and .dvcignore files
    - Ignore Python artifacts, virtual environments, IDE files
    - DVC ignore patterns for temporary files
    - _Requirements: 1.2_

  - [x] 1.5 Create .pre-commit-config.yaml
    - Configure hooks for ruff, black, isort
    - Add trailing whitespace and end-of-file fixers
    - _Requirements: 9.7_

- [x] 2. Initialize DVC and configure remote storage
  - [x] 2.1 Initialize DVC in the repository
    - Run `dvc init` to create .dvc/ directory
    - _Requirements: 1.2_

  - [x] 2.2 Create placeholder for DVC remote configuration
    - Add template .dvc/config with DagsHub remote placeholder
    - Document setup instructions in README
    - _Requirements: 1.4_

## Phase 2: Configuration and Data Layer

- [x] 3. Create configuration system
  - [x] 3.1 Create configs/params.yaml with all hyperparameters
    - Define data paths and split parameters
    - Define model configurations for RandomForest, GradientBoosting, LogisticRegression
    - Define evaluation and monitoring settings
    - _Requirements: 7.1, 10.1_

  - [x] 3.2 Create config loader utility (src/config.py)
    - Implement YAML loading with environment variable substitution
    - Add default value handling
    - _Requirements: 7.3, 7.4_

  - [x] 3.3 Write unit tests for config loader
    - Test YAML parsing, defaults, environment variable substitution
    - _Requirements: 9.3_

- [x] 4. Implement data preparation module
  - [x] 4.1 Create src/data/prepare.py with data loading and preprocessing
    - Implement load_data() function to read CSV
    - Implement preprocess_data() for cleaning (handle missing values)
    - Implement split_data() for train/test split
    - _Requirements: 3.2_

  - [x] 4.2 Add main() entry point for DVC stage
    - Read config from params.yaml
    - Save processed data to data/processed/
    - _Requirements: 3.2_

  - [x] 4.3 Write unit tests for data preparation
    - Test data loading, preprocessing, splitting
    - Test edge cases (empty data, missing columns)
    - _Requirements: 9.3_

  - [x] 4.4 Download wine quality dataset and add to DVC
    - Download wine_quality.csv to data/raw/
    - Run `dvc add data/raw/wine_quality.csv`
    - _Requirements: 1.3_

## Phase 3: Model Training and Evaluation

- [x] 5. Implement model training module
  - [x] 5.1 Create src/models/train.py with model factory
    - Implement get_model() factory function for different model types
    - Support RandomForest, GradientBoosting, LogisticRegression
    - _Requirements: 10.6, 10.7_

  - [x] 5.2 Implement training logic with MLflow integration
    - Implement train_model() with parameter logging
    - Implement train_all_models() to train enabled models from config
    - Log parameters, metrics, and model artifacts to MLflow
    - _Requirements: 2.1, 2.2, 2.3, 10.2_

  - [x] 5.3 Add main() entry point for DVC stage
    - Load training data and config
    - Train all enabled models
    - Save best model to models/
    - _Requirements: 3.3_

  - [x] 5.4 Write unit tests for training module
    - Test model factory with different model types
    - Test training with mock data
    - Test MLflow logging (mock MLflow client)
    - _Requirements: 9.3_

- [x] 6. Implement model evaluation module
  - [x] 6.1 Create src/models/evaluate.py with metrics computation
    - Implement compute_metrics() for accuracy, precision, recall, f1
    - Implement evaluate_model() to run predictions and compute metrics
    - _Requirements: 2.2_

  - [x] 6.2 Implement model selection logic
    - Implement select_best_model() based on primary metric
    - Log comparison to MLflow
    - _Requirements: 10.3, 10.4_

  - [x] 6.3 Add main() entry point for DVC stage
    - Load test data and trained models
    - Evaluate all models and save metrics to reports/metrics.json
    - _Requirements: 3.4_

  - [x] 6.4 Write unit tests for evaluation module
    - Test metrics computation with known values
    - Test model selection logic
    - _Requirements: 9.3_

## Phase 4: DVC Pipeline

- [x] 7. Create DVC pipeline definition
  - [x] 7.1 Create dvc.yaml with prepare, train, evaluate stages
    - Define dependencies, parameters, outputs for each stage
    - Configure metrics and plots outputs
    - _Requirements: 3.1, 3.5, 3.6_

  - [x] 7.2 Test pipeline execution with `dvc repro`
    - Verify all stages execute in correct order
    - Verify caching works when dependencies unchanged
    - _Requirements: 3.5, 3.6_

## Phase 5: Data Drift Monitoring

- [-] 8. Implement drift detection module
  - [x] 8.1 Create src/monitoring/drift.py with Evidently integration
    - Implement load_reference_data() to load training data baseline
    - Implement compute_drift_report() using Evidently DataDriftPreset
    - _Requirements: 8.1, 8.3_

  - [x] 8.2 Implement drift threshold checking
    - Implement check_drift_threshold() to compare against configured threshold
    - Implement get_drift_summary() for API response
    - _Requirements: 8.4, 8.5_

  - [x] 8.3 Add HTML report generation
    - Implement save_drift_report() to generate Evidently HTML report
    - _Requirements: 8.2, 8.6_

  - [ ] 8.4 Write unit tests for drift detection
    - Test drift computation with synthetic data
    - Test threshold checking logic
    - _Requirements: 9.3_

## Phase 6: FastAPI Application

- [ ] 9. Create API schemas
  - [ ] 9.1 Create src/api/schemas.py with Pydantic models
    - Define WineFeatures, PredictionRequest, PredictionResponse
    - Define BatchPredictionRequest, BatchPredictionResponse
    - Define HealthResponse, DriftResponse
    - _Requirements: 5.2, 5.3_

  - [ ] 9.2 Write unit tests for schema validation
    - Test valid and invalid input handling
    - Test serialization/deserialization
    - _Requirements: 9.3_

- [ ] 10. Implement FastAPI application
  - [ ] 10.1 Create src/api/app.py with application setup
    - Initialize FastAPI app with metadata
    - Implement model loading on startup
    - Configure CORS and error handlers
    - _Requirements: 5.1, 5.4_

  - [ ] 10.2 Implement /health endpoint
    - Return service status and model loaded state
    - _Requirements: 5.5_

  - [ ] 10.3 Implement /predict endpoint
    - Accept WineFeatures, return quality prediction with confidence
    - Handle validation errors with proper HTTP codes
    - _Requirements: 5.2, 5.3_

  - [ ] 10.4 Implement /predict/batch endpoint
    - Accept list of WineFeatures, return batch predictions
    - _Requirements: 5.2_

  - [ ] 10.5 Implement /monitoring/drift endpoint
    - Return current drift status from monitoring module
    - _Requirements: 8.5_

  - [ ] 10.6 Write integration tests for API endpoints
    - Test all endpoints with valid and invalid inputs
    - Test error handling and response codes
    - _Requirements: 9.3_

## Phase 7: Containerization

- [ ] 11. Create Docker configuration
  - [ ] 11.1 Create Dockerfile for API service
    - Use python:3.11-slim base image
    - Install dependencies and copy application code
    - Configure health check and entrypoint
    - _Requirements: 6.1, 6.2_

  - [ ] 11.2 Create docker-compose.yaml
    - Define API service with port mapping and volumes
    - Configure environment variables
    - _Requirements: 6.3, 6.4_

  - [ ] 11.3 Test Docker build and run
    - Verify image builds without errors
    - Verify container starts and API responds
    - _Requirements: 6.2, 6.3, 6.4_

## Phase 8: CI/CD Workflows

- [ ] 12. Create GitHub Actions workflows
  - [ ] 12.1 Create .github/workflows/ci.yaml for linting and tests
    - Configure ruff, black, isort checks
    - Configure pytest with coverage reporting
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [ ] 12.2 Create .github/workflows/cml.yaml for ML pipeline
    - Configure DVC and CML setup
    - Run `dvc repro` on PR
    - Generate and post CML report with metrics
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

## Phase 9: Documentation and Final Setup

- [ ] 13. Create project documentation
  - [ ] 13.1 Create comprehensive README.md
    - Project overview and features
    - Installation and setup instructions
    - Usage examples for training, API, Docker
    - DagsHub and MLflow configuration guide
    - _Requirements: 1.1_

  - [ ] 13.2 Create .env.example with required environment variables
    - MLFLOW_TRACKING_URI, DAGSHUB_USERNAME, DAGSHUB_TOKEN
    - _Requirements: 2.5_

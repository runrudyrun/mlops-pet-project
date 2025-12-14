# End-to-End MLOps Pipeline

A comprehensive machine learning pipeline for wine quality classification demonstrating modern MLOps practices.

## Features

- **Data Versioning**: DVC for tracking datasets and model artifacts
- **Experiment Tracking**: MLflow + DagsHub for logging parameters, metrics, and models
- **Reproducible Pipelines**: DVC pipelines for automated ML workflows
- **CI/CD**: GitHub Actions + CML for automated testing and reporting
- **Model Serving**: FastAPI REST API for predictions
- **Containerization**: Docker for consistent deployment
- **Monitoring**: Evidently AI for data drift detection
- **Code Quality**: pytest, ruff, black for testing and linting

## Project Structure

```
mlops-pipeline/
├── .github/workflows/     # CI/CD workflows
├── .dvc/                  # DVC configuration
├── configs/               # Hyperparameters and model configs
├── data/
│   ├── raw/              # Raw datasets (DVC-tracked)
│   └── processed/        # Processed datasets
├── models/               # Trained model artifacts
├── reports/              # Metrics and drift reports
├── src/
│   ├── api/              # FastAPI application
│   ├── data/             # Data preparation
│   ├── models/           # Training and evaluation
│   └── monitoring/       # Drift detection
└── tests/                # Unit and integration tests
```

## Installation

### Prerequisites

- Python 3.10+
- Git
- Docker (optional, for containerization)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mlops-pipeline
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

## DVC and DagsHub Setup

This project uses [DVC](https://dvc.org/) for data versioning with [DagsHub](https://dagshub.com/) as the remote storage.

### Setting Up DagsHub

1. **Create a DagsHub account** at [dagshub.com](https://dagshub.com/)

2. **Create a new repository** on DagsHub or connect an existing GitHub repository

3. **Get your DagsHub credentials**
   - Go to your DagsHub profile settings
   - Navigate to "Tokens" section
   - Create a new token or use an existing one

4. **Configure DVC remote**
   
   Update the `.dvc/config` file with your DagsHub details:
   ```ini
   [core]
       remote = dagshub
       autostage = true

   ['remote "dagshub"']
       url = https://dagshub.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.dvc
   ```

   Or use the DVC CLI:
   ```bash
   dvc remote modify dagshub url https://dagshub.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.dvc
   ```

5. **Set up authentication**
   
   Create a `.env` file (copy from `.env.example`):
   ```bash
   cp .env.example .env
   ```
   
   Add your credentials:
   ```
   DAGSHUB_USERNAME=your_username
   DAGSHUB_TOKEN=your_token
   MLFLOW_TRACKING_URI=https://dagshub.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.mlflow
   ```

   For DVC authentication, you can either:
   
   **Option A**: Use environment variables
   ```bash
   export DVC_REMOTE_DAGSHUB_AUTH_BASIC_USERNAME=<YOUR_USERNAME>
   export DVC_REMOTE_DAGSHUB_AUTH_BASIC_PASSWORD=<YOUR_TOKEN>
   ```
   
   **Option B**: Configure credentials locally (not committed to git)
   ```bash
   dvc remote modify --local dagshub auth basic
   dvc remote modify --local dagshub user <YOUR_USERNAME>
   dvc remote modify --local dagshub password <YOUR_TOKEN>
   ```

### Pulling Data

Once configured, pull the data:
```bash
dvc pull
```

### Pushing Data

After adding new data or models:
```bash
dvc add data/raw/wine_quality.csv
dvc push
git add data/raw/wine_quality.csv.dvc .gitignore
git commit -m "Add wine quality dataset"
```

## Usage

### Running the ML Pipeline

Execute the full pipeline:
```bash
dvc repro
```

Run individual stages:
```bash
dvc repro prepare   # Data preparation
dvc repro train     # Model training
dvc repro evaluate  # Model evaluation
```

### Starting the API

```bash
uvicorn src.api.app:app --reload
```

Access the API documentation at `http://localhost:8000/docs`

### Running Tests

```bash
pytest                      # Run all tests
pytest --cov=src           # With coverage
pytest tests/test_api/     # Run specific tests
```

### Code Quality

```bash
ruff check .               # Linting
black .                    # Formatting
isort .                    # Import sorting
```

## Docker

Build and run with Docker:
```bash
docker build -t mlops-pipeline .
docker run -p 8000:8000 mlops-pipeline
```

Or use Docker Compose:
```bash
docker-compose up
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MLFLOW_TRACKING_URI` | MLflow/DagsHub tracking server URL |
| `DAGSHUB_USERNAME` | DagsHub username |
| `DAGSHUB_TOKEN` | DagsHub access token |
| `MODEL_PATH` | Path to trained model (default: `models/model.pkl`) |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

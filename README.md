# Thyroid Recurrence Predictor

Machine learning system for predicting thyroid cancer recurrence with containerized API and automated CI/CD pipeline.

## Overview

This project builds a machine learning model to predict thyroid cancer recurrence, packages it into a containerized FastAPI application, and deploys it using GitHub Actions and AWS EC2.

## Project Structure

```
├── recurrence_model/         # Model training and prediction code
│   ├── train_pipeline.py     # Training script
│   ├── predict.py            # Prediction logic
│   ├── processing/           # Data preprocessing modules
│   └── trained_models/       # Saved model artifacts
├── recurrence_api/           # API implementation
│   ├── app/                  # FastAPI application
│   └── requirements.txt      # API dependencies
├── requirements/             # Project requirements
├── tests/                    # Test suite
├── Dockerfile                # Container definition
├── .github/workflows/        # CI/CD pipeline definitions
└── setup.py                  # Package configuration
```

## Machine Learning Pipeline

The model predicts thyroid cancer recurrence using Logistic Regression from scikit-learn library. Training pipeline:
- Data loading and validation
- Feature engineering
- Model training and evaluation
- Model persistence

Train the model locally:

```bash
pip install -r requirements/requirements.txt
python recurrence_model/train_pipeline.py
```

## FastAPI Application

The API provides:
- /health - Endpoint to check API status
- /predict - Endpoint for making predictions

API implementation leverages:
- Pydantic for input validation
- FastAPI for efficient API creation
- Joblib for model loading

## Docker Containerization

The application is containerized for consistent deployment:

```dockerfile
FROM python:3.10
ADD . .
WORKDIR /recurrence_api
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8001
CMD ["python", "app/main.py"]
```

## CI/CD Pipeline

### CI Pipeline

The CI workflow automates:
1. Train: Trains the model and uploads artifacts
2. Test: Runs test suite against trained model
3. Build: Packages the model and API
4. Push: Creates and pushes Docker image to DockerHub

```yaml
name: Push Thyroid Recurrence Docker Image

on:
  push:
    branches:
      - main
  workflow_dispatch:

jobs:
  train:
    # Training job
  test:
    # Testing job
  build:
    # Build job
  push-image:
    # Docker push job
```

### CD Pipeline

The CD workflow:
1. Pulls the latest Docker image
2. Removes any existing containers
3. Deploys the new container

```yaml
name: CD Pipeline

on:
  workflow_run:
    workflows: ["Push Thyroid Recurrence Docker Image"]
    types:
      - completed

jobs:
  deploy:
    runs-on: self-hosted
    steps:
      # Deployment steps
```

## Setup and Deployment

### DockerHub Setup

1. Create a DockerHub account
2. Create a repository named thyroid-recurrence-api
3. Generate an access token
4. Add secrets to GitHub repository:
    - DOCKER_USER_NAME: DockerHub username
    - DOCKER_PASS_TOKEN: DockerHub access token

### GitHub Actions Setup

1. Configure workflow files in .github/workflows/
2. Ensure repository secrets are set

### AWS EC2 Self-Hosted Runner Setup

1. Launch an EC2 instance (t2.micro recommended)

2. Install Docker:

```bash
sudo apt-get remove docker docker-engine docker.io
sudo apt-get update
sudo apt-get update -y
sudo apt install docker.io -y
```

3. Configure GitHub self-hosted runner:

- Navigate to repository settings > Actions > Runners
- Click "New self-hosted runner"
- Follow the provided instructions

![ec2](https://github.com/aakashvardhan/thyroid_recurrence_predictor/blob/main/asset/ec2_runner_screenshot.png)

4. Start the runner as a service


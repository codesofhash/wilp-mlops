# MLOps Assignment 01

This repository contains the deliverables for the MLOps Assignment 01, which covers the foundations of MLOps, model experimentation, CI/CD pipeline setup, experiment tracking, and model deployment.

## Objective

The goal of this assignment is to understand and implement core MLOps concepts by setting up CI/CD pipelines, experiment tracking with MLflow, data versioning with DVC, and deploying a machine learning model using Docker and potentially Kubernetes.

### Tasks Overview

- **M1: MLOps Foundations**
  - Set up a CI/CD pipeline using GitHub Actions or GitLab CI.
  - Implement version control using Git.
- **M2: Process and Tooling**
  - Track experiments with MLflow.
  - Use DVC for data versioning.
- **M3: Model Experimentation and Packaging**
  - Hyperparameter tuning with Optuna or GridSearchCV.
  - Package the best model with Docker and Flask.
- **M4: Model Deployment & Orchestration (Optional)**
  - Deploy the model using cloud services like AWS, GCP, or Azure, and manage it with Kubernetes.
- **M5: Final Deliverables**
  - A zip file containing the code, data, model, and a one-page summary.
  - A screen recording explaining and showing the work done.

## Project Setup

### Step 1: Environment Setup

1. **Install Required Tools:**
   - Python 3.8 or higher
   - Git
   - Docker

2. **Install Python Libraries:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib mlflow dvc jupyter

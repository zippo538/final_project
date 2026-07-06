# Final Project Dibimbing AI/ML 7 — Machine Learning Pipeline with FastAPI Deployment
Using python **3.12**

This repository contains the implementation of a complete Machine Learning workflow, including data preprocessing, model training with hyperparameter tuning, experiment tracking using MLflow, and deployment via FastAPI inside Docker. The project is structured for modularity, scalability, and production readiness.

## 🚀 Project Overview

The goal of this project is to build, train, evaluate, and deploy a sentiment analysis model. The system includes:

* **Model Factory pattern** to dynamically load ML algorithms.
* **Optuna** for hyperparameter optimization.
* **MLflow** for experiment tracking and model versioning.
* **FastAPI** as a serving layer for inference APIs.
* **Docker** containerization for consistent and portable deployment.

This project demonstrates real-world ML engineering workflow and MLOps practices.

## 📂 Project Structure

```
final_project/
│── artifact/              # location dataset and model.pkl
│── config/
│   ├── config.yaml        # Global configuration file
│── logs/                  # File save log
│── research/              # Reserch dataset jupyter notebook
│── src/
│   ├── api/               # FastAPI application
│   ├── data/              # Dataset loaders and preprocessing
│   ├── models/            # Model Factory + training logic
│   ├── utils/             # Logging, configuration, helpers
│   ├── services/          # Function of chat llm and predict category 
│   ├── pipeline/          # Training and inference pipelines
│── .dockerignore          # Ignore file not used it
│── docker-compose.yml     # Docker configuration
│── Dockerfile.fastapi     # Docker container FastApi
│── Home.py                # Streamlit Chat Bot
│── README.md              # Project documentation
│── requirements.txt       # Python dependencies
```

## 🧠 Features

### 🔹 Machine Learning

* Multiple model options available through Model Factory
* Automatic hyperparameter optimization using **Optuna**
* Evaluation metrics: accuracy, precision, recall, F1-score

### 🔹 MLOps

* **MLflow Tracking & Model Registry** integrated
* Automatic model logging and artifact storage

### 🔹 Deployment

* REST API built with **FastAPI**
* Docker image for production
* Endpoint for real-time prediction

## ⚙️ Installation

```bash
git clone https://github.com/zippo538/final_project.git
cd final_project
pip install -r requirements.txt
```

## 🏋️ Training the Model

Run the training pipeline:

```bash
python src/run_pipeline.py
```

This will:

* Load dataset
* Run Optuna optimization
* Train the best model
* Log everything to MLflow

## 🧪 API Usage

Start FastAPI server:

```bash
uvicorn src.api.main:app --reload
```

Open docs:

```
http://127.0.0.1:8000/docs
```

Example prediction request:

```json
{
  "text": "This government policy is terrible"
}
```

## 🐳 Docker Deployment

Build Docker image:

```bash
docker build -t myfastapi:latest -f Dockerfile.fastapi .
```

Run container:

```bash
docker run -d -p 8000:8000 -NAME myfastapi myfastpi:latest
```

## 📊 MLflow Tracking

Start MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

View experiments at:

```
http://localhost:5000
```

## 📌 Notes

* Ensure the dataset path inside `config.yaml` is correct.
* MLflow tracking server can be switched to remote storage if needed.

## 🤝 Contribution

Contributions are welcome! Please open an issue or submit a pull request.

## 📜 License

This project is licensed under the MIT License.

---

If you need improvements, restructuring, or a more formal academic format, feel free to ask!


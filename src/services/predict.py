import os
import mlflow
from typing import Any, Tuple,List
from datetime import datetime
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from typing import List

from src.api.schemas import ModelInfo, ModelMetrics
from src.utils.config import config
from src.utils.logger import default_logger as logger
from src.data.data_preprocess import TextPreprocessor


def predict_sentiment(input_text : str, preprocessor : TextPreprocessor,model,tfidf):
        text = pd.DataFrame([input_text],columns=["clean_text"])
        
        # Preprocess data
        processed_data = preprocessor.transform(text)
        
        #clean data
        clean_input = processed_data["clean_text"].iloc[0]
        
        # Make prediction
        output_class = model.predict(tfidf.transform([clean_input]))
        pred= str(output_class[0])
        
        return pred
    

def load_production_model(
    model_name: str = None,
    min_accuracy: float = 0.85,
    metric_name: str = "accuracy_score"
) -> Tuple[Any, ModelInfo]:
    """
    Load model Production langsung dari MLflow Model Registry.

    Returns:
        model, model_info
    """

    # Pastikan tracking URI mengarah ke SQLite MLflow kamu
    tracking_uri = config.get("mlflow.tracking_uri", "sqlite:///mlflow.db")
    experiment_name = config.get("mlflow.experiment_name")

    mlflow.set_tracking_uri(tracking_uri)

    print("Tracking URI:", mlflow.get_tracking_uri())

    client = MlflowClient(tracking_uri=tracking_uri)

    if not model_name:
        model_name = config.get("mlflow.model_name")

    if not model_name:
        raise ValueError("model_name tidak boleh kosong")

    # ============================================================
    # Cek experiment
    # ============================================================
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        raise ValueError(f"Experiment tidak ditemukan: {experiment_name}")

    logger.info(f"Found experiment with ID: {experiment.experiment_id}")

    # ============================================================
    # Ambil semua versi model yang stage-nya Production
    # ============================================================
    model_versions = client.search_model_versions(
        filter_string=f"name='{model_name}'"
    )

    production_versions: List[ModelVersion] = [
        v for v in model_versions
        if v.current_stage == "Production"
    ]

    if not production_versions:
        raise ValueError(
            f"Tidak ada model '{model_name}' dengan stage Production"
        )

    logger.info(f"Found {len(production_versions)} Production model version(s)")

    # ============================================================
    # Pilih Production version dengan accuracy tertinggi
    # ============================================================
    best_version = None
    best_run = None
    best_score = -1.0

    for version in production_versions:
        run_id = version.run_id

        if not run_id:
            logger.warning(
                f"Model version {version.version} tidak memiliki run_id"
            )
            continue

        run = client.get_run(run_id)
        metrics = run.data.metrics

        score = metrics.get(metric_name)

        if score is None:
            logger.warning(
                f"Run {run_id} tidak memiliki metric '{metric_name}'"
            )
            continue

        logger.info(
            f"Model version {version.version}, run {run_id}, "
            f"{metric_name}: {score}"
        )

        if score > best_score:
            best_score = score
            best_version = version
            best_run = run

    if best_version is None or best_run is None:
        raise ValueError(
            f"Tidak ada Production model dengan metric '{metric_name}'"
        )

    if best_score < min_accuracy:
        raise ValueError(
            f"Model terbaik belum memenuhi minimum accuracy. "
            f"{metric_name}={best_score}, minimum={min_accuracy}"
        )

    # ============================================================
    # Load model langsung dari MLflow Registry
    # Tidak perlu os.path.join("mlruns", ...)
    # ============================================================
    model_uri = f"models:/{model_name}/{best_version.version}"

    logger.info(f"Loading model directly from MLflow URI: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri)

    # ============================================================
    # Buat model info
    # ============================================================
    run_metrics = best_run.data.metrics

    metrics = ModelMetrics(
        accuracy_score=run_metrics.get("accuracy_score", 0.0),
        precision=run_metrics.get("precision", 0.0),
        recall=run_metrics.get("recall", 0.0),
        f1_score=run_metrics.get("f1_score", run_metrics.get("f1", 0.0)),
    )

    model_info = ModelInfo(
        run_id=best_run.info.run_id,
        model_name=model_name,
        metrics=metrics,
        load_timestamp=datetime.now().isoformat()
    )

    logger.info(
        f"Loaded model '{model_name}' version {best_version.version} "
        f"with {metric_name}: {best_score}"
    )

    return model, model_info
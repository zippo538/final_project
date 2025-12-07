import os
import mlflow
import datetime
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
    

def load_production_model(model_name:str = None) -> tuple[str, ModelInfo]:
    """
    Get the path to the latest trained model
    
    Returns:
        Tuple containing model path and model info
    """
    print("Tracking URI:", mlflow.get_tracking_uri())
    client = MlflowClient()
    
    #search model version
    model_versions = client.search_model_versions(f"name='{model_name}'")
    for v in model_versions:
        print("Version:", v.version, "Artifact:", v.source)
    
    
    # Get experiment
    experiment = client.get_experiment_by_name(config.get("mlflow.experiment_name"))
    if not experiment:
        raise ValueError("No experiment found")
    
    logger.info(f"Found experiment with ID: {experiment.experiment_id}")
    
    #get latest client version 
    production_versions : List[ModelVersion] = client.get_latest_versions(
        name=model_name,
        stages=["Production"]
    )
    if not production_versions:
            raise ValueError(f"No model found in 'Production' stage for name: {model_name}")
    
    logger.info(f"Found Production version: {production_versions[0]}")
    
    version_object = production_versions[0]
    run_id = version_object.run_id
    
    
    # Get all runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy_score DESC"]
    )
    
    logger.info(f"Found {len(runs)} runs")
    
    if not runs:
        raise ValueError("No runs found in the experiment")
    
    # Find best run based on average of recall and f1
    best_run = None
    best_score = -1
    
    for run in runs:
        metrics = run.data.metrics
        if 'accuracy_score' in metrics :
            acc = metrics['accuracy_score'] 
            logger.info(f"Run {run.info.run_id} score: {acc}")
            if acc > best_score:
                best_score = acc
                best_run = run
                
    if not best_run:
        raise ValueError("No valid runs found with required metrics")
    
    # Get model path
    run_id = best_run.info.run_id
    logger.info(f"Best run ID: {run_id}")
    version = client.search_model_versions(f"name='{model_name}'")
    latest_version = max(version, key=lambda x:int(x.version))
    artifact_uri = latest_version.source.split("models:/")[1]
    
    print(f"uri artifact : {artifact_uri}")
    
    # Try to load model 
    try : 
        logger.info("Trying to load model from local system")
        local_path = os.path.join("mlruns",experiment.experiment_id,"models",artifact_uri,"artifacts")
        if not os.path.exists(local_path):
            raise ValueError(f"Local path does not exist {local_path}")
        model = mlflow.pyfunc.load_model(local_path)
    except Exception as e:
        logger.error(f"Error load model local system : {e}")
    
    # Create model info
    metrics = ModelMetrics(
        accuracy_score=best_run.data.metrics.get('accuracy', 0.0),
        precision=best_run.data.metrics.get('precision', 0.0),
        recall=best_run.data.metrics.get('recall', 0.0),
        f1_score=best_run.data.metrics.get('f1', 0.0),
    )
    
    model_info = ModelInfo(
        run_id=run_id,
        model_name= model_name,
        metrics=metrics,
        load_timestamp=datetime.now().isoformat()
        )
    return model, model_info
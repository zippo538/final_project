from fastapi import FastAPI, HTTPException
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from typing import List
import pandas as pd
import os
from src.utils.config import config
from datetime import datetime
from src.utils.logger import default_logger as logger
from src.data.data_preprocess import TextPreprocessor
from src.api.schemas import (
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
    ModelInfo,
    ModelMetrics,
    LLMRequest,
    LLMResponse
)
from src.services.chat_services import ChatLLMService
from src.api.chat_llm import PipelineCategory, ChatHistoryPipeline, chat_message_store
import pickle

app = FastAPI(
    title="Chat Sentiment Analysis Prediction API",
    description="API for Sentiment Analysis",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Load model and preprocessor on startup"""
    global model, preprocessor, model_info,tfidf
    
    # 1. Set MLflow tracking URI
    # Pastikan ini menunjuk ke lokasi mlflow.db Anda, defaultnya adalah 'sqlite:///mlflow.db'
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    
    try:
        logger.info("Loading production model from MLflow Registry.")
        
        # 2. Cari dan muat model Production
        model, model_info = load_production_model(config.get("mlflow.best_model"))
        
        # 3. Inisialisasi dan muat preprocessor
        # Ini penting agar API dapat menggunakan preprocessor yang sudah di-fit
        preprocessor = TextPreprocessor()
        preprocessor.load(path=config.get_path("paths.model_preprocess"))
        tfidf = preprocessor.load(config.get_path("paths.model_tfidf"))
        
        logger.info(f"Model ({model_info.model_name} and preprocessor loaded successfully")
        
    except Exception as e:
        logger.error(f"FATAL ERROR during startup: {str(e)}")
        # Biarkan pengecualian naik sehingga FastAPI gagal startup jika model tidak dapat dimuat
        raise


def predict_sentiment(input_text : str,):
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
    client = MlflowClient()
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    
    
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
    
    # Try to load model by last version
    try:
        logger.info("Trying to load model from last version")
        version = client.search_model_versions(f"name='{model_name}'")
        latest_version = max(version, key=lambda x:int(x.version))
        last_version_number = latest_version.version
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{last_version_number}")
    except Exception as e : 
        logger.error(f"Error load model last version : {e}")
        try :
            logger.info("Trying to load model from Production") 
            model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
        except Exception as e : 
            logger.error(f"Error load model with production : {e}")
            # try local system
            try : 
                logger.info("Trying to load model from local system")
                local_path = os.path.join("mlruns",experiment.experiment_id,"models",f"m-{run_id}","artifacts","")
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



@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Sentiment Analysis Modi API",
        "model_status": "Loaded",
        "timestamp": datetime.now().isoformat()
    }
    
    
# ===========================
#  ENDPOINT: SENTIMENT ONLY
# ===========================

@app.post("/predict", response_model=SentimentAnalysisResponse)
async def predict(request: SentimentAnalysisRequest):
    
    """
    Classfication sentiment analysis modi
    
    Args:
        request: Classfication request containing input text
        
    Returns:
        Classfication response with category and model info
    """
    try:
        logger.info(f"Received input text : {request}")
        
        sentiment = predict_sentiment(request.input_text)
        
        response = SentimentAnalysisResponse(
            input_text=request.input_text,
            category=sentiment
        )
        
        logger.info(f"Sentiment Analysis with Input : {request.input_text} ({sentiment})")
        return response
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ===========================
#  ENDPOINT: LLM GENERATOR WITH SENTIMENT RESULT
# ===========================
@app.post("/chat_llm",response_model=LLMResponse)
async def chat_llm(request : LLMRequest):
    
    sentiment = predict_sentiment(request.user_message)
    context = ""
    predictor_pipeline = PipelineCategory()
    chat_history_pipeline =  ChatHistoryPipeline(chat_message_store)
    chat_service = ChatLLMService(predictor_pipeline,chat_history_pipeline)
    
    response = chat_service.process_chat(request.user_message,sentiment,context)
    
    return LLMResponse(
        sentiment=sentiment,
        llm_answer=response
    )
    
    
    
    
    
    
    
    
    
    


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_info": model_info
    }
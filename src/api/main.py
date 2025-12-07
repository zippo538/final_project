import mlflow
from fastapi import FastAPI, HTTPException
from mlflow.entities.model_registry import ModelVersion
from src.utils.config import config
from datetime import datetime
from src.services.store_history import chat_store
from src.data.data_preprocess import TextPreprocessor


from src.utils.logger import default_logger as logger
from src.api.schemas import (
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
    LLMRequest,
    LLMResponse
)
from src.services.chat_services import ChatLLMService
from src.services.chat_llm import PipelineCategory, ChatHistoryPipeline
from src.services.predict import predict_sentiment,load_production_model

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
    predictor_pipeline = PipelineCategory()
    chat_history_pipeline =  ChatHistoryPipeline(chat_store)
    chat_service = ChatLLMService(predictor_pipeline,chat_history_pipeline,chat_store)
    
    response,history = chat_service.process_chat(request.user_message,sentiment)
    
    print(history)
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
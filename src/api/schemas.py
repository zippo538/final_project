from pydantic import BaseModel, Field
from typing import Dict, Any

# --- Metrik untuk Masalah classification ---
class ModelMetrics(BaseModel):
    """Metrics specific to the deployed Regression Model."""
    accuracy_score: float = Field(..., description="Accuracy Score")
    precision: float = Field(..., description="Precision")
    recall: float = Field(..., description="Recall")
    f1_score: float = Field(..., description="F1 Score")

# --- Informasi Model yang Dikerahkan ---
class ModelInfo(BaseModel):
    """Information about the currently loaded MLflow model."""
    run_id: str = Field(..., description="MLflow Run ID where the model was logged.")
    model_name: str = Field(..., description="Registered Model Name.")
    metrics: ModelMetrics = Field(..., description="Key metrics from the best run.")
    load_timestamp: str = Field(..., description="Timestamp when the model was loaded.")

# ===========================
#  SCHEMAS : PREDICT SENTIMENT
# ===========================

# --- Skema Input Prediksi ---
class SentimentAnalysisRequest(BaseModel):
    """Input text required for classification category."""
    input_text : str = Field(..., description="Input Text ") 
    
    class Config:
        # Contoh data untuk dokumentasi Swagger/Redoc
        schema_extra = {
            "example": {
                "clean_text" : "I Love India",
            }
        }

# --- Skema Output Prediksi ---
class SentimentAnalysisResponse(BaseModel):
    """Prediction output containing the category sentiment."""
    input_text : str = Field(..., description="I love modi")
    category: str = Field(..., description="Positive")
    # model_info: ModelInfo = Field(..., description="Metadata of the model used for prediction.")
    
# ===========================
#  SCHEMAS : GENERATOR LLM
# ===========================    

class LLMRequest(BaseModel):
    user_message: str = Field(..., description="Modi is great leader!")

class LLMResponse(BaseModel):
    sentiment: str = Field(..., description="2")
    llm_answer: str = Field(..., description="Terima kasih atas opininya! ...")

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import asynccontextmanager
from src.utils.config import config
from datetime import datetime
from src.services.mysql_store import MySQLChatMessageStore
from src.data.data_preprocess import TextPreprocessor
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore

from src.utils.logger import default_logger as logger
from src.api.schemas import (
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
    LLMRequest,
    LLMResponse,
)
from src.services.chat_services import ChatLLMService
from src.services.chat_llm import PipelineCategory, ChatHistoryPipeline
from src.services.predict import predict_sentiment, load_production_model
import mlflow


model = None
preprocessor = None
model_info = None
tfidf = None

# Per-session InMemory stores for non-persistent /chat_llm endpoint.
# Keyed by session_id. Lives only for the lifetime of the server process.
inmemory_stores: dict[str, InMemoryChatMessageStore] = {}


def _get_inmemory_store(session_id: str) -> InMemoryChatMessageStore:
    if session_id not in inmemory_stores:
        inmemory_stores[session_id] = InMemoryChatMessageStore()
    return inmemory_stores[session_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and preprocessor on startup."""
    global model, preprocessor, model_info, tfidf

    mlflow.set_tracking_uri('sqlite:///mlflow.db')

    try:
        logger.info("Loading production model from MLflow Registry.")
        model, model_info = load_production_model(config.get("mlflow.best_model"))

        preprocessor = TextPreprocessor()
        preprocessor.load(path=config.get_path("paths.model_preprocess"))
        tfidf = preprocessor.load(config.get_path("paths.model_tfidf"))

        logger.info("Model and preprocessor loaded successfully")

    except Exception as e:
        logger.error(f"FATAL ERROR during startup: {e}")
        raise

    yield

    inmemory_stores.clear()


app = FastAPI(
    title="Chat Sentiment Analysis Prediction API",
    description="API for Sentiment Analysis",
    version="1.0.0",
    lifespan=lifespan,
)


def _safe_int_category(sentiment_str: str) -> int:
    """Convert sentiment string ('0','1','2') to int. Unknown -> -1."""
    try:
        return int(sentiment_str)
    except (ValueError, TypeError):
        return -1


@app.get("/")
async def root():
    return {
        "message": "Sentiment Analysis Modi API",
        "model_status": "Loaded",
        "timestamp": datetime.now().isoformat(),
    }


# ===========================
#  ENDPOINT: SENTIMENT ONLY
# ===========================
@app.post("/predict", response_model=SentimentAnalysisResponse)
async def predict(request: SentimentAnalysisRequest):
    try:
        logger.info(f"Received input text : {request}")
        sentiment = predict_sentiment(
            input_text=request.user_message,
            preprocessor=preprocessor,
            model=model_info,
            tfidf=tfidf,
        )
        response = SentimentAnalysisResponse(
            input_text=request.input_text,
            category=sentiment,
        )
        logger.info(f"Sentiment Analysis with Input : {request.input_text} ({sentiment})")
        return response
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
#  ENDPOINT: LLM - NON-PERSISTENT (InMemory per session)
#  History lost on server restart. No category stored.
# ==========================================================
@app.post("/chat_llm", response_model=LLMResponse)
async def chat_llm(request: LLMRequest):
    session_id = request.session_id
    store = _get_inmemory_store(session_id)
    history_limit = int(config.get("chatbot.history_limit", 20))

    sentiment = predict_sentiment(
        input_text=request.user_message,
        preprocessor=preprocessor,
        model=model,
        tfidf=tfidf,
    )

    predictor_pipeline = PipelineCategory()
    chat_history_pipeline = ChatHistoryPipeline(store, limit=history_limit)
    chat_service = ChatLLMService(predictor_pipeline, chat_history_pipeline, store, persist_category=False)

    response, history = chat_service.process_chat(
        query=request.user_message,
        sentiment=sentiment,
        category=None,
    )

    logger.debug(f"chat_llm history preview: {history[:200]}")
    return LLMResponse(sentiment=sentiment, llm_answer=response)


# ==========================================================
#  ENDPOINT: LLM - PERSISTENT (MySQL) with category
#  Survives restarts. Stores sentiment category per message.
# ==========================================================
@app.post("/chat_llm/memory", response_model=LLMResponse)
async def chat_llm_memory(request: LLMRequest):
    session_id = request.session_id
    store = MySQLChatMessageStore(session_id=session_id)
    history_limit = int(config.get("chatbot.history_limit", 20))

    sentiment = predict_sentiment(
        input_text=request.user_message,
        preprocessor=preprocessor,
        model=model,
        tfidf=tfidf,
    )
    category = _safe_int_category(sentiment)

    predictor_pipeline = PipelineCategory()
    chat_history_pipeline = ChatHistoryPipeline(store, limit=history_limit)
    chat_service = ChatLLMService(predictor_pipeline, chat_history_pipeline, store, persist_category=True)

    response, history = chat_service.process_chat(
        query=request.user_message,
        sentiment=sentiment,
        category=category,
    )

    store.close()
    logger.debug(f"chat_llm_memory history preview: {history[:200]}")
    return LLMResponse(sentiment=sentiment, llm_answer=response)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_info": model_info,
    }


@app.delete("/chat/{session_id}")
async def delete_chat_history(session_id: str = "default"):
    """Delete all MySQL chat messages for a session + clear in-memory cache."""
    try:
        store = MySQLChatMessageStore(session_id=session_id)
        store.delete_messages()
        store.close()

        if session_id in inmemory_stores:
            inmemory_stores[session_id].delete_messages()
            del inmemory_stores[session_id]

        return {"status": "deleted", "session_id": session_id}
    except Exception as e:
        logger.error(f"Error deleting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/{session_id}")
async def get_chat_history(session_id: str = "default"):
    """Get all chat messages for a session from MySQL."""
    try:
        store = MySQLChatMessageStore(session_id=session_id)
        messages = store.retrieve()
        store.close()
        return {
            "session_id": session_id,
            "total_messages": len(messages),
            "messages": [
                {
                    "role": m.role.value if hasattr(m.role, "value") else str(m.role),
                    "content": m.text,
                    "category": m.meta.get("category"),
                }
                for m in messages
            ],
        }
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def list_sessions():
    """List all chat sessions with title = first user message."""
    from src.utils.config import config as cfg
    import pymysql
    from pymysql.cursors import DictCursor

    host = cfg.get("mysql.host", "localhost")
    port = int(cfg.get("mysql.port", 3306))
    user = cfg.get("mysql.user", "root")
    password = cfg.get("mysql.password", "password")
    database = cfg.get("mysql.database", "chatbot_modi")
    table = cfg.get("mysql.table", "chat_messages")

    try:
        conn = pymysql.connect(
            host=host, port=port, user=user,
            password=password, database=database,
            cursorclass=DictCursor, autocommit=True,
        )
        sql = f"""
        SELECT
            s.session_id,
            MIN(s.created_at) AS created_at,
            (
                SELECT content FROM {table} t2
                WHERE t2.session_id = s.session_id AND t2.role = 'user'
                ORDER BY t2.created_at ASC LIMIT 1
            ) AS title,
            COUNT(*) AS message_count
        FROM {table} s
        GROUP BY s.session_id
        ORDER BY MIN(s.created_at) DESC
        """
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        conn.close()

        sessions = []
        for r in rows:
            raw_title = r.get("title") or r.get("title".upper()) or ""
            title = raw_title[:50] if raw_title else "Untitled Session"
            sessions.append({
                "session_id": r["session_id"],
                "title": title,
                "created_at": str(r["created_at"]) if r.get("created_at") else None,
                "message_count": r["message_count"],
            })
        return {"sessions": sessions}

    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

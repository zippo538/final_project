## run pipeline

```bash
python src/run_pipeline.py
```

## run api

```bash
uvicorn src.api.main:app --reload
```

## view experimenttmlflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## run streamlit

```bash
streamlit run Home.py
```

## Docker

### docker build & run

1. FastAPI Image :

```bash
#build FastAPI image
docker build -t myfastapi:latest -f Dockerfile.fastapi .

# run FastAPI Container
docker run -d -p 8000:8000 -NAME myfastapi myfastpi:latest
```

2. Streamlit Image

```bash
#build Streamlit image
docker build -t streamlit:latest -f Dockerfile.streamlit .

# run Streamlit Container
docker run -d -p 8000:8000 -NAME streamlit streamlit:latest

```

### docker compose

```bash
docker-compose up --build -d
```

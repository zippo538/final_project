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

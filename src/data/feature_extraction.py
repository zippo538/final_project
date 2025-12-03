import mlflow
import joblib
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

from src.utils.logger import default_logger as logger
from src.utils.config import config


class TfidfFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Professional TF-IDF feature extraction class with MLflow integration.
    """

    def __init__(
        self,
        feature_column: Optional[str] = None,
        target_column: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        artifact_path: str = "tfidf_vectorizer",
        register_model_name: str = "TFIDFVectorizer",
        save_path: str = "models/tfidf"
    ):
        self.feature_column = feature_column or config.get("features.feature_column")
        self.target_column = target_column or config.get("features.target_column")
        self.mlflow_tracking_uri = mlflow_tracking_uri or config.get("mlflow.tracking_uri")

        self.artifact_path = artifact_path
        self.register_model_name = register_model_name
        self.save_path = Path(save_path)

        self.vectorizer = None
        self.label_encoder = LabelEncoder()

        logger.info("TF-IDF FeatureExtractor initialized.")

    # ===========================================================
    # Fit / Transform (Scikit-learn compatible)
    # ===========================================================
    def fit(self, X, y=None):
        logger.info("Fitting TF-IDF Vectorizer...")
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(X[self.feature_column])
        return self

    def transform(self, X: pd.DataFrame):
        if self.vectorizer is None:
            raise ValueError("Vectorizer is not fitted. Call fit() before transform().")

        logger.info("Transforming text to TF-IDF matrix...")
        tfidf_matrix = self.vectorizer.transform(X[self.feature_column])
        return tfidf_matrix

    # ===========================================================
    # Full Fit-Transform Pipeline
    # ===========================================================
    def fit_transform(self, df: pd.DataFrame):
        logger.info("Running fit_transform for TF-IDF...")
        self.vectorizer = TfidfVectorizer()
        X_tfidf = self.vectorizer.fit_transform(df[self.feature_column])
        return X_tfidf

    # ===========================================================
    # Label Encoding
    # ===========================================================
    def encode_labels(self, df: pd.DataFrame):
        logger.info("Encoding target labels...")
        y = self.label_encoder.fit_transform(df[self.target_column].values).astype(int)
        print(y)
        return y

    # ===========================================================
    # Train-Test Split (optional)
    # ===========================================================
    def get_X_y(self, df: pd.DataFrame) -> Tuple:
        """
        Extract X (TF-IDF) and y (encoded labels)
        """
        X = self.fit_transform(df)
        y = self.encode_labels(df)
        return X, y

    # ===========================================================
    # SMOTE Oversampling
    # ===========================================================
    def apply_smote(self, X_train, y_train):
        logger.info("Applying SMOTE oversampling...")
        smote = SMOTE()
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return X_res, y_res

    # ===========================================================
    # MLflow Model Logging
    # ===========================================================
    def log_to_mlflow(self):
        """
        Log TF-IDF vectorizer to MLflow
        """
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)

            with mlflow.start_run(nested=True):
                logger.info("Logging TF-IDF Vectorizer to MLflow...")

                mlflow.sklearn.log_model(
                    sk_model=self.vectorizer,
                    artifact_path=self.artifact_path,
                    registered_model_name=self.register_model_name
                )

            logger.info("Successfully logged TF-IDF vectorizer to MLflow.")

        except Exception as e:
            logger.error(f"Error logging TF-IDF model to MLflow: {e}")
            raise

    # ===========================================================
    # Save / Load Local Vectorizer
    # ===========================================================
    def save(self):
        """Save TF-IDF vectorizer and LabelEncoder locally"""
        try:
            self.save_path.mkdir(parents=True, exist_ok=True)

            joblib.dump(self.vectorizer, self.save_path / "tfidf_vectorizer.joblib")
            joblib.dump(self.label_encoder, self.save_path / "label_encoder.joblib")

            logger.info(f"Saved TF-IDF vectorizer to: {self.save_path}")

        except Exception as e:
            logger.error(f"Error saving vectorizer: {e}")
            raise

    @staticmethod
    def load(path: str):
        """Load TF-IDF vectorizer and LabelEncoder"""
        vectorizer = joblib.load(Path(path) / "tfidf_vectorizer.joblib")
        label_encoder = joblib.load(Path(path) / "label_encoder.joblib")

        extractor = TfidfFeatureExtractor()
        extractor.vectorizer = vectorizer
        extractor.label_encoder = label_encoder

        logger.info(f"Loaded TF-IDF extractor from: {path}")
        return extractor

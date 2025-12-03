import re
import joblib
import nltk
import pandas as pd
from pathlib import Path
from typing import Optional
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logger import default_logger as logger
from src.utils.config import config

nltk.download('stopwords') 
nltk.download('punkt_tab')

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Professional text preprocessing pipeline compatible with scikit-learn.
    Follows SOLID principles & clean ML architecture.
    """

    def __init__(
        self,
        feature_column: str = None,
        target_column: str = None,
        preprocessing_path: Optional[str] = None,
        language: str = "english"
    ):
        self.feature_column = feature_column or config.get("features.feature_column")
        self.target_column = target_column or config.get("features.target_column")
        self.preprocessing_path = preprocessing_path or config.get(
            "preprocessing_path", "models/preprocessing"
        )
        self.language = language

        self.stop_words = set(stopwords.words(self.language))
        #download nltk
        

        logger.info("Text Preprocessor initialized.")

    # ============================================================
    # Fit is not used for text cleaning but required for pipeline
    # ============================================================
    def fit(self, X, y=None):
        return self

    # ============================================================
    # TRANSFORM — Core Preprocessing Logic
    # ============================================================
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()  # avoid modifying original dataset
        try:
            logger.info("Starting preprocessing...")

            df = self._clean_dataframe(df)
            df[self.feature_column] = (
                df[self.feature_column]
                .astype(str)
                .apply(self._normalize_text)
                .apply(self._remove_non_alphanumeric)
                .apply(self._remove_emojis)
                .apply(self._remove_stopwords)
            )

            if self.target_column in df.columns:
                df[self.target_column] = df[self.target_column].astype(int)
            
            print(df[self.target_column].value_counts())
            logger.info("Preprocessing completed successfully.")
            return df

        except Exception as e:
            logger.error(f"Transform error: {e}")
            raise

    # ============================================================
    # 🔧 CLEANING UTILITIES
    # ============================================================
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna()
        df = df.drop_duplicates()
        logger.info("Dropped NaN and duplicate rows.")
        return df

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\n", " ").lower()
        text = re.sub(r"[_-]", " ", text)
        return text

    def _remove_non_alphanumeric(self, text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9\s]", "", text)

    def _remove_emojis(self, text: str) -> str:
        emoji_pattern = r"[^\w\s,.!?-]"
        clean_text = re.sub(emoji_pattern, "", text)
        clean_text = clean_text.replace("\uFFFD", "")
        return clean_text

    def _remove_stopwords(self, text: str) -> str:
        tokens = nltk.word_tokenize(text)
        filtered = [t for t in tokens if t not in self.stop_words]
        return " ".join(filtered)

    # ============================================================
    # SAVE / LOAD
    # ============================================================
    def save(self):
        """Save the transformer object using joblib"""
        Path(self.preprocessing_path).mkdir(parents=True, exist_ok=True)

        save_path = Path(self.preprocessing_path) / "preprocessor.joblib"
        joblib.dump(self, save_path)

        logger.info(f"Preprocessor saved to: {save_path}")

    @staticmethod
    def load(path: str):
        """Load saved transformer"""
        preprocessor = joblib.load(path)
        logger.info(f"Preprocessor loaded from: {path}")
        return preprocessor

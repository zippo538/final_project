#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.utils.logger import default_logger as logger
from src.utils.config import config
import pandas as pd
from imblearn.over_sampling import SMOTE
import mlflow


class FeatureExtraction:
    def __init__(self):
        pass

    def transform_tfidf(self, df: pd.DataFrame, feature_column: str) :
        """
        Transform text in the given dataframe column to TF-IDF and return the vectorizer and matrix.
        """
        try:
            logger.info("Transform TFIDF")
            tfidf_vectorizer = TfidfVectorizer()
            X_tfidf = tfidf_vectorizer.fit_transform(df[feature_column])
            mlflow.sklearn.log_model(sk_model=tfidf_vectorizer)
            return X_tfidf
        
        except Exception as e:
            logger.exception(f"Error Transform TFIDF : {e}")
            raise
    
    def split_data(self,df:pd.DataFrame,X_tf_idf,target_column :str) : 
        try :
            logger.info("Split Data")
            
            le = LabelEncoder()
            X = X_tf_idf
            y = le.fit_transform(df[target_column])
            
            X_train, y_train, X_test, y_test =train_test_split(X,y,test_size=0.2,random_state=42)
            
            return X_train, y_train, X_test, y_test
        
        except Exception as e : 
            logger.error(f"Error Split Data : {e}")
            raise
    
    def smote_processing(self, X_train, y_train) : 
        try : 
            logger.info("Smote Processing")
            smote = SMOTE()
            X_res,y_res = smote.fit_resample(X_train,y_train)
            
            return X_res,y_res
        except Exception as e:
            logger.error(f"Error SMOTE Processing : {e}")
            
            
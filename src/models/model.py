from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from typing import Dict,Any, Type
from src.utils.logger import default_logger as logger
from src.utils.config import config


class ModelFactory:
    @staticmethod
    def get_model_config()-> Dict[str,Dict[str,Any]]:
        return { 
                'logistic_regression': {
        'model': LogisticRegression(),
        'params': {
            'C' : 10.0,
            'solver': 'saga',
            'max_iter': 200
        }
    },

    'naive_bayes': {
        'model': MultinomialNB(),
        'params': {
            'alpha': 1.0
        }
    },

    'knn_classifier': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': 5,
            'weights': 'distance',
            'metric': 'euclidean',
        }
    },

    'random_forest_classifier': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': 100,
            'max_depth': 6,
            'min_samples_split': 2,
            'min_samples_leaf': 2,
            'random_state': 42
        }
    },
                
}
        
    @classmethod
    def create_model(cls, model_type: str) -> Any:
        """
        Create model instance
        
        Args:
            model_type: Type of model to create
            
        Returns:
            Model instance
        """
        try:
            logger.info(f"Creating model of type: {model_type}")
            
            # Get model configurations
            model_configs = cls.get_model_config()
            
            if model_type not in model_configs:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Get model class and parameters
            model_info = model_configs[model_type]
            model_class = model_info['class']
            model_params = model_info['params']
            
            
            # Create model instance
            model = model_class(**model_params)
            
            logger.info(f"Successfully created {model_type} model")
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise
        
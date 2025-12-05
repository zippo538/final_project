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
        'class': LogisticRegression,
        'params': {
            'C' : 10.0,
            'solver': 'saga',
            'max_iter': 200
        }
    },

    'naive_bayes': {
        'class': MultinomialNB,
        'params': {
            'alpha': 1.0
        }
    },

    'knn_classifier': {
        'class': KNeighborsClassifier,
        'params': {
            'n_neighbors': 5,
            'weights': 'distance',
            'metric': 'euclidean',
        }
    },

    'random_forest_classifier': {
        'class': RandomForestClassifier,
        'params': {
            'n_estimators': 100,
            'max_depth': 6,
            'min_samples_split': 2,
            'min_samples_leaf': 2,
            'random_state': 42
        }
    },
                
}
# Optuna
    @staticmethod
    def get_hyperparameter_space(trial, model_type: str) -> Dict[str, Any]:

        if model_type == "logistic_regression":
            return {
                "C": trial.suggest_float("C", 0.01, 20.0),
                "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
                "max_iter": trial.suggest_int("max_iter", 100, 500)
            }

        elif model_type == "naive_bayes":
            return {
                "alpha": trial.suggest_float("alpha", 0.1, 2.0)
            }

        elif model_type == "knn_classifier":
            return {
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan"])
            }

        elif model_type == "random_forest_classifier":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5)
            }

        else:
            raise ValueError(f"No hyperparameter space for model: {model_type}")
        
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
        
    @classmethod
    def create_model_hypertuning(cls, model_type: str, params_override: Dict[str, Any] = None):
        """Create model with default OR overridden parameters"""
        try:
            logger.info(f"Creating model {model_type}")

            config = cls.get_model_config()
            if model_type not in config:
                raise ValueError(f"Invalid model type: {model_type}")

            model_class = config[model_type]["class"]
            params = config[model_type]["params"].copy()

            if params_override:
                params.update(params_override)

            return model_class(**params)

        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
        
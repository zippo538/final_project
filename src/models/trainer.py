from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import mlflow
from src.utils.logger import default_logger as logger
from src.utils.config import config
from src.models.model import ModelFactory
class ModelTrainer : 
    def __init__(self,experiment_name : str = "sentiment_analysis_modi_category"):
        
        self.experiment_name = experiment_name
        self.models_info = {}
        self.tuned_info = {}
        self.best_model = None
        self.setup_mlflow()
        logger.info(f"Initialized ModelTrainer with experiment: {experiment_name}")
        
    def _calculate_metrics(self,y_test: np.ndarray,y_pred:np.ndarray) -> Dict[str,float]:
        try : 
            metrics = {
            'accuracy_score' : accuracy_score(y_test,y_pred),
            'precision' : precision_score(y_test,y_pred,average='weighted',zero_division=0),
            'recall' : recall_score(y_test,y_pred,average='weighted',zero_division=0),
            'f1_score' : f1_score(y_test,y_pred,average='weighted',zero_division=0)
            }
            return metrics
        except Exception as e:
            logger.error(f"Error Calculating metrics : {e}")
            
    def setup_mlflow(self) -> None:
        """Setup MLflow tracking"""
        try:
            # Set MLflow tracking URI
            tracking_uri = config.get('mlflow.tracking_uri', 'sqlite:///mlflow.db')
            mlflow.set_tracking_uri(tracking_uri)
            
            # Create or get experiment
            try:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
            except:
                self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
            
            mlflow.set_experiment(self.experiment_name)
            logger.info("MLflow setup completed successfully")
            
        except Exception as e:
            logger.error(f"Error setting up MLflow: {str(e)}")
            raise
    
    def train_model(self,model_type:str, X_train, y_train,
                    X_test, y_test) -> Dict[str,Any]: 
            """
            Train Single Model
            """
            try : 
                logger.info(f"Training {model_type} model")
                 # Create and train model
                model = ModelFactory.create_model(model_type)
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred)
                
                # Log with MLflow using nested runs
                with mlflow.start_run(run_name=model_type, nested=True) as run:
                    # Log parameters and metrics
                    mlflow.log_params(model.get_params())
                    mlflow.log_metrics(metrics)
                    
                    # Log model
                    mlflow.sklearn.log_model(
                        model,
                        model_type,
                        registered_model_name=f"sentiment_analysis_{model_type}_model"
                    )
                
                # Store model info
                model_info = {
                    'model': model,
                    'metrics': metrics,
                    'run_id': run.info.run_id
                }
                self.models_info[model_type] = model_info
                
                logger.info(f"Completed training {model_type} model")
                logger.info(f"Metrics: {metrics}")
                
                return model_info
                    
            except Exception as e:
                logger.error(f"Error training {model_type} model: {str(e)}")
                raise
            
    def train_all_models(self, X_train, y_train, X_test , y_test) -> Dict[str, Dict[str, Any]]:
        """
        Train all configured models
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary containing info for all trained models
        """
        try:
            logger.info("Starting training of all models")
            
            for model_type in ModelFactory.get_model_config().keys():
                self.train_model(model_type, X_train, y_train, X_test, y_test)
            
            # Select best model
            self._select_best_model()
            
            logger.info("Completed training all models")
            return self.models_info
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
    
    def _select_best_model(self) -> None:
        """Select best model based on recall and f1 score"""
        try:
            logger.info("Selecting best model")
            
            
            best_score = -1
            best_model_type = None
            
                
            for model_type, model_info in self.models_info.items():
                # Calculate combined score (average of recall and f1)
                metrics = model_info['metrics']
                acc = metrics['accuracy_score'] 
                
                if acc > best_score:
                    best_score = acc
                    best_model_type = model_type
            
            if best_model_type:
                self.best_model = self.models_info[best_model_type]
                
                # Transition best model to production in MLflow
                client = mlflow.tracking.MlflowClient()
                model_name = f"sentiment_analysis_{best_model_type}_optuna_best"
                
                latest_versions = client.get_latest_versions(model_name)
                if latest_versions:
                    latest_version = latest_versions[0]
                    client.transition_model_version_stage(
                        name=model_name,
                        version=latest_version.version,
                        stage="Production"
                    )
                
                logger.info(f"Selected {best_model_type} as best model")
                logger.info(f"Best model metrics: {self.best_model['metrics']}")
            
        except Exception as e:
            logger.error(f"Error selecting best model: {str(e)}")
            raise
    
    def get_best_model(self) -> Dict[str, Any]:
        """Get best model info"""
        if self.best_model is None:
            raise ValueError("No best model selected. Train models first.")
        return self.best_model
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get metrics for all trained models"""
        return {model_type: info['metrics'] 
                for model_type, info in self.models_info.items()}
    
    def tune_with_optuna(self, model_type: str, X_train, y_train, X_test, y_test, n_trials=20):
        def objective(trial):
            params = ModelFactory.get_hyperparameter_space(trial, model_type)

            model = ModelFactory.create_model_hypertuning(model_type, params_override=params)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            metrics = self._calculate_metrics(y_test, preds)

            return metrics["accuracy_score"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        logger.info(f"Best params for {model_type}: {best_params}")

        # Train ulang model dengan best params
        final_model = ModelFactory.create_model_hypertuning(model_type, params_override=best_params)
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)

        # Evaluate
        metrics = self._calculate_metrics(y_test, y_pred)

        # Log ke MLflow
        with mlflow.start_run(run_name=f"{model_type}_optuna",nested=True) as run:
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(final_model,
                                     model_type, 
                                     registered_model_name= f"sentiment_analysis_{model_type}_optuna_best")
        
        # Store model info
        model_info = {
            'model': final_model,
            'metrics': metrics,
            'run_id': run.info.run_id
        }
        self.models_info[model_type] = model_info

        return model_info
    
    def tune_all_models(self, X_train, y_train, X_test, y_test, n_trials=20):
        try : 
            logger.info("Startedd all Tuning models")
            
            for model_type in ModelFactory.get_model_config().keys():
                self.models_info[model_type] = self.tune_with_optuna(
                    model_type,
                    X_train, y_train,
                    X_test, y_test,
                    n_trials=n_trials
                )
            self._select_best_model()
                
            logger.info("Completed training all Tuning models")
                
            return self.models_info
        except Exception as e:
            logger.error(f"Error Hypertuning Model : {e}")
    
    
        
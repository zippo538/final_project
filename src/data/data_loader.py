import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Optional, Tuple
from src.utils.logger import default_logger as logger
from src.utils.config import config


class DataLoader:
    def __init__(self,data_path : Optional[Path]=None):
        self.data_path = data_path 
        self.feature_coloum =  config.get('features.feature_column')
        self.target_coloum = config.get('features.target_column')
        logger.info(f"Intialized Data loader : {self.data_path}")
    
    def load_data(self) -> pd.DataFrame : 
        try :
            logger.info("Loading data from csv")
            df = pd.read_csv(self.data_path)
           
            logger.info(f"Data Successfully Load")
            
            self.validate_data(df)            
            
            return df   
        except Exception as e:
            logger.error(f"Error Feature Engineering : {e}")
            raise
    
    def validate_data(self,df:pd.DataFrame)-> True : 
        try : 
            logger.info(f"Validate Data")
             
             #missing value
            miss_feature = df[self.feature_coloum].isnull().sum()
            miss_target = df[self.target_coloum].isnull().sum()
            logger.info(f"Missing Value {miss_feature} : ",miss_feature)
            logger.info(f"Missing Value {miss_target} : ",miss_target)
            df.dropna()
            logger.info(f"Missing Value has been remove")
            
            #duplicate data
            dup_text = df.duplicated(subset=[self.feature_coloum]).sum()
            dup_full = df.duplicated().sum()
            logger.info(f"- Duplicate text only: {dup_text}")
            logger.info(f"- Full duplicate rows: {dup_full}\n")
            df.drop_duplicates()
            
            #label distribution
            logger.info(f"Label Distribution : \n{df[self.target_coloum].value_counts()}")
            
            return True
        
        except Exception as e :
            logger.error(f"Error Validate Data : {e}")
            raise 
        
        
    
    
             
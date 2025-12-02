import pandas as pd
import numpy as np
import re
import nltk
import joblib
from pathlib import Path
from typing import Optional, Tuple
from src.utils.logger import default_logger as logger
from src.utils.config import config
from nltk.corpus import stopwords

class DataPreprocess :
    def __init__(self,preprocessing_path : Optional[str] = None):
        self.preprocessing_path = preprocessing_path or config.get('preprocessing_path','models/preprocessing')
        self.preprocess = False 
        
        logger.info("Initialized Data Preprocessor")
    
    def _preprocessing_path(self)->None:
        Path(self.preprocessing_path).mkdir(parents=True,exist_ok=True)
        
    def run_preprocess(self,df:pd.DataFrame, feature_cols : str, target_col : str) ->True : 
        try : 
            #change int
            df[target_col] = df[target_col].astype(int)
            
            self.clean_data(df)
            self.remove_unimportant_word(df,feature_cols)
            self.stop_word(df,feature_cols)
            self.preprocess = True
            self.save_preprocessor()
            logger.info("Data Preprocessor Has Successfully!!")
            
            
            return self.preprocess
        except Exception as e :
            logger.error(f"Error Run Preprocessor : {e}")
            raise 
     
    def clean_data(self,df:pd.DataFrame)-> pd.DataFrame : 
        try : 
             logger.info(f"Clean Data ")
             
             df.dropna()
             logger.info(f"Delete Nan Values ")
             
             df.drop_duplicates()
             logger.info(f"Delete Duplicate Values ")
             
             return df
        
        except Exception as e:
            logger.error(f"Error Preprocessing Data : {e}")
            raise
        
    def remove_unimportant_word(self,df:pd.DataFrame,feature_cols : str)-> pd.DataFrame : 
        try : 
            logger.info(f"Remove Unimportant Word ")

            #remove space
            df[feature_cols] = df[feature_cols].str.replace('\n',' ')
            logger.info(f"Remove Space")
             
            #change to lower str
            df[feature_cols] = df[feature_cols].apply(lambda x: x.lower())
            logger.info(f"Lower Text")
            
            
            #remove tag like - or _
            df[feature_cols] = df[feature_cols].str.replace('-'," ").str.replace('_'," ")
            logger.info(f"Remove Tag")
            

            #remove characters non alphanumeric
            for text, i in df[feature_cols].items():
                df.at[text,feature_cols] = re.sub(r'[^a-zA-Z0-9\s]', '', i)
            logger.info(f"Remove Character non Alphanumeric")
            
            #remove emoji
            text = []
            emoji_pattern = r"[^\w\s,.!?-]"
            pattern = r"\uFFFD"
            for i in df[feature_cols].values.tolist():
                clean_text = re.sub(emoji_pattern, '', i)
                clean_text = re.sub(pattern, '', clean_text)
                text.append(clean_text)
                
            df[feature_cols] = text 
            
            return df
        
        except Exception as e:
            logger.error(f"Error Remove Uninimportant Word : {e}")
            raise
        
    def stop_word(self,df:pd.DataFrame, feature_cols : str)-> pd.DataFrame : 
        try : 
            logger.info(f"Stop Word ")
            
            df[feature_cols] = df[feature_cols].apply(self.__preprocess_text)
            
            return df
        
        except Exception as e:
            logger.error(f"Error Preprocessing Data : {e}")
            raise    
    
    def __preprocess_text(self,text:str)-> str : 
        nltk.download('stopwords') # cleaning text dari noise khusus stop words
        nltk.download('punkt_tab')
        
        words = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered_token = []
        for token in words :
            if token not in stop_words:
                filtered_token.append(token)
        return ' '.join(filtered_token)

    def save_preprocessor(self)->None:
        try : 
            logger.info(f"Saving preprocessor to {self.preprocessing_path}")
            self._prepare_preprocessing_path()
            
            # save pipeline 
            joblib.dump(
                self.preprocessor_pipeline,
                Path(self.preprocessing_path) / 'pipeline.joblib'
            )
        except Exception as e:
            logger.error(f"Error Save Preprocessor : {str(e)}")
            raise
        
    def load_preprocessor(self)-> None:
        try :
            logger.info(f"Loading Preprocessor from {self.preprocessing_path}")
            
            #load pipeline
            pipeline_path = Path(self.preprocessing_path) / 'pipeline.joblib'
            self.preprocessor_pipeline = joblib.load(pipeline_path)
            
            self.trained = True 
            logger.info("Preprocessor Loaded Successfully")
        except Exception as e:
            logger.error(f"Error loading preprocessor : {str(e)}")
            raise
    
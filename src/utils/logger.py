import logging
import sys
from pathlib import Path
from typing import Optional
import logging.handlers
import time

class CustomLogger:
    @staticmethod
    def setup_logger(name: str, log_file : Optional[str]= None) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
        
        # create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        #create file handler
        if log_file:
            log_path = Path(log_file)
            
            #tentukan path direktori
            log_dir = log_path.parent
            
            # jika tidak ada maka buat folder logs
            log_dir.mkdir(parents=True,exist_ok=True)
            
            file_handler = logging.handlers.TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=7
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger

# create default logger 
default_logger = CustomLogger.setup_logger(
    'sentiment_analysis',
    log_file='logs/sentiment_analysis.log'
    
) 
            
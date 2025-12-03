import yaml
from pathlib import Path
from typing import Dict, Any
from src.utils.logger import default_logger as logger

class Config :
    def __init__(self,config_path:str = "config/config.yaml"):
        
        self.config_path = Path(config_path)
        self.base_dir = self.config_path.parent.parent.resolve()
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str,any]:
        try : 
            logger.info(f"Loading configuration from {self.config_path}")
            with open(self.config_path,'r') as f:
                config = yaml.safe_load(f)
            logger.info("Configuration loaded sucessfully")
            return config
        except Exception as e:
            logger.error(f"Erro loading Configuration : {e}")
        
    def get(self,key:str, default : Any = None) -> Any:
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    def get_path(self, key: str) -> Path:
        value = self.get(key)
        if not value:
            return None
        return self.base_dir / Path(value)

config = Config()
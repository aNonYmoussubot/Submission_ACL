# configs/config.py
import os

class Config:
    API_KEY = os.getenv("DEEPSEEK_API_KEY", "xxx")
    BASE_URL = "https://api.deepseek.com"
    MODEL_NAME = "deepseek-chat"
    TEMPERATURE = 0.0  
    
    Z3_TIMEOUT = 5000 


from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    
    groq_api_key : str = Field(..., validation_alias="GROQ_API_KEY")
    mw_api_key : str = Field(..., validation_alias="MW_API_KEY")
    
    class Config:
        env_file = ".env"


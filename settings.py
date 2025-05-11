from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    
    groq_api_key : str = Field(..., validation_alias="GROQ_API_KEY")
    mw_api_key : str = Field(..., validation_alias="MW_API_KEY")
    openai_api_key : str = Field(..., validation_alias="OPENAI_API_KEY")
    supabase_url : str = Field(..., validation_alias="SUPABASE_URL")
    supabase_key : str = Field(..., validation_alias="SUPABASE_KEY")

    class Config:
        env_file = ".env"


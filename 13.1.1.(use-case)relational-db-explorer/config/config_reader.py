from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class DotEnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=Path(__file__).parent / ".env", extra="allow")
    REDIS_HOST: str
    REDIS_PORT: int
    CHAT_MODEL:str
    POSTGRES_HOST:str
    POSTGRES_PORT:int
    POSTGRES_USERNAME:str
    POSTGRES_PASSWORD:str
    POSTGRES_DATABASE:str
    DATA_DIR:str
    
settings = DotEnvSettings()

if __name__ == "__main__":
    print(settings)

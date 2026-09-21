from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from pydantic import PostgresDsn

class DotEnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=Path(__file__).parent / ".env", extra="allow")
    REDIS_HOST: str
    REDIS_PORT: int
    UVICORN_HOST: str
    UVICORN_PORT: int
    ALLOW_ORIGINS: str
    CHAT_MODEL:str
    ROOT_DIR: str

settings = DotEnvSettings()

if __name__ == "__main__":
    print(settings)

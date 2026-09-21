from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class DotEnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=Path(__file__).parent.parent / ".env", extra="allow")
    REDIS_HOST: str
    REDIS_PORT: int
    UVICORN_HOST: str
    UVICORN_PORT: int
    ALLOW_ORIGINS: str
    MODEL:str

settings = DotEnvSettings()

if __name__ == "__main__":
    print(settings)

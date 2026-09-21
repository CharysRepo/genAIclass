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
    EMBEDDING_MODEL: str
    OCR_MODEL: str
    WATCHED_DIR: str
    DATABASE_URL: PostgresDsn = "postgresql://postgres:postgres@localhost:5432/postgres"
    VECTROR_DIM: int = 1024
    CHUNK_SIZE: int = 1024
    ALLOWED_EXTENSIONS: tuple[str, ...] = (".pdf", ".md")
    MAX_CONCURRENT_TASKS_OCR: int = 10
    MAX_CONCURRENT_TASKS_EMBEDDING: int = 20
    GOOGLE_CREDENTIALS_PATH: str

settings = DotEnvSettings()

if __name__ == "__main__":
    print(settings)

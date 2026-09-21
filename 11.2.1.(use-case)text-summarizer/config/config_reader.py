from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class DotEnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=Path(__file__).parent / ".env", extra="allow")
    CHAT_MODEL:str
    EMBEDDING_MODEL: str
    CHUNK_SIZE: int = 1024
    MAX_CONCURRENT_TASKS: int = 3
    SHORT_DOC_THRESHOLD: int = 20000
    MEDIUM_DOC_THRESHOLD: int = 70000
    LONG_DOC_THRESHOLD: int = 100000

settings = DotEnvSettings()

if __name__ == "__main__":
    print(settings)

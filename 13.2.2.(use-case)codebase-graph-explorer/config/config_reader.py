from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class DotEnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=Path(__file__).parent / ".env", extra="allow")
    REDIS_HOST: str
    REDIS_PORT: int
    CHAT_MODEL:str
    NEO4J_URI:str
    NEO4J_USER:str
    NEO4J_PASSWORD: str
    NEO4J_DATABASE: str
    REPO_PATH:str
    GRAPH_STORE_PATH:str

settings = DotEnvSettings()

if __name__ == "__main__":
    print(settings)

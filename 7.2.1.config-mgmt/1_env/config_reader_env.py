from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class DotEnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=Path(__file__).parent / "config.env")
    HOST: str
    PORT: int
    SECRET_KEY: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: str
    POSTGRES_DB: str


settings = DotEnvSettings()

if __name__ == "__main__":
    print(settings)

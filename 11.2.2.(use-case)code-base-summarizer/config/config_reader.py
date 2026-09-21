from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class DotEnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=Path(__file__).parent / ".env", extra="allow")
    CHAT_MODEL:str
    SUMMARY_MODEL:str

settings = DotEnvSettings()

if __name__ == "__main__":
    print(settings)

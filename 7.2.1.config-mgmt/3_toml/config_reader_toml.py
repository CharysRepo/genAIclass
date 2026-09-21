from typing import Tuple, Type
from pydantic import BaseModel
from pathlib import Path
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class ModelTomlSettings(BaseModel):
    name: str
    base_url: str
    api_key: str
    temperature: float
    max_tokens: int


class PromptsTomlSettings(BaseModel):
    zero_shot: str


class LlmTomlSettings(BaseModel):
    model: ModelTomlSettings
    prompts: PromptsTomlSettings


class EmbedderTomlSettings(BaseModel):
    model: str
    token: str
    normalize: bool


class TomlSettings(BaseSettings):
    model_config = SettingsConfigDict(toml_file=Path(__file__).parent / "config.toml")

    llm: LlmTomlSettings
    embedder: EmbedderTomlSettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls),)


settings = TomlSettings()

if __name__ == "__main__":
    print(settings)

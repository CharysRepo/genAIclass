from typing import Tuple, Type
from pydantic import BaseModel
from pathlib import Path
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class ModelYamlSettings(BaseModel):
    name: str
    base_url: str
    api_key: str
    temperature: float
    max_tokens: int


class PromptsYamlSettings(BaseModel):
    zero_shot: str


class LlmYamlSettings(BaseModel):
    model: ModelYamlSettings
    prompts: PromptsYamlSettings


class EmbedderYamlSettings(BaseModel):
    model: str
    token: str
    normalize: bool


class YamlSettings(BaseSettings):
    model_config = SettingsConfigDict(yaml_file=Path(__file__).parent / "config.yaml")

    llm: LlmYamlSettings
    embedder: EmbedderYamlSettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)


settings = YamlSettings()

if __name__ == "__main__":
    print(settings)

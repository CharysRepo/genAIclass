from pydantic_ai import Embedder
from config.config_reader import settings

embedder = Embedder(settings.EMBEDDING_MODEL)
from dotenv import load_dotenv
load_dotenv(override=True)
from config.config_reader import settings
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoint import chat_router
import logfire

app = FastAPI()
app.include_router(chat_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.ALLOW_ORIGINS], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_fastapi(app)
logfire.instrument_redis()
logfire.instrument_httpx()

if __name__ == "__main__":
    uvicorn.run(app, host=settings.UVICORN_HOST, port=settings.UVICORN_PORT)
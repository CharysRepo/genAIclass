from dotenv import load_dotenv
load_dotenv(override=True)
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoint import chat_router
import logfire

app = FastAPI()
app.include_router(chat_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_fastapi(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
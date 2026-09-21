from dotenv import load_dotenv
load_dotenv(override=True)
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logfire

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logfire.configure()
logfire.instrument_fastapi(app)

class UserRequest(BaseModel):
    name: str

@app.post("/api/greet")
async def greet_user(user: UserRequest):
    return {"message": f"Hello, {user.name}!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
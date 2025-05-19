import os
import uvicorn
from fastapi import FastAPI, Header, HTTPException, status
from typing import Optional

SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
SERVER_ACCESS_TOKEN = os.getenv("SERVER_ACCESS_TOKEN", "token")

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "FastAPI server is running!"}

@app.get("/secure_data")
async def read_secure_data(x_access_token: Optional[str] = Header(None)):
    if x_access_token != SERVER_ACCESS_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token"
        )
    return {"message": "Secure data accessed."}

if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)

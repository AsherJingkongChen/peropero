import os
import torch

from fastapi import FastAPI, Header, HTTPException, status
from typing import Optional, List
from pydantic import BaseModel

SERVER_ACCESS_TOKEN = os.getenv("SERVER_ACCESS_TOKEN", "token")

app = FastAPI()


class TensorInput(BaseModel):
    data: List[List[float]]


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


@app.post("/tensor/mean")
async def post_tensor_mean(payload: TensorInput):
    device = get_device()
    input_tensor = torch.tensor(payload.data, dtype=torch.float32).to(device)
    mean_value = torch.mean(input_tensor).item()
    return {"item": mean_value, "device": str(device)}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

import torch

from fastapi import FastAPI
from typing import List
from pydantic import BaseModel

app = FastAPI()


class TensorInput(BaseModel):
    data: List


@app.get("/")
async def read_root():
    return {"message": "FastAPI server is running!"}


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

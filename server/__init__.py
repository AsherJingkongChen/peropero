import torch as tch
import logging
import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

app = FastAPI()
logger = logging.getLogger("uvicorn")
logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)


@app.exception_handler(Exception)
async def base_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"{request.method} {request.url} — {exc}\n"
        f"{''.join(traceback.format_tb(exc.__traceback__)[-3:])}"
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )


class TensorInput(BaseModel):
    data: List


@app.get("/")
async def read_root():
    return {"message": "Server is running!"}


@app.post("/tensor/mean")
async def post_tensor_mean(payload: TensorInput):
    if not payload.data:
        raise HTTPException(status_code=400, detail="Input tensor data is empty")
    input_tensor = tch.tensor(payload.data, dtype=tch.float32, device=get_device())
    mean_value = input_tensor.mean()
    return {"item": mean_value.item(), "device": str(input_tensor.device)}


def get_device():
    if tch.cuda.is_available():
        return tch.device("cuda")
    elif tch.mps.is_available():
        return tch.device("mps")
    else:
        return tch.device("cpu")

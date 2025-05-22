import io
import logging
import traceback
import torch as tch
from typing import List
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

app = FastAPI()

logger = logging.getLogger("uvicorn")
logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)

from .noposplat_service import (
    init_noposplat_model,
    reconstruct_scene_from_images,
    get_tch_device_str,
)


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


@app.on_event("startup")
async def startup_event():
    logger.info("Attempting to initialize NoPoSplat model on startup...")
    try:
        init_noposplat_model()
    except Exception as e:
        logger.error(
            f"Failed to initialize NoPoSplat model during startup: {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )


@app.get("/")
async def read_root():
    return {"message": "Server is running!"}


@app.post("/tensor/mean")
async def post_tensor_mean(payload: dict):
    if not payload.get("data"):
        raise HTTPException(status_code=400, detail="Input tensor data is empty")

    current_device_str = get_tch_device_str()
    input_tensor = tch.tensor(
        payload["data"], dtype=tch.float32, device=current_device_str
    )
    mean_value = input_tensor.mean()
    return {"item": mean_value.item(), "device": current_device_str}


@app.post("/reconstruction")
async def reconstruct_scene_endpoint(images: List[UploadFile] = File(...)):
    logger.info(f"Received reconstruction request for {len(images)} images.")
    processed_images = []
    for i, file in enumerate(images):
        try:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents))
            processed_images.append(img)
            logger.info(
                f"Processed image {i + 1}: {file.filename}, format: {img.format}, size: {img.size}"
            )
        finally:
            await file.close()

    if not processed_images:
        raise HTTPException(
            status_code=400, detail="No images provided or failed to process."
        )

    ply_binary_data = reconstruct_scene_from_images(processed_images)
    return Response(
        content=ply_binary_data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=reconstruction.ply"},
    )

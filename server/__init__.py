import os
import shutil
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

from .instantsplat_service import InstantSplatService

# Initialize the service
instantsplat_service = InstantSplatService()


@app.get("/")
async def read_root():
    return {"message": "Server is running!"}


@app.post("/reconstruction")
async def reconstruct_scene_endpoint(images: List[UploadFile] = File(...)):
    logger.info(f"Received reconstruction request for {len(images)} images.")
    
    processed_images = []
    for file in images:
        try:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents))
            processed_images.append(img)
        finally:
            await file.close()

    if not processed_images:
        raise HTTPException(
            status_code=400, detail="No images provided or failed to process."
        )

    try:
        ply_binary_data = instantsplat_service.reconstruct_scene(processed_images)
        return Response(
            content=ply_binary_data,
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=reconstruction.ply"},
        )
    except (RuntimeError, FileNotFoundError) as e:
        logger.error(f"Reconstruction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

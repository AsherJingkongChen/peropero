# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a FastAPI server that provides 3D scene reconstruction using InstantSplat, deployed on Vast.ai cloud instances. The project wraps the InstantSplat neural rendering library to create PLY files from multiple input images.

## Key Architecture

- **FastAPI Service** (`server/__init__.py`): Main HTTP server with reconstruction endpoint
- **InstantSplat Wrapper** (`server/instantsplat_service.py`): Service class that manages the InstantSplat pipeline
- **InstantSplat Submodule** (`server/InstantSplat/`): Git submodule containing the core 3D Gaussian Splatting implementation
- **Deployment Script** (`deploy_locally.sh`): Automated Vast.ai instance creation and setup

## Development Commands

### Local Development Setup
```bash
# Initialize submodules (required first step)
git submodule update --init --recursive

# Download MASt3R checkpoint (required for reconstruction)
cd server/InstantSplat
mkdir -p mast3r/checkpoints/
curl -fsSL https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -o mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
cd ../..

# Install dependencies
uv sync

# Install InstantSplat CUDA modules (GPU required)
uv pip install --no-build-isolation server/InstantSplat/submodules/simple-knn
uv pip install --no-build-isolation server/InstantSplat/submodules/diff-gaussian-rasterization
uv pip install --no-build-isolation server/InstantSplat/submodules/fused-ssim

# Compile RoPE CUDA kernels (optional performance optimization)
cd server/InstantSplat/croco/models/curope/
uv run python setup.py build_ext --inplace
cd ../../../../..
```

### Running the Server
```bash
# Run locally for development
uv run uvicorn server:app --host localhost --port 8888 --reload
```

### Linting
```bash
# Run code formatting and linting
uv run ruff check .
uv run ruff format .
```

## API Endpoints

- `GET /`: Health check endpoint
- `POST /reconstruction`: Main reconstruction endpoint
  - Input: Multiple image files as multipart form data
  - Output: Binary PLY file for 3D visualization
  - Minimum 3 images required

## Reconstruction Pipeline

The `InstantSplatService.reconstruct_scene()` method:
1. Creates temporary job directory with uploaded images
2. Runs InstantSplat geometry initialization (`init_geo.py`)
3. Runs InstantSplat training (`train.py`) with 500 iterations
4. Extracts final PLY file from training output
5. Returns binary PLY data and cleans up temporary files

## Dependencies

- **Core**: FastAPI, uvicorn, PyTorch, torchvision
- **Computer Vision**: Pillow, opencv-python, lpips
- **3D Processing**: trimesh, open3d, plyfile
- **InstantSplat Requirements**: numpy, scipy, einops, matplotlib, omegaconf
- **Development**: ruff (linting)

## GPU Requirements

This project requires CUDA-compatible GPU for:
- InstantSplat CUDA kernels (simple-knn, diff-gaussian-rasterization, fused-ssim)
- PyTorch operations during 3D reconstruction
- RoPE CUDA kernels compilation (optional but recommended)

## Deployment

Use `deploy_locally.sh` for automated Vast.ai deployment which handles:
- Instance creation with CUDA base image
- Repository cloning and submodule initialization
- Dependency installation and CUDA module compilation
- Server startup on port 8888
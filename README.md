# 3D Scene Reconstruction API

A FastAPI server that provides 3D scene reconstruction using InstantSplat, deployed on Vast.ai cloud instances. The service converts multiple input images into 3D Gaussian Splats and returns PLY files for 3D visualization.

## What is InstantSplat?

InstantSplat is a sparse-view 3D scene reconstruction framework using Gaussian Splatting that can reconstruct large-scale scenes from just a few input images in seconds. It supports 3D-GS, 2D-GS, and uses MASt3R for geometry initialization and DUSt3R for stereo matching.

## Scripts

-   `deploy_locally.sh`: Automated Vast.ai deployment script that creates GPU instances, installs dependencies, and starts the FastAPI server

## Prerequisites

-   Python 3.9+, `git`, `uv`
-   Vast.ai CLI (`pip install vastai`)
-   `jq` (for JSON parsing)
-   CUDA-compatible GPU (for local development)

## Quick Start - Cloud Deployment

1.  **Set up Vast.ai CLI**:
    ```bash
    pip install vastai
    vastai set api-key YOUR_API_KEY  # Get from vast.ai account
    ```

2.  **Clone and prepare repository**:
    ```bash
    git clone https://github.com/AsherJingkongChen/peropero.git
    cd peropero
    chmod +x deploy_locally.sh
    ```

3.  **Deploy to Vast.ai**:
    ```bash
    ./deploy_locally.sh
    ```
    - Script will prompt for Vast.ai offer ID (find at [vast.ai/create](https://cloud.vast.ai/create/))
    - Optionally provide SSH public key path for secure access
    - Wait 5-10 minutes for setup to complete

4.  **Access your deployed service**:
    ```bash
    # Use the SSH tunnel command provided by the script
    ssh -L 8888:localhost:8888 root@[SSH_HOST] -p [SSH_PORT]
    
    # Test the service
    curl http://localhost:8888/
    ```

## Project Structure

```
peropero/
├── server/                    # FastAPI application
│   ├── __init__.py           # Main FastAPI server with endpoints
│   ├── instantsplat_service.py # InstantSplat wrapper service
│   └── InstantSplat/         # Git submodule for 3D reconstruction
├── deploy_locally.sh         # Vast.ai deployment automation
├── pyproject.toml           # Python dependencies and project config
└── README.md                # This file
```

## API Endpoints

### `GET /`
Health check endpoint that returns a welcome message.
```bash
curl http://localhost:8888/
```

### `POST /reconstruction`
Main reconstruction endpoint that converts multiple images into a 3D scene.

**Requirements:**
- Minimum 3 images required
- Images uploaded as multipart form data
- Returns binary PLY file for 3D visualization

**Example:**
```bash
curl -X POST \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "images=@image3.jpg" \
  http://localhost:8888/reconstruction \
  -o scene.ply
```

**Process:**
1. Uploads images are saved to temporary directory
2. InstantSplat geometry initialization (`init_geo.py`)
3. InstantSplat training with 500 iterations (`train.py`)
4. Extracts PLY file from training output
5. Returns binary PLY data and cleans up

## Management Commands

**Monitor deployment:**
```bash
vastai logs <INSTANCE_ID>
```

**Destroy instance:**
```bash
vastai destroy instance <INSTANCE_ID>
```

## Local Development Setup

**Requirements:** CUDA-compatible GPU, Python 3.9+, `git`, `uv`

1.  **Initialize submodules:**
    ```bash
    git submodule update --init --recursive
    ```

2.  **Download MASt3R checkpoint:**
    ```bash
    cd server/InstantSplat
    mkdir -p mast3r/checkpoints/
    curl -fsSL https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -o mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
    cd ../..
    ```

3.  **Install dependencies:**
    ```bash
    uv sync --no-build-isolation
    ```

4.  **Install InstantSplat CUDA modules:**
    ```bash
    uv pip install --no-build-isolation server/InstantSplat/submodules/simple-knn
    uv pip install --no-build-isolation server/InstantSplat/submodules/diff-gaussian-rasterization
    uv pip install --no-build-isolation server/InstantSplat/submodules/fused-ssim
    ```

5.  **Compile RoPE CUDA kernels (optional but recommended):**
    ```bash
    cd server/InstantSplat/croco/models/curope/
    uv run python setup.py build_ext --inplace
    cd ../../../../..
    ```

6.  **Start development server:**
    ```bash
    uv run uvicorn server:app --host localhost --port 8888 --reload
    ```

## What the Deployment Script Does

The `deploy_locally.sh` script automates the entire cloud deployment process:

1. **Instance Creation**: Creates a Vast.ai GPU instance with CUDA 12.4 base image
2. **Environment Setup**: Installs `uv`, `git`, and system dependencies
3. **Repository Cloning**: Clones the project with all submodules
4. **Model Download**: Downloads the 1.5GB MASt3R checkpoint file
5. **Dependency Installation**: Installs Python packages and CUDA modules
6. **Server Launch**: Starts the FastAPI server on port 8888
7. **SSH Setup**: Optionally configures SSH key access
8. **Connection Info**: Provides SSH tunnel command for local access

The script handles the entire setup automatically, typically taking 5-10 minutes to complete.

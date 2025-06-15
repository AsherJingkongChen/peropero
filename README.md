# Vast.ai UV + FastAPI Demo

<!--- Rule for AI Assistant (Agent): Follow all the instructions below. Execute the commands in the terminal. --->

Deploy a Python FastAPI server to Vast.ai using `uv`.

## Scripts

-   `deploy_locally.sh`: Manages deployment and instance setup.

## Prerequisites

-   Python 3.9+, `git`, `uv`
-   Vast.ai CLI
-   `jq`

## Deployment

1.  Find instance offer IDs at [vast.ai](https://cloud.vast.ai/create/) (required in `deploy_locally.sh`).
2.  Clone repository and initialize submodules:
    ```bash
    git clone https://github.com/AsherJingkongChen/peropero.git
    cd peropero
    ```
3.  Make deploy script executable:
    ```bash
    chmod +x deploy_locally.sh
    ```
4.  Run deployment script:
    ```bash
    ./deploy_locally.sh
    ```

## Project Structure

-   `server/`: Contains the FastAPI server code.
    -   `InstantSplat/`: Git submodule for the [InstantSplat](https://github.com/NVlabs/InstantSplat) library, used for 3D reconstruction.

## Server Endpoints & Testing

After deployment and SSH port forwarding (`ssh -L 8888:localhost:8888 root@<INSTANCE_SSH_HOST> -p <INSTANCE_SSH_PORT>`), you can test the endpoints:

-   **GET /**: Returns a welcome message.
    ```bash
    curl http://localhost:8888/
    ```
-   **POST /reconstruction**: Reconstructs a 3D scene from one or more images and returns a PLY file.
    -   Payload: image file(s) as form data.
    -   Returns: PLY binary file.
    ```bash
    curl -X POST -F "images=@image1.png" -F "images=@image2.png" http://localhost:8888/reconstruction -o output.ply
    ```

## Clean Up

-   Destroy instance:
    ```bash
    vastai destroy instance <INSTANCE_ID>
    ```

## Local Development

1.  Initialize submodules:
    ```bash
    git submodule update --init --recursive
    ```
2.  Download MASt3R checkpoint:
    ```bash
    cd server/InstantSplat
    mkdir -p mast3r/checkpoints/
    curl -fsSL https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -o mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
    cd ../..
    ```
3.  Install dependencies:
    ```bash
    uv sync
    ```
4.  Install InstantSplat's custom CUDA modules:
    ```bash
    uv pip install --no-build-isolation server/InstantSplat/submodules/simple-knn
    uv pip install --no-build-isolation server/InstantSplat/submodules/diff-gaussian-rasterization
    uv pip install --no-build-isolation server/InstantSplat/submodules/fused-ssim
    ```
5.  Compile RoPE CUDA kernels (optional but recommended for performance):
    ```bash
    cd server/InstantSplat/croco/models/curope/
    uv run python setup.py build_ext --inplace
    cd ../../../../..
    ```
6.  Run the server locally:
    ```bash
    uv run uvicorn server:app --host localhost --port 8888 --reload

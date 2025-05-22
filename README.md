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
    -   `NoPoSplat/`: Git submodule for the [NoPoSplat](https://github.com/cvg/NoPoSplat) library, used for 3D reconstruction.
        -   **Important for CUDA:** Ensure `rmurai0610/diff-gaussian-rasterization-w-pose` (a key CUDA component for NoPoSplat) is correctly installed.

## Server Endpoints & Testing

After deployment and SSH port forwarding (`ssh -L 8888:localhost:8888 root@<INSTANCE_SSH_HOST> -p <INSTANCE_SSH_PORT>`), you can test the endpoints:

-   **GET /**: Returns a welcome message.
    ```bash
    curl http://localhost:8888/
    ```
-   **POST /tensor/mean**: Calculates the mean of a 2D float tensor.
    -   Payload: `{"data": [[<float>, ...], ...]}`
    -   Returns: `{"item": <float>, "device": "<cpu|cuda|mps>"}`
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"data": [[1.0, 2.0], [3.0, 4.0]]}' http://localhost:8888/tensor/mean
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
2.  Download NoPoSplat pretrained model (if not already present):
    ```bash
    mkdir -p server/NoPoSplat/pretrained_weights
    curl -fsSL https://huggingface.co/botaoye/NoPoSplat/resolve/main/re10k.ckpt -o server/NoPoSplat/pretrained_weights/re10k.ckpt
    ```
3.  Install dependencies:
    ```bash
    uv sync
    ```
4.  Install `rmurai0610/diff-gaussian-rasterization-w-pose` (for CUDA):
    ```bash
    uv pip install --no-build-isolation git+https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git
    ```
5.  Run the server locally:
    ```bash
    uv run uvicorn server:app --host localhost --port 8888 --reload
    ```

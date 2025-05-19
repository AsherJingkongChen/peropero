# Vast.ai UV + FastAPI Demo

Deploy a Python FastAPI server to Vast.ai using `uv`.

## Scripts

-   `deploy_locally.sh`: Manages deployment and instance setup.

## Prerequisites

-   Python 3.9+, `git`, `uv`
-   Vast.ai CLI
-   `jq`

## Deployment

1.  Clone repository.
    ```bash
    git clone https://github.com/AsherJingkongChen/vast_ai_uv_fastapi_demo.git
    cd vast_ai_uv_fastapi_demo
    ```
2.  ```bash
    chmod +x deploy_locally.sh
    ```
3.  ```bash
    ./deploy_locally.sh
    ```

## Server Endpoints & Testing

After deployment and SSH port forwarding (`ssh -L 8888:localhost:8888 root@<INSTANCE_SSH_HOST> -p <INSTANCE_SSH_PORT>`), you can test the endpoints:

-   **GET /**: Returns a welcome message.
    ```bash
    curl http://localhost:8888/
    ```
-   **GET /secure_data**: Returns a secure message. Requires a valid `X-Access-Token`.
    (Replace `<YOUR_SERVER_SECRET_TOKEN>` with the actual token).
    ```bash
    curl -H "X-Access-Token: <YOUR_SERVER_SECRET_TOKEN>" http://localhost:8888/secure_data
    ```
-   **POST /tensor/mean**: Calculates the mean of a 2D float tensor.
    -   Payload: `{"data": [[<float>, ...], ...]}`
    -   Returns: `{"item": <float>, "device": "<cpu|cuda|mps>"}`
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"data": [[1.0, 2.0], [3.0, 4.0]]}' http://localhost:8888/tensor/mean
    ```

## Clean Up

-   Destroy instance:
    ```bash
    vastai destroy instance <INSTANCE_ID>
    ```

## Local Development

1.  ```bash
    uv sync
    ```
2.  Run the server locally:
    ```bash
    SERVER_ACCESS_TOKEN='token' uv run uvicorn server:app --host localhost --port 8888 --reload
    ```

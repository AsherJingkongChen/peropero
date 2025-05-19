# Vast.ai UV + FastAPI Demo

Deploy a Python FastAPI server to Vast.ai using `uv`.

## Scripts

*   `deploy_locally.sh`: Manages deployment and instance setup.

## Prerequisites

*   Git, Python 3.9+, `uv`
*   Vast.ai CLI
*   `jq`

## Deployment

1.  Clone repository.
2.  ```bash
    chmod +x deploy_locally.sh
    ```
3.  ```bash
    ./deploy_locally.sh
    ```
    (Follow prompts)

## Test Server (After Deployment)

1.  SSH Port Forward (details from `deploy_locally.sh`):
    ```bash
    ssh -L 8888:localhost:8000 root@<INSTANCE_SSH_HOST> -p <INSTANCE_SSH_PORT>
    ```
2.  Test:
    ```bash
    curl http://localhost:8888/
    ```
    ```bash
    curl -H "X-Access-Token: <YOUR_SERVER_SECRET_TOKEN>" http://localhost:8888/secure_data
    ```

## Clean Up

*   Destroy instance:
    ```bash
    vastai destroy instance <INSTANCE_ID>
    ```

## Local Development

1.  ```bash
    uv sync
    ```
2.  ```bash
    SERVER_ACCESS_TOKEN='token' uv run uvicorn server:app
    ```
3.  ```bash
    curl http://127.0.0.1:8000/
